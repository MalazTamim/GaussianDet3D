import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
from mmdet.models import build_detector
from mmdet3d.models import builder
from mmdet3d.ops import scatter_v2, Voxelization
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential

@DETECTORS.register_module()
class GSFormerFSDV2(SingleStage3DDetector):
    """
    A unified detector that:
      1) Uses GSFormer (image-based) to produce 3D Gaussian 'points' on the fly.
      2) Feeds these points to an FSDV2 pipeline to do 3D detection.
      3) Trains both parts (GSFormer + FSDV2) end-to-end.
    """

    def __init__(self,
                 gsformer,          # dict config for your GSFormer submodule
                 voxel_layer=None,  # config for voxelization, if needed
                 voxel_encoder=None,
                 middle_encoder=None,
                 backbone=None,     # spconv-based 3D backbone for FSDV2
                 bbox_head=None,    # FSDV2 head config
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        """
        Args:
            gsformer (dict): Config for the image-based GSFormer sub-detector 
                             (which outputs x,y,z,semantics, etc.).
            voxel_layer (dict): Voxelization configuration.
            voxel_encoder (dict): Configuration for your voxel-encoder used in FSDV2.
            middle_encoder (dict): Middle encoder config (spconv middle layers, if any).
            backbone (dict): 3D backbone config (spconv).
            bbox_head (dict): 3D detection head config (FSDV2 head).
            train_cfg (dict): Training config.
            test_cfg (dict): Testing config.
            pretrained (str, optional): Path to pretrained weights.
            init_cfg (dict, optional): Model initialization config.
        """
        super(GSFormerFSDV2, self).__init__(
            backbone=backbone,
            neck=None,           # FSDV2 commonly keeps neck=None or a second stage
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg
        )

        # 1) Build the GSFormer submodule 
        #    (this is your standard image-based "detector/segmentor" that 
        #     produces a 3D Gaussian representation from images)
        self.gsformer = build_detector(gsformer)

        # 2) Additional voxelization pieces (the "FSDV2" pipeline):
        if voxel_layer is not None:
            self.voxel_layer = Voxelization(**voxel_layer)

        if voxel_encoder is not None:
            self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
            self.point_cloud_range = voxel_encoder['point_cloud_range']
            self.voxel_size = voxel_encoder['voxel_size']  # sometimes called 'virtual_voxel_size' in your snippet

        if middle_encoder is not None:
            self.middle_encoder = builder.build_middle_encoder(middle_encoder)

        # By the time we get here, self.backbone and self.bbox_head are already set 
        # by SingleStage3DDetector's constructor, if 'backbone'/'bbox_head' dicts are given.

        # Some placeholders and extra state:
        self.runtime_info = dict()  # if you want to pass around ephemeral data
        self.train_cfg = train_cfg if train_cfg else {}
        self.test_cfg = test_cfg if test_cfg else {}
        self.print_info = {}

    @force_fp32()
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """
        Forward training:
          1) GSFormer uses images -> produce 3D "Gaussian" points (x, y, z, ...).
          2) We treat those 'points' as input to an FSDV2-like pipeline:
             voxelize -> spconv backbone -> head -> detection losses.
          3) Combine GSFormer losses + FSDV2 losses for a single end-to-end backprop.
        """
        losses = {}

        # -------------------------
        # 1) Run the GSFormer submodule
        #    We assume self.gsformer returns a dict containing:
        #    {
        #      'gauss_points': <FloatTensor, shape = [N, D], e.g. x, y, z, opacity, semantics>,
        #      'batch_idx': <LongTensor, shape = [N]>,
        #      'losses': <dict of GSFormer-specific losses, e.g. feature matching, etc.>,
        #      ... possibly more ...
        #    }
        #    Exactly how you name or structure them depends on your GSFormer code.
        # -------------------------
        gsformer_out = self.gsformer(
            img=img,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            # if your GSFormer’s forward requires more arguments, add them
            **kwargs
        )

        # Collect the GSFormer’s losses if it returns them in a dict
        if 'losses' in gsformer_out:
            losses.update(gsformer_out['losses'])

        # The new "points" from the GSFormer (treated like a LiDAR cloud)
        # We assume you have x,y,z in columns [0..2], plus possibly an intensity or 
        # occupancy channel, plus semantic/logit channels, etc.
        # For example shape = [N, 6 or 7], or any arrangement that your voxel_encoder expects.
        gauss_points = gsformer_out['gauss_points']
        batch_inds = gsformer_out['batch_idx']  # which frame each row belongs to

        # Some users store everything in a single (N, C) array and rely on 
        # voxel_encoder to parse out features. 
        # E.g. gauss_points[:,:3] = xyz, gauss_points[:,3] = "opacity", 
        #      gauss_points[:,4:] = "semantics" or so.
        #
        # If your FSDV2 pipeline requires [x, y, z, <extra feats>], 
        # you can keep them all in gauss_points.

        # -------------------------
        # 2) Now do the FSDV2-like pipeline using gauss_points
        #    This is analogous to your "points" flow from standard LiDAR
        #    but now we just replaced them with the GSFormer's output.
        # -------------------------

        # a) Possibly clip the points to the valid range
        gauss_points = self.clip_points(gauss_points, self.point_cloud_range)

        # b) Dynamic voxelization
        coors = self.voxelize_with_batch_idx(gauss_points, batch_inds)

        # c) Voxel encoder
        #    The assumption: your voxel_encoder can accept cat([xyz, feats]) 
        #    as input. 
        voxel_encoder_in = gauss_points  # or cat() if you have batch_inds separate
        voxel_feats, voxel_coors, unq_inv = self.voxel_encoder(
            voxel_encoder_in, coors, return_inv=True
        )

        # d) Middle encoder, if you have it
        if hasattr(self, 'middle_encoder') and self.middle_encoder is not None:
            voxel_feats, voxel_coors, _ = self.middle_encoder(voxel_feats, voxel_coors, None)

        # e) Forward the backbone (spconv)
        batch_size = voxel_coors[:, 0].max().item() + 1
        out_voxel_feats, out_coors, sparse_shape = self.backbone(
            voxel_feats, voxel_coors, batch_size
        )

        # f) Optionally compute the 3D voxel center coordinate
        device = out_voxel_feats.device
        voxel_size = torch.tensor(self.voxel_size, device=device)
        pc_range = torch.tensor(self.point_cloud_range, device=device)
        # out_coors is shape [M, 4], columns = [batch_idx, z, y, x]
        # so the real-world center is (z + 0.5)*sz, (y + 0.5)*sy, ...
        voxel_centers = (out_coors[:, [3, 2, 1]] + 0.5) * voxel_size[None, :] + pc_range[None, :3]

        # g) Forward the FSDV2 head
        outs = self.bbox_head(out_voxel_feats)
        # outs might have:
        #  {
        #    'cls_logits': shape [M, num_classes],
        #    'reg_preds': shape [M, box_code_size],
        #    'iou_logits': optional
        #  }

        # h) Compute detection losses
        #    We pass in (cls_logits, reg_preds, voxel_centers, ...) plus GT
        loss_inputs = (outs['cls_logits'], outs['reg_preds']) + (
            voxel_centers, out_coors[:, 0]
        ) + (gt_bboxes_3d, gt_labels_3d, img_metas)

        det_loss = self.bbox_head.loss(
            *loss_inputs,
            iou_logits=outs.get('iou_logits', None),
            gt_bboxes_ignore=gt_bboxes_ignore
        )
        losses.update(det_loss)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        """
        Inference without test-time augmentation:
          1) GSFormer -> produce 3D Gaussian points from images
          2) Treat them as a point cloud -> FSDV2 pipeline -> final boxes
        """
        # 1) Get 3D gaussians from GSFormer
        gsformer_out = self.gsformer.simple_test(img, img_metas, **kwargs)
        # Suppose gsformer_out returns a dict with 'gauss_points' and 'batch_idx'
        gauss_points = gsformer_out['gauss_points']
        batch_inds = gsformer_out['batch_idx']

        # 2) The FSDV2 detection steps
        gauss_points = self.clip_points(gauss_points, self.point_cloud_range)
        coors = self.voxelize_with_batch_idx(gauss_points, batch_inds)

        voxel_feats, voxel_coors, _ = self.voxel_encoder(gauss_points, coors)
        if hasattr(self, 'middle_encoder') and self.middle_encoder is not None:
            voxel_feats, voxel_coors, _ = self.middle_encoder(voxel_feats, voxel_coors, None)

        batch_size = voxel_coors[:, 0].max().item() + 1
        out_voxel_feats, out_coors, sparse_shape = self.backbone(
            voxel_feats, voxel_coors, batch_size
        )

        # get voxel centers
        device = out_voxel_feats.device
        voxel_size = torch.tensor(self.voxel_size, device=device)
        pc_range = torch.tensor(self.point_cloud_range, device=device)
        voxel_centers = (out_coors[:, [3, 2, 1]] + 0.5) * voxel_size[None, :] + pc_range[None, :3]

        # detection head forward
        outs = self.bbox_head(out_voxel_feats)

        # get_bboxes -> final boxes/scores/labels
        bbox_list = self.bbox_head.get_bboxes(
            outs['cls_logits'],
            outs['reg_preds'],
            voxel_centers,
            out_coors[:, 0],
            img_metas,
            iou_logits=outs.get('iou_logits', None)
        )

        # convert to mmdet3d-style results
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def clip_points(self, points, pc_range):
        """
        Clamp X, Y, Z to the valid point_cloud_range. 
        We assume points[:, :3] are the coordinates.
        """
        eps = 1e-5
        points[:, 0] = points[:, 0].clamp(pc_range[0] + eps, pc_range[3] - eps)
        points[:, 1] = points[:, 1].clamp(pc_range[1] + eps, pc_range[4] - eps)
        points[:, 2] = points[:, 2].clamp(pc_range[2] + eps, pc_range[5] - eps)
        return points

    @torch.no_grad()
    @force_fp32()
    def voxelize_with_batch_idx(self, points, batch_idx):
        """
        Dynamic voxelization, identical to your FSDV2 snippet.
        points: (N, >=3) – we assume columns 0..2 are x,y,z
        batch_idx: (N,)
        Returns coors: (N, 4) = [batch_ind, z_coor, y_coor, x_coor].
        """
        device = points.device
        voxel_size = torch.tensor(self.voxel_size, device=device)
        pc_range = torch.tensor(self.point_cloud_range, device=device)

        # floor((points - pc_range[:3]) / voxel_size)
        res_coors = torch.div(points[:, :3] - pc_range[None, :3],
                              voxel_size[None, :],
                              rounding_mode='floor').long()
        # reorder to z, y, x
        res_coors = res_coors[:, [2, 1, 0]]
        coors_batch = torch.cat([batch_idx[:, None], res_coors], dim=1)
        return coors_batch

    def aug_test(self, *args, **kwargs):
        """Optional: implement test-time augmentation if you need it."""
        raise NotImplementedError
