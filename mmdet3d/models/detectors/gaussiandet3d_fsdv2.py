import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
# from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
# from .. import builder
from mmdet3d.models import builder

from .centerpoint import CenterPoint
# from mmseg.models import SEGMENTORS
# from mmseg.models import build_backbone
import numpy as np
# from .base_segmentor import CustomBaseSegmentor
# import torch, time
# from mmdet3d.models.detectors.pointpillars import PointPillars
import os
# @SEGMENTORS.register_module()

import matplotlib.pyplot as plt

@DETECTORS.register_module()
class GaussianFormerFSDV2(CenterPoint): 
    def __init__(
        self,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_lifter=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        # use_post_fusion=False,
        fsdv2_cfg=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

         # Initialize FSDV2 detector
        if fsdv2_cfg is not None:
            "fsdv2_cfg is not None - Initializing it with the config now"
            self.fsdv2 =  builder.build_detector(fsdv2_cfg)
        else:
            raise ValueError("FSDV2 configuration must be provided.")


        # self.fp16_enabled = False
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices
        # self.use_post_fusion = use_post_fusion

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if freeze_lifter:
            self.pts_voxel_encoder.requires_grad_(False)
            if hasattr(self.pts_voxel_encoder, "random_anchors"):
                self.pts_voxel_encoder.random_anchors.requires_grad = True
        if extra_img_backbone is not None:
            self.extra_img_backbone = build_backbone(extra_img_backbone)

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        result = {}
         # Convert list of tensors to a single tensor if imgs is a list
        if isinstance(imgs, list):
            imgs = torch.stack(imgs, dim=0)  # Stack along batch dimension
            # print(f"Shape of imgs after stacking: {imgs.shape}")

           # Handle unexpected shapes
        if imgs.ndim == 6:  # Example: (B, 1, N, C, H, W)
            imgs = imgs.squeeze(1)  # Remove the extra dimension

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        img_feats = self.img_neck(img_feats)
        if isinstance(img_feats, dict):
            secondfpn_out = img_feats["secondfpn_out"][0]
            BN, C, H, W = secondfpn_out.shape
            secondfpn_out = secondfpn_out.view(B, int(BN / B), C, H, W)
            img_feats = img_feats["fpn_out"]
            result.update({"secondfpn_out": secondfpn_out})

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        result.update({'ms_img_feats': img_feats_reshaped})
        return result
    
    def forward_extra_img_backbone(self, imgs, **kwargs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.extra_img_backbone(imgs)

        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W))
        return img_feats_backbone_reshaped

    # PCR = [-50, -50, -4.99, 50, 50, 2.99] — matches offline PointsRangeFilter
    _PCR = [-50.0, -50.0, -4.99, 50.0, 50.0, 2.99]

    def _apply_pcr_filter(self, points_list):
        pcr = self._PCR
        filtered = []
        for pts in points_list:
            xyz = pts[:, :3]
            mask = (
                (xyz[:, 0] >= pcr[0]) & (xyz[:, 0] <= pcr[3]) &
                (xyz[:, 1] >= pcr[1]) & (xyz[:, 1] <= pcr[4]) &
                (xyz[:, 2] >= pcr[2]) & (xyz[:, 2] <= pcr[5])
            )
            filtered.append(pts[mask])
        return filtered

    def forward_train(self,
                img=None,
                metas=None,#img_metas
                img_metas=None,
                points=None,
                extra_backbone=False,
                occ_only=False,
                rep_only=False,
                **kwargs,
        ):
        """Forward training function.
        """
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=img)
        
        results = {
            'imgs': img,
            'metas': metas,
            'img_metas': img_metas,#you could delete bro
            'points': points
        }

        results.update(kwargs)
        outs = self.extract_img_feat(**results)
        results.update(outs)

        outs = self.pts_voxel_encoder(**results)
        results.update(outs)

        outs = self.pts_middle_encoder(**results)
        
        if rep_only:
            return outs['representation']
        results.update(outs)
        
        # print("results[representation][-1]", results["representation"][-1])

        device = results['imgs'].device
        # print("Device used for imgs: ", device)
        # Extract Gaussian representation and convert to pseudo point cloud
        if "representation" in results:
            gaussians = results["representation"][-1]  # Use the last Gaussian representation
            if isinstance(gaussians, dict) and "gaussian" in gaussians:
                means = gaussians["gaussian"].means.to(device)       # (B, N, 3)
                scales = gaussians["gaussian"].scales.to(device)     # (B, N, 3)
                rots = gaussians["gaussian"].rotations.to(device)    # (B, N, 4)
                opas = gaussians["gaussian"].opacities.to(device)    # (B, N, 1)
                sems = gaussians["gaussian"].semantics.to(device)    # (B, N, sem_dim)

                assert means.shape[1] == scales.shape[1] == rots.shape[1] == opas.shape[1] == sems.shape[1], \
                    "Gaussian field shapes do not match!"
                # Combine Gaussian properties into a pseudo point cloud: (B, N, 28)
                pseudo_points = torch.cat([means, opas, scales, rots, sems], dim=-1)
                # Break autograd graph before feeding into external detector
                pseudo_points = pseudo_points.detach()
                # FSDV2 expects a list of per-sample point clouds, one tensor per batch item
                results["points"] = [pseudo_points[i] for i in range(pseudo_points.shape[0])]

                # print("Pseudo points shape:", pseudo_points.shape)  # Should be (N, 28)

        results["points"] = self._apply_pcr_filter(results["points"])

        # Forward pass through FSDV2
        fsdv2_losses = self.fsdv2.forward_train(points=results["points"], 
            img_metas=img_metas,
            gt_bboxes_3d = results["gt_bboxes_3d"],  
            gt_labels_3d = results["gt_labels_3d"])
        # print(f"fsdv2_losses: {fsdv2_losses}")
        # print(f"After fsdv2: {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in outs.items()]}")

        # Ensure losses are in the correct format
        if isinstance(fsdv2_losses, dict):
            # Convert dictionary of losses to a format the base detector can handle
            losses = {}
            for k, v in fsdv2_losses.items():
                if isinstance(v, (list, tuple)):
                    losses[k] = torch.stack(v).mean()
                elif isinstance(v, torch.Tensor):
                    losses[k] = v.mean()
                else:
                    losses[k] = torch.tensor(v, device=img.device)
            return losses
        else:
            return fsdv2_losses

    def forward_test(self,
                 img=None,
                 metas=None,
                 points=None,
                 extra_backbone=False,
                 **kwargs):
        """Forward function for testing."""
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=img)

        results = {
            'imgs': img,
            'metas': metas,
            'points': points
        }
        results.update(kwargs)

        # Extract image features
        outs = self.extract_img_feat(**results)
        results.update(outs)

        # Pass through voxel encoder
        outs = self.pts_voxel_encoder(**results)
        results.update(outs)

        # Pass through middle encoder
        outs = self.pts_middle_encoder(**results)
        results.update(outs)

        if "representation" in results:
            gaussians = results["representation"][-1]  # Use the last Gaussian representation
            if isinstance(gaussians, dict) and "gaussian" in gaussians:
                means = gaussians["gaussian"].means                        # (B, N, 3)
                scales = gaussians["gaussian"].scales.to(means.device)     # (B, N, 3)
                rots = gaussians["gaussian"].rotations.to(means.device)    # (B, N, 4)
                opas = gaussians["gaussian"].opacities.to(means.device)    # (B, N, 1)
                sems = gaussians["gaussian"].semantics.to(means.device)    # (B, N, sem_dim)

                assert means.shape[1] == scales.shape[1] == rots.shape[1] == opas.shape[1] == sems.shape[1], \
                    "Gaussian field shapes do not match!"

                pseudo_points = torch.cat([means, opas, scales, rots, sems], dim=-1)  # (B, N, 28)
                # FSDV2 expects a list of per-sample point clouds, one tensor per batch item
                results["points"] = [pseudo_points[i] for i in range(pseudo_points.shape[0])]

        results["points"] = self._apply_pcr_filter(results["points"])

        # Forward pass through FSDV2 for predictions
        predictions = self.fsdv2.simple_test(
            points=results["points"],
            img_metas=results["img_metas"],
        )
        return predictions


