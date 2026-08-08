'''
    We have some issues here
    -   project_mat is fixed
    -   image_Wh is fixed

'''



# from mmengine.registry import MODELS
# from mmengine.model import BaseModule
from mmcv.utils import build_from_cfg #This shoudl work also: from mmdet.utils import build_from_cfg

# from mmengine import build_from_cfg
# from mmengine.model import xavier_init, constant_init
from mmcv.cnn import xavier_init, constant_init

import torch, torch.nn as nn
import numpy as np
from typing import List, Optional
from ....utils.safe_ops import safe_sigmoid
from ....utils.utils import get_rotation_matrix
from .utils import linear_relu_ln
try:
    from .ops import DeformableAggregationFunction as DAF
except:
    DAF = None


# @MODELS.register_module()
# from ....builder import VOXEL_ENCODERS
# @VOXEL_ENCODERS.register_module()
from ....builder import MIDDLE_ENCODERS
@MIDDLE_ENCODERS.register_module()
class SparseGaussian3DKeyPointsGenerator(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        learnable_fixed_scale=1,
        fix_scale=None,
        pc_range=None,
        scale_range=None,
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        **kwargs,
    ):
        super(SparseGaussian3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        self.learnable_fixed_scale = learnable_fixed_scale
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = np.array(fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 3)

        self.pc_range = pc_range
        self.scale_range = scale_range
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
    ):
        bs, num_anchor = anchor.shape[:2]
        fix_scale = anchor.new_tensor(self.fix_scale)
        scale = fix_scale[None, None].tile([bs, num_anchor, 1, 1])
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                safe_sigmoid(self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3))
                - 0.5
            )
            scale = torch.cat([scale, learnable_scale * self.learnable_fixed_scale], dim=-2)
        
        gs_scales = anchor[..., None, 3:6]
        if self.scale_act == "sigmoid":
            gs_scales = safe_sigmoid(gs_scales)
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales

        key_points = scale * gs_scales
        rots = anchor[..., 6:10]
        rotation_mat = get_rotation_matrix(rots).transpose(-1, -2)
        
        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1)

        xyz = anchor[..., :3]
        if self.xyz_act == 'sigmoid':
            xyz = safe_sigmoid(xyz)
        
        xxx = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        yyy = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        zzz = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        xyz = torch.stack([xxx, yyy, zzz], dim=-1)
        
        key_points = key_points + xyz.unsqueeze(2)
        return key_points


# @MODELS.register_module()
# @VOXEL_ENCODERS.register_module()
# from ....builder import MIDDLE_ENCODERS
@MIDDLE_ENCODERS.register_module()
class DeformableFeatureAggregation(nn.Module):#BaseModule
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_deformable_func = use_deformable_func and DAF is not None
        assert self.use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator,MIDDLE_ENCODERS )#MODELS
        self.num_pts = self.kps_generator.num_pts
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        # img_metas=None, 
        lidar2img: List[torch.Tensor],
        **kwargs: dict,
    ):
        '''
        print("metas at the start of forward deformable", metas)
        print("lidar2img matrix :", lidar2img)  # Debugging
        '''

        # Build projection_mat as (B, num_cams, 4, 4) for any batch size
        if isinstance(lidar2img, list) and isinstance(lidar2img[0], list):
            # Format: List[B][List[num_cams][Tensor(4,4)]]
            projection_mat = torch.stack(
                [torch.stack(mats, dim=0) for mats in lidar2img], dim=0
            )  # (B, num_cams, 4, 4)
        else:
            # Format: List[num_cams][Tensor(B, 4, 4)]
            projection_mat = torch.stack(lidar2img, dim=1)  # (B, num_cams, 4, 4)
        projection_mat = projection_mat.to(dtype=torch.float32)
        # Squeeze any spurious extra dims introduced by test-time collation (e.g. (B,n,1,4,4) -> (B,n,4,4))
        while projection_mat.ndim > 4:
            projection_mat = projection_mat.squeeze(2)
        # print("projection_mat matrix after the 3 steps :", projection_mat)  # Debugging
        # print("projection_mat shape after the 3 steps:", projection_mat.shape)  # Debugging
        
        


        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        temp_key_points_list = (
            feature_queue
        ) = meta_queue = temp_anchor_embeds = []
        if self.use_deformable_func:
            feature_maps = DAF.feature_maps_format(feature_maps)
        

        bs, num_anchor = instance_feature.shape[:2]

        for (
            temp_feature_maps,
            temp_metas,
            temp_key_points,
            temp_anchor_embed,
        ) in zip(
            feature_queue[::-1] + [feature_maps],
            meta_queue[::-1] + [metas],
            temp_key_points_list[::-1] + [key_points],
            temp_anchor_embeds[::-1] + [anchor_embed],
        ):
            weights, weight_mask = self._get_weights(
                instance_feature, temp_anchor_embed, metas, projection_mat=projection_mat
            )
            if self.use_deformable_func:
                weights = (
                    weights.permute(0, 1, 4, 2, 3, 5)
                    .contiguous()
                    .reshape(
                        bs,
                        num_anchor,
                        self.num_pts,
                        self.num_cams,
                        self.num_levels,
                        self.num_groups,
                    )
                )
                weight_mask = (
                    weight_mask.permute(0, 1, 4, 2, 3, 5)
                    .contiguous()
                    .reshape(
                        bs,
                        num_anchor,
                        self.num_pts,
                        self.num_cams,
                        self.num_levels,
                        self.num_groups,
                    )
                )
               

                # projection_mat = torch.stack(lidar2img[0], dim=0).to(dtype=torch.float32, device=instance_feature.device)
                # projection_mat = torch.stack(lidar2img[0], dim=0).unsqueeze(0).to(dtype=torch.float32, device=instance_feature.device)
                # projection_mat = lidar2img
                # projection_mat = torch.stack(projection_mat, dim=0)  # Stack the list into a tensor
                '''
                print("projection_mat in deformable module:", projection_mat)
                '''


            
                # projection_mat = torch.from_numpy(projection_mat).to(instance_feature.device)
                # projection_mat = torch.tensor(projection_mat, dtype=torch.float32, device=instance_feature.device)
                '''
                print("projection_mat   ",projection_mat)
                print("temp_metas     ",temp_metas)
                print("temp_key_points   ",temp_key_points.shape)
                '''

                # print("temp_metas of projection_mat   ",temp_metas["lidar2img"])
                
                # print("temp_metas of projection_mat   ",temp_metas["projection_mat"])


                points_2d, mask = self.project_points(
                    temp_key_points,
                    # temp_metas["projection_mat"],
                    projection_mat,

                    # temp_metas.get("image_wh"),
                    # torch.tensor([1600, 900], dtype=torch.float32, device=instance_feature.device)
                     torch.tensor([[[1600.,  864.],#image_wh =
                          [1600.,  864.],
                          [1600.,  864.],
                          [1600.,  864.],
                          [1600.,  864.],
                          [1600.,  864.]]], device=instance_feature.device)

                )
                # print("Before permute, points_2d.shape:", points_2d.shape)

                points_2d = points_2d.permute(0, 2, 3, 1, 4).reshape(
                    bs, num_anchor * self.num_pts, self.num_cams, 2)
                mask = mask.permute(0, 2, 3, 1)
                mask = mask[..., None, None] & weight_mask
                all_miss = mask.sum(dim=[2, 3, 4], keepdim=True) == 0
                all_miss = all_miss.expand(-1, -1, self.num_pts, self.num_cams, self.num_levels, -1)
                weights[~mask] = - float('inf')
                weights[all_miss] = 0.
                weights = weights.flatten(2, 4).softmax(dim=-2).reshape(
                    bs,
                    num_anchor * self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups)
                # weights_clone = weights.detach().clone()
                # weights_clone[~all_miss.flatten(1, 2)] = 0.
                # weights = weights - weights_clone
                weights = weights * (1 - all_miss.flatten(1, 2).float())

                temp_features_next = DAF.apply(
                    *temp_feature_maps, points_2d, weights
                ).reshape(bs, num_anchor, self.num_pts, self.embed_dims)
            else:
                temp_features_next = self.feature_sampling(
                    temp_feature_maps,
                    temp_key_points,
                    temp_metas["projection_mat"],
                    temp_metas.get("image_wh"),
                )
                temp_features_next = self.multi_view_level_fusion(
                    temp_features_next, weights
                )

            features = temp_features_next

        features = features.sum(dim=2)  # fuse multi-point features
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None, projection_mat=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            # print("metas    ", metas)
        #     projection_mat= np.stack([[[[-1.2610e+03,  7.5817e+02,  5.3317e+01, -5.5829e+02],
        #   [-1.6372e+01,  5.0177e+02, -1.2271e+03, -7.5858e+02],
        #   [-1.0706e-02,  9.9844e-01,  5.4747e-02, -7.3323e-01],
        #   [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        #  [[-1.4367e+01,  1.4773e+03,  9.2768e+01, -8.6603e+02],
        #   [ 3.5015e+02,  3.0154e+02, -1.2405e+03, -7.2267e+02],
        #   [ 8.4382e-01,  5.3561e-01,  3.3064e-02, -7.3843e-01],
        #   [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        #  [[-1.3506e+03, -5.9594e+02, -1.3219e+01, -3.1915e+02],
        #   [-3.5832e+02,  3.0151e+02, -1.2390e+03, -7.4173e+02],
        #   [-8.2274e-01,  5.6694e-01,  4.1020e-02, -7.4109e-01],
        #   [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        #  [[ 7.9043e+02, -7.4823e+02, -3.6085e+01, -6.8996e+02],
        #   [-1.0412e+01, -4.0911e+02, -8.1359e+02, -6.3152e+02],
        #   [-8.6229e-03, -9.9919e-01, -3.9353e-02, -9.2870e-01],
        #   [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        #  [[-3.2954e+02, -1.4344e+03, -4.8320e+01, -6.4224e+01],
        #   [-4.2843e+02, -9.0956e+01, -1.2526e+03, -5.4591e+02],
        #   [-9.4758e-01, -3.1950e-01,  3.0871e-03, -4.3207e-01],
        #   [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        #  [[ 1.1938e+03,  8.5696e+02,  5.4646e+01, -6.1784e+02],
        #   [ 4.1207e+02, -1.0874e+02, -1.2501e+03, -5.3912e+02],
        #   [ 9.2383e-01, -3.8278e-01, -3.3915e-03, -3.9997e-01],
        #   [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]],)

            # Convert to np.float32 as done in input_dict["projection_mat"]/commented in karte may
            # projection_mat = np.float32(projection_mat)

             # Convert the NumPy array to a PyTorch tensor on the correct device
            device = instance_feature.device
            cam_input = projection_mat[:, :, :3].reshape(bs, self.num_cams, -1)
            # cam_input = torch.from_numpy(cam_input).to(device)  # float32 by default
            camera_embed = self.camera_encoder(cam_input)


            # camera_embed = self.camera_encoder(
            #     # metas["projection_mat"][:, :, :3].reshape(
            #     #     bs, self.num_cams, -1
            #     # )
            #     projection_mat[:, :, :3].reshape(
            #         bs, self.num_cams, -1
            #     )
            # )
            feature = feature[:, :, None] + camera_embed[:, None]
        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            # .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            # mask = torch.rand(
            #     bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            # )
            # mask = mask.to(device=weights.device, dtype=weights.dtype)
            # weights = ((mask > self.attn_drop) * weights) / (
            #     1 - self.attn_drop
            # )
            mask = torch.rand_like(weights)
            mask = mask > self.attn_drop
        else:
            mask = torch.ones_like(weights) > 0
        return weights, mask

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        depth = points_2d[..., 2]
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        if image_wh is not None:
            # print("image_wh ",image_wh)
            points_2d = points_2d / image_wh[:, :, None, None]
        mask = (depth > 1e-5) & (points_2d[..., 0] > 0) & (points_2d[..., 0] < 1) & \
                                (points_2d[..., 1] > 0) & (points_2d[..., 1] < 1)
        return points_2d, mask

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d, _ = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features

