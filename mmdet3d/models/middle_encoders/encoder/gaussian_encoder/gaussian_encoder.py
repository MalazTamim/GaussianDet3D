from typing import List, Optional
import torch, torch.nn as nn

# from mmseg.registry import MODELS
# from mmengine import build_from_cfg
from mmcv.utils import build_from_cfg #This shoudl work also: from mmdet.utils import build_from_cfg

from ..base_encoder import BaseEncoder



# @MODELS.register_module()

# from ....builder import VOXEL_ENCODERS
from ....builder import MIDDLE_ENCODERS
@MIDDLE_ENCODERS.register_module()
class GaussianOccEncoder(BaseEncoder):
    def __init__(
        self,
        anchor_encoder: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        mid_refine_layer: dict = None,
        spconv_layer: dict = None,#SparseConv3D
        num_decoder: int = 6,
        operation_order: Optional[List[str]] = None,
        freeze_encoder: bool = False,  # ✅ added flag
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder
        self.freeze_encoder = freeze_encoder


        if operation_order is None:
            operation_order = [
                "spconv",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        self.anchor_encoder = build(anchor_encoder, MIDDLE_ENCODERS)
        self.op_config_map = {
            "norm": [norm_layer, MIDDLE_ENCODERS],
            "ffn": [ffn, MIDDLE_ENCODERS],
            "deformable": [deformable_model, MIDDLE_ENCODERS],
            "refine": [refine_layer, MIDDLE_ENCODERS],
            "mid_refine":[mid_refine_layer, MIDDLE_ENCODERS],
            "spconv": [spconv_layer, MIDDLE_ENCODERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        # ✅ Freeze encoder if requested
        if self.freeze_encoder:
            print("[INFO] Freezing GaussianOccEncoder parameters")
            for param in self.parameters():
                param.requires_grad = False
        
    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        representation,
        rep_features,
        ms_img_feats=None,
        metas=None,
        lidar2img=None,
        **kwargs
    ):
        # print("representation    GAUSSIAN encoder   ", representation)
        '''
        print("beginning of gaussian encoder forward pass")
        print("metas at gaussian encoder forwad pass beginning:")
        print(metas)
        print("ms_img_feats shape aft gaussian encoder forwad pas beginning:",len(ms_img_feats) )
        '''

        feature_maps = ms_img_feats
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        instance_feature = rep_features
        anchor = representation
        '''
        print(f"Anchor Shape: {anchor.shape}")#torch.Size([1, 6400, 28])
        print("representation    GAUSSIAN encoder   ", representation)
        print("mid of gaussian encoder forward pass")
        '''



        
        # This 28 comes from the gaussian properties
        # means: 3
        # scales: 3
        # rotations: 4
        # opacities (semantics): c=17
        # origi_opa: 1
        # Note we do not have covariance here
        anchor_embed = self.anchor_encoder(anchor)
        # print("hereeee")
        # print(f"Anchor Embedding Shape: {anchor_embed.shape}")

        prediction = []
        for i, op in enumerate(self.operation_order):
            if op == 'spconv':
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor)
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                    lidar2img=lidar2img
                )
            elif "refine" in op:
                anchor, gaussian = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                )
            
                prediction.append({'gaussian': gaussian})
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
            else:
                raise NotImplementedError(f"{op} is not supported.")
        # print("end of gaussian encoder forward pass")
        # print("prediction at gaussian encoder forward pass end:")
        # print(prediction)
        return {"representation": prediction}