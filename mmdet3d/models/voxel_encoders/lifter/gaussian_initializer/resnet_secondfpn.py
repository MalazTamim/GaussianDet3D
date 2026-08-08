# from mmengine.model import BaseModule
# from mmengine.registry import MODELS
from mmseg.models import builder
# from mmdet3d.registry import MODELS as mmdet3dMODELS
# from mmdet3d.models import builder as mmdet3dMODELS  # Fix registry issue
from ....builder import MODELS

import torch, torch.nn as nn


# @MODELS.register_module()
from ....builder import VOXEL_ENCODERS
from ....builder import NECKS

@VOXEL_ENCODERS.register_module()
class ResNetSecondFPN(nn.Module):
    def __init__(
        self, 
        img_backbone_config, 
        neck_confifg,
        img_backbone_out_indices,
        pretrained_path=None
    ):

        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone_config)
        # self.img_neck = MODELS.build(neck_confifg)
        self.img_neck = NECKS.build(neck_confifg)

        self.img_backbone_out_indices = img_backbone_out_indices
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            ckpt = ckpt.get("state_dict", ckpt)
            print(self.load_state_dict(ckpt, strict=False))
            print("ResNetSecondFPN Weight Loaded Successfully.")

    def forward(self, imgs):
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        # print("img_feats-resent_2ndfpn shape:", img_feats.shape)
          # print the shape of every feature map
        # print("img_feats-resent_2ndfpn shapes:",
        #   [feat.shape for feat in img_feats])   # or feat.size()

        # img_feats = [img_feats_backbone[idx] for idx in self.img_backbone_out_indices]
        # img_feats = [img_feats_backbone[i] for i in self.img_backbone_out_indices]      

        secondfpn_out = self.img_neck(img_feats)[0]
        return secondfpn_out
