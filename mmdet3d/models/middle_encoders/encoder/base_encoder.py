# from mmseg.registry import MODELS
# from mmengine.model import BaseModule

import torch.nn as nn
# @MODELS.register_module()
from ...builder import MIDDLE_ENCODERS
@MIDDLE_ENCODERS.register_module()
class BaseEncoder(nn.Module):#BaseModule
    """Further encode 3D representations.
    image backbone -> neck -> lifter -> encoder -> segmentor
    """

    def __init__(self, init_cfg=None, **kwargs):
        # super().__init__(init_cfg)
        super().__init__()  # Remove init_cfg from super()
        self.init_cfg = init_cfg  # Store it separately if needed
    
    def forward(
        self, 
        representation,
        ms_img_feats=None,
        metas=None,
        **kwargs
    ):
        pass