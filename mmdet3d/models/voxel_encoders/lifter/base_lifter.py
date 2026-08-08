# from mmseg.registry import MODELS
# from mmengine.model import BaseModule


# @MODELS.register_module()
import torch.nn as nn

from ...builder import VOXEL_ENCODERS

@VOXEL_ENCODERS.register_module()
class BaseLifter(nn.Module):#Added nn.Module 

    """Base lifter class.
    image backbone -> neck -> lifter -> encoder -> segmentor
    Lift multi-scale image features to 3D representations, e.g. Voxels or TPV or BEV.
    """

    def __init__(self, init_cfg=None, **kwargs) -> None:
        # super().__init__(init_cfg)
        super().__init__()  # Remove init_cfg from super()
        self.init_cfg = init_cfg  # Store it separately if needed
    
    def forward(
        self, 
        ms_img_feats, 
        metas=None, 
        **kwargs
    ):
        pass