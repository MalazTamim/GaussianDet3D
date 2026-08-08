# from torch.nn import LayerNorm
# @MIDDLE_ENCODERS.register_module(name='LN', module=LayerNorm)

# from mmcv.utils import Registry
from torch.nn import LayerNorm
from ..builder import MIDDLE_ENCODERS

# MIDDLE_ENCODERS = Registry('MIDDLE_ENCODERS')

@MIDDLE_ENCODERS.register_module()
class LN(LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)
