# from mmengine.registry import MODELS
# from mmengine.model import BaseModule
import spconv.pytorch as spconv
import torch.nn as nn, torch
from functools import partial
from .utils import spherical2cartesian, cartesian


# @MODELS.register_module()
from ....builder import MIDDLE_ENCODERS
@MIDDLE_ENCODERS.register_module()
class SparseConv3D(nn.Module):
    def __init__(
        self, 
        in_channels,
        embed_channels,
        pc_range,
        grid_size,
        xyz_activation="sigmoid",
        use_out_proj=False,
        kernel_size=5,
        use_multi_layer=False,
        init_cfg=None,
        **kwargs,
    ):
        # super().__init__(init_cfg)
        super().__init__()
        self.init_cfg = init_cfg  # Store it separately if needed

        
        if use_multi_layer:
            self.layer = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, embed_channels, kernel_size, 1, (kernel_size - 1) // 2),
                nn.LayerNorm(embed_channels),
                nn.ReLU(True),
                spconv.SubMConv3d(embed_channels, embed_channels, kernel_size, 1, (kernel_size - 1) // 2),
                nn.LayerNorm(embed_channels),
                nn.ReLU(True),
                spconv.SubMConv3d(embed_channels, embed_channels, kernel_size, 1, (kernel_size - 1) // 2),
                nn.LayerNorm(embed_channels),
                nn.ReLU(True),
            )        
        else:
            self.layer = spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False)
        if use_out_proj:
            self.output_proj = nn.Linear(embed_channels, embed_channels)
        else:
            self.output_proj = nn.Identity()
        self.get_xyz = partial(cartesian, pc_range=pc_range, use_sigmoid=(xyz_activation=="sigmoid"))
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float))

    def forward(self, instance_feature, anchor):
        # anchor: b, g, 11
        # instance_feature: b, g, c
        '''
        print("SparseConv3D:    Instance feature shape:     ", instance_feature.shape)
        print("SparseConv3D:    anchor  shape:     ", anchor.shape)
        '''


        bs, g, _ = instance_feature.shape

        # sparsify
        anchor_xyz = anchor[..., :3]
        anchor_xyz = self.get_xyz(anchor_xyz).flatten(0, 1) 

        # indices = anchor_xyz - anchor_xyz.min(0, keepdim=True)[0]
        # self.pc_range = torch.tensor(self.pc_range, device=anchor_xyz.device)#Added
        # self.grid_size = torch.tensor(self.grid_size, device=anchor_xyz.device)#Added

        indices = anchor_xyz - self.pc_range[None, :3]
        indices = indices / self.grid_size[None, :] # bg, 3
        indices = indices.to(torch.int32)
        batched_indices = torch.cat([
            torch.arange(bs, device=indices.device, dtype=torch.int32).reshape(
            bs, 1, 1).expand(-1, g, -1).flatten(0, 1), indices], dim=-1)
        
        # spatial_shape = indices.max(0)[0]
        spatial_shape = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        spatial_shape = spatial_shape.to(torch.int32)


        input = spconv.SparseConvTensor(
            instance_feature.flatten(0, 1), # bg, c
            indices=batched_indices, # bg, 4
            spatial_shape=spatial_shape,
            batch_size=bs)
        
        # Debugging prints
        '''

        print("SparseConv3D: SparseConvTensor Construction:")
        print(f" SparseConv3D: Instance Feature Shape (flattened): {instance_feature.flatten(0, 1).shape}")  # Shape of the flattened features
        print(f"SparseConv3D:  Indices Shape: {batched_indices.shape}")  # Shape of the batched indices
        print(f" SparseConv3D: Spatial Shape: {spatial_shape.tolist()}")  # Spatial shape of the grid
        print(f"SparseConv3D:  Batch Size: {bs}")  # Number of batches
        '''


        output = self.layer(input)
        # print("SparseConv3D:   Output  original :", output)

        output = output.features.unflatten(0, (bs, g))
        '''

        print("SparseConv3D:   Output shape final :", output.shape)
        print("SparseConv3D: self.output_proj(output)     ", self.output_proj(output).shape)
        '''


        return self.output_proj(output)
