import torch
from collections.abc import Iterable
from .anchor_generator import AnchorGenerator
from .build import ANCHOR_GENERATOR_REG

@ANCHOR_GENERATOR_REG.register()
class RoIAnchorGenerator(AnchorGenerator):
    def __init__(
        self,
        base_stride,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        '''
        base_stride has two mode: 
            If base_stride is float/int:it is the stride of the first feature level sending to the FPN
            If base_stride are tuple(tuple(stride_height, stride_width)...):
            then the stride are fixed.
        '''
        super().__init__(sizes=sizes, aspect_ratios=aspect_ratios)
        if isinstance(base_stride, Iterable):
            self.base_stride = None
            self.strides = base_stride
        else:
            self.base_stride = base_stride
        


    def forward(self, inputs, feature_maps):
        grid_sizes = tuple([feature_map.shape[-2:] for feature_map in feature_maps])
        if 'data' not in inputs:
            batch_size = inputs['batch_size']
        else:
            batch_size = len(inputs['data'])
        if self.base_stride is None:
            assert len(self.strides) == len(grid_sizes)
            strides = self.strides
        else:
            strides = tuple(((grid_sizes[0][0] // g[0])* self.base_stride, (grid_sizes[0][1] // g[1])*self.base_stride) for g in grid_sizes)
        try:
            # for earlier version torchvision
            self.set_cell_anchors(feature_maps[0].device)
        except TypeError:
            self.set_cell_anchors(feature_maps[0].dtype,feature_maps[0].device)

        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        anchors = []
        for _ in range(batch_size):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors
