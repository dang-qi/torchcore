from typing import List, Optional
import torch
from torchvision.models.detection.rpn import AnchorGenerator as TorchAnchorGenerator
#from zmq import device
from .build import ANCHOR_GENERATOR_REG

@ANCHOR_GENERATOR_REG.register(force=True)
class AnchorGenerator(TorchAnchorGenerator):
    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
        strides=None,
        round_anchor=True,
    ):
        '''
        if strides is None, then the stride will be calculated in the process, if it is given, then use the given one. The difference is that the stride can be changed in the higher level of FPN since the conv calculation
        '''
    # each tuple of sizes indicate the sizes in the layer for multi layer ouput settings
        super(TorchAnchorGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            raise ValueError('Wrong anchors set up')
            #sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            #raise ValueError('Wrong anchors set up')
            # This is bad
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}
        self.strides = [(s,s) for s in strides]
        self.round_anchor = round_anchor

    def forward(self, inputs, feature_maps):
        '''anchors:List(anchors_per_im:ANCHOR_NUMx4)'''
        grid_sizes = tuple([feature_map.shape[-2:] for feature_map in feature_maps])
        if 'data' not in inputs:
            input_size = inputs['input_size']
            batch_size = inputs['batch_size']
        else:
            input_size = inputs['data'].shape[-2:]
            batch_size = len(inputs['data'])
        if self.strides is None:
            strides = tuple((input_size[0] // g[0], input_size[1] // g[1]) for g in grid_sizes)
        else:
            strides = self.strides
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

    #def set_cell_anchors(self):


    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None
        assert len(grid_sizes) == len(strides) == len(cell_anchors)

        for size, stride, base_anchors in zip(
            grid_sizes, strides, cell_anchors
        ):
            #for base_anchors in cell_anchors:
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            #shifts_x = (torch.arange(
            #    0, grid_width, dtype=torch.float32, device=device
            #)+0.5) * stride_width
            #shifts_y = (torch.arange(
            #    0, grid_height, dtype=torch.float32, device=device
            #)+0.5) * stride_height
            ### torchvision version
            shifts_x = (torch.arange(
                0, grid_width, dtype=torch.float32, device=device
            )) * stride_width
            shifts_y = (torch.arange(
                0, grid_height, dtype=torch.float32, device=device
            )) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors
    #def grid_anchors(self, grid_sizes, strides):
    #    anchors = []
    #    cell_anchors = self.cell_anchors
    # For every (aspect_ratios, scales) combination, output a zero-centered anchor with those values.
    # (scales, aspect_ratios) are usually an element of zip(self.scales, self.aspect_ratios)
    # This method assumes aspect ratio = height / width for an anchor.
    def generate_anchors(
        self,
        scales: List[int],
        aspect_ratios: List[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        if self.round_anchor:
            return base_anchors.round()
        else:
            return base_anchors