import torch

from torch.nn.modules.utils import _pair

from .build import ANCHOR_GENERATOR_REG

@ANCHOR_GENERATOR_REG.register(force=True)
class MultiLevelGridPointGenerator():
    def __init__(self,
                 strides,
                 offset=0.5) -> None:
        # convert strides to (w, h) pairs if needed
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    def single_level_grid(self, feature_size, level, dtype=torch.float32, device='cpu', with_strides=False):
        h, w = feature_size
        stride_w, stride_h = self.strides[level]

        x = (torch.arange(w, device=device, dtype=dtype) + self.offset) * stride_w
        y = (torch.arange(h, device=device, dtype=dtype) + self.offset) * stride_h
        yy, xx = torch.meshgrid([y,x])
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)
        if with_strides:
            stride_w_vec = torch.full_like(xx, stride_w)
            stride_h_vec = torch.full_like(xx, stride_h)
            return torch.stack([xx,yy,stride_w_vec, stride_h_vec], dim=-1)
        else:
            return torch.stack([xx, yy], dim=-1)

    def multi_level_grid(self, features_sizes, with_strides=False, dtype=torch.float32, device='cpu'):
        '''Generate point pirors,
        The output should be [(N_lvl1,2), (N_lvl2,2)...]
        For example, we only have one level (2,2) feature and stride is 32, offset is 0
        the output should be:
        tensor([[ 0.,  0.],
                [32.,  0.],
                [ 0., 32.],
                [32., 32.]])
        '''
        all_priors = []
        for i, feature_size in enumerate(features_sizes):
            all_priors.append(
                self.single_level_grid(feature_size, i, dtype=dtype, device=device, with_strides=with_strides))
        return all_priors
