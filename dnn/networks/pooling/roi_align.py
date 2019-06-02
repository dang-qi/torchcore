import numpy as np
import torch
from torch.nn import Module
from torchvision.ops import RoIAlign

class ROIAlign(Module):
    def __init__(self, pool_h, pool_w, scale, sampling=-1):
        super().__init__()
        self.op = RoIAlign( [pool_h, pool_w], scale, sampling )

    # feat: BxCxHxW,  rois: Kx4 (batch_idx, xmin, ymin, xmax, ymax) without normalize
    def forward(self, feat, rois, roibatches):
        rr = torch.cat([roibatches,rois], dim=1 )
        return self.op( feat, rr )
