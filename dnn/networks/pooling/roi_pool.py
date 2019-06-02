import numpy as np
import torch
from torch.nn import Module
from torchvision.ops import RoIPool

class ROIPool(Module):
    def __init__(self, pool_h, pool_w, scale):
        super().__init__()
        self.op = RoIPool( [pool_h, pool_w], scale )

    # feat: BxCxHxW,  rois: Kx4 (batch_idx, xmin, ymin, xmax, ymax) without normalize
    def forward(self, feat, rois, roibatches):
        roibatches = roibatches.detach().cpu().numpy()
        roibatches = roibatches.astype( np.float32 )
        roibatches = torch.from_numpy( roibatches ).to( feat.device )

        rr = torch.cat([roibatches,rois], dim=1 )
        return self.op( feat, rr )
