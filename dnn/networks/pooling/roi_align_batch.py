import numpy as np
import torch
from torch.nn import Module
from torchvision.ops import RoIAlign, roi_align

class ROIAlignBatch(Module):
    def __init__(self, pool_h, pool_w, sampling=-1):
        super().__init__()
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.sampling =sampling


    # feat: BxCxHxW,  rois: Kx4 (batch_idx, xmin, ymin, xmax, ymax) without normalize
    def forward(self, feat, roibatches, stride):
        roi_num_per_batch = [len(roi) for roi in roibatches]
        rois_out = roi_align(feat, roibatches, (self.pool_h, self.pool_w), spatial_scale=1.0/stride, sampling_ratio=self.sampling)
        rois_out = torch.split(rois_out, roi_num_per_batch)
        return rois_out
