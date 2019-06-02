import torch
import numpy as np
from torch.nn import Module
from torchvision.ops import nms

class Nms( Module ):
    def __init__( self, threshold ):
        super().__init__()
        self._threshold = threshold

    def forward( self, rois, scores ):
        return nms(rois, scores, self._threshold)
