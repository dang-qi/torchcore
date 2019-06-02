import torch
import numpy as np
from torch.nn import Module
from torchvision.ops import nms

class Nms( Module ):
    def __init__( self, threshold, max ):
        super().__init__()
        self._threshold = threshold
        self._max = max

    def forward( self, rois, scores ):
        keep = nms( rois, scores, self._threshold )
        print( scores[keep] )

        input()


        return nms(rois, scores, self._threshold)
