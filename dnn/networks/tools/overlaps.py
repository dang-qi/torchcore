import torch
import numpy as np
from torch.nn import Module
from torch.autograd import Function

import overlaps_cpu
if torch.cuda.is_available() :
    import overlaps_cuda

class OverlapsFunction( Function ):
    @staticmethod
    def forward(self, rois0, roilabels0, roibatches0, rois1, roilabels1, roibatches1):
        overlaps = torch.zeros( (rois0.size(0), rois1.size(0)), dtype=rois0.dtype, device=rois0.device )


        if rois0.is_cuda :
            overlaps_cuda.forward_gpu( rois0, roilabels0, roibatches0,
                                    rois1, roilabels1, roibatches1,
                                    overlaps )
        else :
            overlaps_cpu.forward_cpu( rois0, roilabels0, roibatches0,
                                    rois1, roilabels1, roibatches1,
                                    overlaps )

        overlaps = overlaps.detach()

        return overlaps

    @staticmethod
    def bachward( self, outgrad ):
        return None, None, None, None, None, None

class Overlaps( Module ):
    def __init__( self ):
        super().__init__()

    def forward( self, rois0, roilabels0, roibatches0, rois1, roilabels1, roibatches1 ):
        return OverlapsFunction.apply(rois0, roilabels0, roibatches0, rois1, roilabels1, roibatches1)
