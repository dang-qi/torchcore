import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

#import _init
import overlaps_cpu
if torch.cuda.is_available() :
    import overlaps_gpu

class OverlapsFunction(Function):
    def __init__( self, ignore_labels=False, ignore_batch=False ):
        self._ignore_labels = ignore_labels
        self._ignore_batch = ignore_batch

    def forward( self, boxes0, labels0, batches0,
                       boxes1, labels1, batches1 ):
        device = boxes0.device
        ov = torch.zeros( [ boxes0.size()[0], boxes1.size()[0] ],
                          dtype=torch.float32,
                          device=device )

        if boxes0.is_cuda :
            overlaps_gpu.forward( boxes0, labels0, batches0,
                                  boxes1, labels1, batches1, ov,
                                  self._ignore_labels, self._ignore_batch )
        else :
            overlaps_cpu.forward( boxes0, labels0, batches0,
                                  boxes1, labels1, batches1, ov,
                                  self._ignore_labels, self._ignore_batch )
        return ov

    def backward( self, grad_ouputs ):
        return None

class Overlaps( nn.Module ):
    def __init__( self, ignore_labels=False, ignore_batch=False ):
        super().__init__()
        self._ignore_labels=ignore_labels
        self._ignore_batch=ignore_batch

    def forward( self, boxes0, labels0, batches0,
                       boxes1, labels1, batches1 ) :

        return OverlapsFunction(self._ignore_labels, self._ignore_batch)( boxes0, labels0, batches0,
                                                                          boxes1, labels1, batches1 )
