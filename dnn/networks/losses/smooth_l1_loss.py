import torch
import torch.nn as nn
from torch.nn import SmoothL1Loss


from .build import LOSS_REG

LOSS_REG.register(SmoothL1Loss, 'SmoothL1Loss')


class SmoothL1LossOld( nn.Module ):
    def __init__( self, sigma ):
        super().__init__()
        self._sigma2 = sigma*sigma

    def forward( self, targets, preds ):
        diff = targets - preds
        diff_abs = torch.abs( diff )

        cond = diff_abs < 1.0/self._sigma2
        loss = torch.where( cond, 0.5 * (diff_abs ** 2) * self._sigma2, diff_abs - (0.5/self._sigma2))

        return loss.sum(dim=1)
