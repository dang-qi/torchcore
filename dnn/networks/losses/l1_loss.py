import torch
import torch.nn as nn
import torch.nn.functional as F

class L1LossWithMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask, groundtruth):
        pred = torch.masked_select(pred, mask)
        groundtruth = torch.masked_select(groundtruth, mask)
        loss = F.l1_loss(pred, groundtruth) 
        loss = loss / (mask.sum() + 1e-4)
        return loss

        