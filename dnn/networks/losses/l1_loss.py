import torch
import torch.nn as nn
import torch.nn.functional as F

class L1LossWithMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask, groundtruth):
        pred = torch.masked_select(pred, mask)
        groundtruth = torch.masked_select(groundtruth, mask)
        loss = F.l1_loss(pred, groundtruth, reduction='sum') 
        loss = loss / (mask.sum() + 1e-4)
        return loss

class L1LossWithInd(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, inds, ind_mask, gt):
        batch, c, h, w = pred.shape
        pred = pred.view(batch, c, -1)
        inds = inds.unsqueeze(1).expand(batch, c, inds.size(1) )
        ind_mask = ind_mask.unsqueeze(1).expand(batch, c, ind_mask.size(1))
        pred = pred.gather(2, inds)
        loss = F.l1_loss(pred*ind_mask, gt*ind_mask, reduction='sum')
        loss = loss / (ind_mask.sum() + 1e-4)
        return loss
        