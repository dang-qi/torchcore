import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, groundtruth):
        pass

class FocalLossHeatmap(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, groundtruth):
        pred = pred.sigmoid()
        

    def _forword_impl(self, pred, groundtruth):
        '''
            pred: batch_size*class_num*w*h
        '''

