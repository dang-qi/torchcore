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
    def __init__(self, alpha=0.5, beta=4, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, groundtruth):
        #pred = pred.sigmoid()
        # IMPORTANT: pred can not be 0 or 1 because it goes to log function
        pred = torch.clamp(pred.sigmoid_(), min=1e-4, max=1-1e-4)
        loss = self._forword_impl(pred, groundtruth)
        return loss
        

    def _forword_impl(self, pred, groundtruth):
        '''
            pred: batch_size*class_num * h * w
                  (after sigmoid)
            groundtruth: batch_size * class_num * h * w 
                         (gaussian heatmap)
        '''
        pos_mask = groundtruth.eq(1).float()
        neg_mask = groundtruth.lt(1).float()

        neg_weight = torch.pow((1 - groundtruth), self.beta) # just follow the original code TODO: check it later

        pos_loss = self.alpha * torch.pow(1 - pred, self.gamma) * torch.log(pred) * pos_mask
        neg_loss = (1 - self.alpha) * torch.pow(pred, self.gamma) * torch.log(1-pred) * neg_mask * neg_weight

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        pos_num = pos_mask.sum()

        loss = 0
        if pos_num == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (neg_loss + pos_loss) / pos_num
        return loss


