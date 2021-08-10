import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha: float=0.25, gamma:float=2, reduction: str='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, beta=4, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, pred, groundtruth):
        return self._forword_impl(pred, groundtruth)

    def _forword_impl(self, pred, groundtruth):
        '''
            pred and groundtruth should be same size. 
            pred should have value between 0-1
            groundtruth shold have 0, 1 value
        '''
        # IMPORTANT: pred can not be 0 or 1 because it goes to log function
        pred = torch.clamp(pred, min=1e-4, max=1-1e-4)

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

class FocalLossSigmoid(nn.Module):
    '''
        Please remember to init the conv layer for focal loss properly
    '''
    def __init__(self, alpha=0.5, beta=4, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, groundtruth:torch.Tensor):
        '''
            pred: original output without sigmoid, shape: N * class_num, 
                (class_num does not count backgroud)
            groundtruth: class labels (0 for background), shape: N
        '''
        # IMPORTANT: pred can not be 0 or 1 because it goes to log function
        pred = torch.clamp(pred.sigmoid_(), min=1e-4, max=1-1e-4)
        
        # convert groundtruth to proper targets (one hot for non background, 
        # zero for background)
        N, C = pred.shape
        non_zero_ind = groundtruth.nonzero(as_tuple=False)
        flatten_non_zero_ind = non_zero_ind.flatten()
        target = pred.new_zeros(pred.shape)
        target[flatten_non_zero_ind] = target[flatten_non_zero_ind].scatter_(1, groundtruth[non_zero_ind].long()-1, 1)

        loss = self._forword_impl(pred, target)
        return loss

    def _forword_impl(self, pred, groundtruth):
        '''
            pred and groundtruth should be same size. 
            pred should have value between 0-1
            groundtruth shold have 0, 1 value
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

class FocalLossHeatmap(nn.Module):
    '''
        Please remember to init the conv layer for focal loss properly
    '''
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


