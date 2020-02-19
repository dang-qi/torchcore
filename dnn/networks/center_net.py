import torch
import torch.nn as nn

from .one_stage_detector import OneStageDetector
from .heads import CommonHead, ComposedHead
from .losses import FocalLossHeatmap, L1LossWithMask, L1LossWithInd

class CenterNet(OneStageDetector):
    def __init__(self, backbone, num_classes, parts=['heatmap'], cfg=None, neck=None, pred_heads=None):
        #super(OneStageDetector,self).__init__()
        if pred_heads is None:
            if neck is None:
                in_channel = backbone.out_channel
            else:
                in_channel = neck.out_channel
            heads = get_center_head(in_channel, num_classes)
        self.parts = parts

        losses = CenterNetLoss(parts)

        #self.backbone = backbone
        #self.neck = neck
        #self.pred_heads = pred_heads
        #self.losses = losses
        super().__init__(backbone, heads, losses, neck=neck)

    def postprocess(self, pred, inputs):
        heatmap = pred['heatmap'].sigmoid_()

        k=100
        heatmap = point_nms(heatmap)
        mask = topk_mask(heatmap, k=k)
        batch_size = heatmap.size(0)
        scores= heatmap.masked_select(mask).view(batch_size, k)
        ys, xs, categories = decode_mask(mask) # (batch_size, k)
        if 'offset' in self.parts:
            offset = pred['offset']
            offset = offset.masked_select(mask)
            
        if 'width_height' in self.parts:
            width_height = pred['width_height']
            width_height = width_height.masked_select(mask)

        return pred

class CenterNetLoss(nn.Module):
    def __init__(self, loss_parts, loss_weight=None):
        super().__init__()
        self.loss_parts = loss_parts
        if loss_weight is None:
            loss_weight = {'heatmap':1., 'offset':1., 'width_height':0.1}
            self.loss_weight = loss_weight
        if 'heatmap' in loss_parts:
            self.heatmap_loss = FocalLossHeatmap(alpha=0.5, gamma=2)
        if 'offset' in loss_parts:
            #self.offset_loss = L1LossWithMask()
            self.offset_loss = L1LossWithInd()
        if 'width_height' in loss_parts:
            #self.width_height_loss = L1LossWithMask() 
            self.width_height_loss = L1LossWithInd() 
    
    def forward(self, pred, targets ):
        losses = {}
        if 'heatmap' in self.loss_parts:
            losses['heatmap'] = self.heatmap_loss(pred['heatmap'], targets['heatmap']) * self.loss_weight['heatmap']
        if 'offset' in self.loss_parts:
            #losses['offset'] = self.offset_loss(pred['offset'], targets['mask'], targets['offset']) * self.loss_weight['offset']
            losses['offset'] = self.offset_loss(pred['offset'], targets['ind'],targets['ind_mask'], targets['offset']) * self.loss_weight['offset']
        if 'width_height' in self.loss_parts:
            #losses['width_height'] = self.width_height_loss(pred['width_height'], targets['mask'], targets['width_height']) * self.loss_weight['width_height']
            losses['width_height'] = self.width_height_loss(pred['width_height'], targets['ind'], targets['ind_mask'], targets['width_height']) * self.loss_weight['width_height']
        return losses



def get_center_head(in_channel, num_classes):
    head_names = ['heatmap', 'offset', 'width_height']
    heatmap_head = CommonHead(num_classes, in_channel, head_conv_channel=64)
    offset_head = CommonHead(2, in_channel, head_conv_channel=64)
    witdh_height_head = CommonHead(2, in_channel, head_conv_channel=64)
    heads = [heatmap_head, offset_head, witdh_height_head]
    return ComposedHead(head_names, heads)

def point_nms(heatmap, kernal_size=3):
    padding = (kernal_size - 1) // 2
    heatout = nn.functional.max_pool2d(heatmap, kernal_size=kernal_size, padding=padding, stride=1 )
    mask = (heatout == heatmap).float()
    return mask*heatmap

def topk_mask(heatmap, k=100):
    n, c, h, w = heatmap.shape
    # getting the topk in 2 demension
    topk, inds = torch.topk(heatmap.view(n,-1), k=k)
    # set the mask in 2 demension and transfer to desired shape
    mask = torch.zeros((n, c*h*w), dtype=bool)
    mask = mask.scatter(1, inds, True)
    mask = mask.view(n,c,h,w)
    return mask
    
def decode_mask(mask):
    # decode other stuff from the mask indexes
    n, c, h, w = mask.shape
    indexes = mask.nonzero()
    categories = indexes[:,1].view(n,-1)
    ys = indexes[:,2].view(n,-1)
    xs = indexes[:,1].view(n,-1)
    return ys, xs, categories

#def get_from_mask(mask, heatmap)
    