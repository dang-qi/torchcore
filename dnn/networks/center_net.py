import torch.nn as nn
from .one_stage_detector import OneStageDetector
from .heads import CommonHead, ComposedHead
from .losses import FocalLossHeatmap

class CenterNet(OneStageDetector):
    def __init__(self, backbone, num_classes, loss_parts=['heatmap'], cfg=None, neck=None, pred_heads=None):
        #super(OneStageDetector,self).__init__()
        if pred_heads is None:
            if neck is None:
                in_channel = backbone.out_channel
            else:
                in_channel = neck.out_channel
            heads = get_center_head(in_channel, num_classes)

        losses = CenterNetLoss(loss_parts)

        #self.backbone = backbone
        #self.neck = neck
        #self.pred_heads = pred_heads
        #self.losses = losses
        super().__init__(backbone, heads, losses, neck=neck)

    def postprocess(self, pred, inputs):
        return pred

class CenterNetLoss(nn.Module):
    def __init__(self, loss_parts):
        super().__init__()
        self.loss_parts = loss_parts
        if 'heatmap' in loss_parts:
            self.heatmap_loss = FocalLossHeatmap(alpha=0.5, gamma=2)
    
    def forward(self, pred, targets):
        losses = {}
        if 'heatmap' in targets:
            losses['heatmap'] = self.heatmap_loss(pred['heatmap'], targets['heatmap'])
        return losses



def get_center_head(in_channel, num_classes):
    head_names = ['heatmap', 'offset', 'width_height']
    heatmap_head = CommonHead(num_classes, in_channel, head_conv_channel=64)
    offset_head = CommonHead(2, in_channel, head_conv_channel=64)
    witdh_height_head = CommonHead(2, in_channel, head_conv_channel=64)
    heads = [heatmap_head, offset_head, witdh_height_head]
    return ComposedHead(head_names, heads)
