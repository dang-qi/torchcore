import torch.nn as nn
from .one_stage_detector import OneStageDetector
from .heads import CommonHead, ComposedHead

class CenterNet(OneStageDetector):
    def __init__(self, backbone, num_classes, losses=None, cfg=None, neck=None, pred_heads=None):
        if pred_heads is None:
            if neck is None:
                in_channel = backbone.out_channel
            else:
                in_channel = neck.out_channel
            heads = get_center_head(in_channel, num_classes)

        super().__init__(backbone, heads, losses, neck=neck)

    def postprocess(self, pred, inputs):
        return pred


def get_center_head(in_channel, num_classes):
    head_names = ['heatmap', 'offset', 'width_height']
    heatmap_head = CommonHead(num_classes, in_channel, head_conv_channel=64)
    offset_head = CommonHead(2, in_channel, head_conv_channel=64)
    witdh_height_head = CommonHead(2, in_channel, head_conv_channel=64)
    heads = [heatmap_head, offset_head, witdh_height_head]
    return ComposedHead(head_names, heads)
