import torch
from torch import nn
import collections
from ..common import init_focal_loss_head, init_head_gaussian

class RetinaHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        #super(RPNHead, self).__init__()
        super().__init__()
        cls_layers = []
        for _ in range(4):
            cls_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_layers.append(nn.ReLU(inplace=True))
        cls_layers.append(nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1))
        self.cls_head = nn.Sequential(*cls_layers)

        bbox_layers = []
        for _ in range(4):
            bbox_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            bbox_layers.append(nn.ReLU(inplace=True))
        bbox_layers.append(nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1))
        self.bbox_head = nn.Sequential(*bbox_layers)

        init_focal_loss_head(self.cls_head, pi=0.01)
        init_head_gaussian(self.bbox_head, std=0.01 )

        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, features):
        # the input support dict(key->string: val->torch.tensor),
        # list([torch.tensor....])
        # torch.tenser
        if isinstance(features, dict):
            #features = list(features.values())
            out = collections.OrderedDict()
            for k,v in features.items():
                out[k] = self._forward_once(v)
        elif isinstance(features, list):
            out = [self._forward_once(feature) for feature in features]
        elif torch.is_tensor(features):
            out = self._forward_once(features)
        else:
            raise ValueError('Wrong input type {} for RPN head.'.format(type(features)))
        return out

    def _forward_once(self, feature):
        class_pred = self.cls_head(feature)
        bbox_pred = self.bbox_head(feature)
        return class_pred, bbox_pred

