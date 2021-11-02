import torch
from torch import nn
import collections
from ..common import init_focal_loss_head, init_head_gaussian
import torch.nn.functional as F
from ..base.norm import build_norm_layer

from .build import HEAD_REG

@HEAD_REG.register()
class FCOSFeatureHead(nn.Module):
    def __init__(self, in_channels, num_classes, norm_layer_cfg, strides, centerness=True, center_with_cls=True, num_conv=4, norm_on_bbox=False,):
        #super(RPNHead, self).__init__()
        super().__init__()
        cls_layers = []
        for _ in range(num_conv):
            cls_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_layers.append(build_norm_layer(norm_layer_cfg, in_channels)[1])
            cls_layers.append(nn.ReLU(inplace=True))
        self.cls_conv = nn.Sequential(*cls_layers)
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.norm_on_bbox = norm_on_bbox
        self.strides = strides

        bbox_layers = []
        for _ in range(num_conv):
            bbox_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            bbox_layers.append(build_norm_layer(norm_layer_cfg, in_channels)[1])
            bbox_layers.append(nn.ReLU(inplace=True))
        self.bbox_conv = nn.Sequential(*bbox_layers)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)


        init_head_gaussian(self.bbox_conv, std=0.01 )
        init_head_gaussian(self.bbox_pred, std=0.01 )
        init_head_gaussian(self.cls_conv, std=0.01 )
        init_focal_loss_head(self.cls_logits, pi=0.01)

        if centerness:
            self.centerness_head = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1,
                padding=1)
            init_head_gaussian(self.centerness_head, std=0.01 )

        #self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.centerness = centerness
        self.center_with_cls=center_with_cls

    def forward(self, features):
        # the input support dict(key->string: val->torch.tensor),
        # list([torch.tensor....])
        # torch.tenser
        if isinstance(features, dict):
            #features = list(features.values())
            out = collections.OrderedDict()
            for i, (k,v) in enumerate(features.items()):
                out[k] = self._forward_once(v, self.strides[i])
        elif isinstance(features, list):
            out = [self._forward_once(feature, self.strides[i]) for i, feature in enumerate(features)]
        elif torch.is_tensor(features):
            out = self._forward_once(features, self.strides[0])
        else:
            raise ValueError('Wrong input type {} for RPN head.'.format(type(features)))
        return out

    def _forward_once(self, feature, stride=None):
        class_feature = self.cls_conv(feature)
        class_pred = self.cls_logits(class_feature)
        bbox_feature = self.bbox_conv(feature)
        bbox_pred = self.bbox_pred(bbox_feature)

        if not self.centerness:
            return class_pred, bbox_pred

        if self.center_with_cls:
            centerness = self.centerness_head(class_feature)
        else:
            centerness = self.centerness_head(bbox_feature)

        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        #bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return class_pred, bbox_pred, centerness

