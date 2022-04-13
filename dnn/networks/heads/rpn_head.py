import torch
from torch import nn
from torch.nn import functional as F
import collections
from .build import HEAD_REG

@HEAD_REG.register()
class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        #super(RPNHead, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.class_pred = nn.Conv2d(in_channels, num_anchors*1, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors*4, kernel_size=1, stride=1)

        for layer in self.children():
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

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
        feature1 = F.relu(self.conv1(feature))
        class_pred = self.class_pred(feature1)
        bbox_pred = self.bbox_pred(feature1)
        return class_pred, bbox_pred

