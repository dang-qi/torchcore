import torch
from torch import nn
from ..tools import AnchorBoxesCoder

from .build import HEAD_REG

@HEAD_REG.register()
class FastRCNNHead(nn.Module):
    def __init__(self, class_num, pool_w, pool_h, out_feature_num):
        super().__init__()
        self.class_num = class_num
        feature_num = 1024
        self.label_head = nn.Linear(feature_num, class_num+1)
        self.bbox_head = nn.Linear(feature_num, 4*(class_num))
        self.feature_head = nn.Sequential( 
            nn.Linear(out_feature_num*pool_h*pool_w, feature_num),
            nn.ReLU(inplace=True),
            nn.Linear(feature_num, feature_num),
            nn.ReLU(inplace=True)
        )

        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        nn.init.normal_(m.weight, std=0.01)
        #        nn.init.constant_(m.bias, 0)

    def forward(self, rois ):

        rois = rois.flatten(start_dim=1)
        features = self.feature_head(rois)
        label_pre = self.label_head(features)
        bbox_pre = self.bbox_head(features)

        return label_pre, bbox_pre




class FastRCNNHeadConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.class_num = cfg.class_num
        class_num = cfg.class_num
        self.cfg = cfg
        feature_num = 1024
        pool_h = cfg.pool_h
        pool_w = cfg.pool_w
        self.feature_head = nn.Sequential( 
            nn.Conv2d(in_channels=cfg.out_feature_num, out_channels=feature_num, kernel_size=(pool_h, pool_w), padding=0 ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_num, out_channels=feature_num, kernel_size=1, padding=0 ),
            nn.ReLU(inplace=True)
        )
        self.label_head = nn.Conv2d(in_channels=feature_num, out_channels=class_num+1, kernel_size=1, padding=0 )
        self.bbox_head = nn.Conv2d(in_channels=feature_num, out_channels=4*(class_num+1), kernel_size=1, padding=0 )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, rois ):

        features = self.feature_head(rois)
        label_pre = self.label_head(features)
        label_pre = label_pre.view(label_pre.shape[:2])
        bbox_pre = self.bbox_head(features)
        bbox_pre = bbox_pre.view(bbox_pre.shape[:2])

        return label_pre, bbox_pre
