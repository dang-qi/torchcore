import torch
from torch import nn
from ..tools import AnchorBoxesCoder

class FastRCNNHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.class_num = cfg.class_num
        class_num = cfg.class_num
        self.cfg = cfg
        feature_num = 1024
        pool_h = cfg.roi_pool.pool_h
        pool_w = cfg.roi_pool.pool_w
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

        # concat the rois and targets and then split them
        #rois_per_im = [len(roi) for roi in rois]
        #rois = torch.cat(rois, dim=0)
        features = self.feature_head(rois)
        label_pre = self.label_head(features)
        label_pre = label_pre.view(label_pre.shape[:2])
        bbox_pre = self.bbox_head(features)
        bbox_pre = bbox_pre.view(bbox_pre.shape[:2])

        #label_pre = torch.split(label_pre, rois_per_im)
        #bbox_pre = torch.split(bbox_pre, rois_per_im)

        return label_pre, bbox_pre


