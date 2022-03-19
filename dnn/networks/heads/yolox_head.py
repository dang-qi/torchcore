from turtle import forward
import torch
import math
from torch import nn
from ..base import BaseModule, ConvBlock
from ..common import init_focal_loss
from .build import HEAD_REG

@HEAD_REG.register(force=True)
class YOLOXFeatureHead(BaseModule):
    def __init__(self, 
                 num_classes,
                 in_channels,
                 out_channels=256,
                 stacked_convs=2,
                 strides=[8, 16, 32],
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='swish'),
                 init_cfg=dict(
                     type='kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super().__init__(init_cfg)
        self.num_class = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.layer_num = len(strides)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.stacked_convs = stacked_convs
        self.build_layers()

    def forward(self,x):
        if isinstance(x, dict):
            x = list(x.values())
        pred_all = []
        for i in range(self.layer_num):
            cls_feature = self.multi_level_features_cls[i](x[i])
            reg_feature = self.multi_level_features_reg[i](x[i])

            cls_pred = self.multi_level_cls[i](cls_feature)
            reg_pred = self.multi_level_reg[i](reg_feature)
            obj_pred = self.multi_level_obj[i](reg_feature)
            pred_all.append((cls_pred,reg_pred,obj_pred))
        return pred_all

            

    def build_layers(self):
        self.multi_level_features_cls = nn.ModuleList()
        self.multi_level_features_reg = nn.ModuleList()
        self.multi_level_cls = nn.ModuleList()
        self.multi_level_reg = nn.ModuleList()
        self.multi_level_obj = nn.ModuleList()
        # build feature layers
        for _ in range(self.layer_num):
            self.multi_level_features_cls.append(self._build_stacked_convs())
            self.multi_level_features_reg.append(self._build_stacked_convs())
            self.multi_level_cls.append(
                nn.Conv2d(self.out_channels, self.num_class, 1)
            )
            self.multi_level_reg.append(
                nn.Conv2d(self.out_channels, 4, 1)
            )
            self.multi_level_obj.append(
                nn.Conv2d(self.out_channels, 1, 1)
            )

    def _build_stacked_convs(self):
        conv = ConvBlock
        stacked_feature_conv = []
        for i in range(self.stacked_convs):
            in_channels = self.in_channels if i==0 else self.out_channels
            stacked_feature_conv.append(
                conv(
                    in_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )
        return nn.Sequential(*stacked_feature_conv)

    def init_weights(self):
        super().init_weights()
        # init head for focal loss
        for cls_head, obj_head in zip(self.multi_level_cls, self.multi_level_obj):
            init_focal_loss(cls_head)
            init_focal_loss(obj_head)

        
