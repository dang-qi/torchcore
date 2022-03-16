from turtle import forward
from typing import OrderedDict
import torch
import math
from ..base import BaseModule, ConvBlock
from torch import nn
from ..base.csp_layer import CSPLayer
from .build import NECK_REG

@NECK_REG.register()
class YOLOXPAFPN(BaseModule):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 num_csp_blocks=3,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='swish'),
                 init_cfg=dict(
                     type='kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu'),):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # top-down layers
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for i in range(len(in_channels)-1,0,-1):
            self.reduce_layers.append(
                ConvBlock(
                    in_channels[i],
                    in_channels[i-1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

            self.top_down_blocks.append(
                CSPLayer(
                    2*in_channels[i-1],
                    in_channels[i-1],
                    add_identity=False,
                    num_blocks=num_csp_blocks,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )

        # bottom-up layers
        conv = ConvBlock
        self.down_sample = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for i in range(len(in_channels)-1):
            self.down_sample.append(
                conv(
                    in_channels[i],
                    in_channels[i],
                    3,
                    padding=1,
                    stride=2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )

            self.bottom_up_blocks.append(
                CSPLayer(
                    2*in_channels[i],
                    in_channels[i+1],
                    add_identity=False,
                    num_blocks=num_csp_blocks,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
        
        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvBlock(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )

    def forward(self, features):
        dict_out=False
        if isinstance(features, dict):
            keys = features.keys()
            features = list(features.values())
            dict_out = True
        inner_outs = [features[-1]]
        for i in range(len(self.in_channels)-1,0,-1):
            ind = len(self.in_channels)-1-i
            feat_high = inner_outs[0]
            feat_low = features[i-1]
            feat_high = self.reduce_layers[ind](feat_high)
            inner_outs[0] = feat_high

            up_feat = self.upsample(feat_high)

            inner_out = self.top_down_blocks[ind](torch.cat((up_feat, feat_low),1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for i in range(len(self.in_channels)-1):
            feat_low = outs[-1]
            feat_high = inner_outs[i+1]
            down_feat = self.down_sample[i](feat_low)

            out = self.bottom_up_blocks[i](torch.cat((down_feat, feat_high),1))
            outs.append(out)

        for i in range(len(self.in_channels)):
            outs[i] = self.out_convs[i](outs[i])

        if dict_out:
            out_dict = OrderedDict([(k,v) for k,v in zip(keys, outs)])
            return out_dict
        else:
            return tuple(outs)