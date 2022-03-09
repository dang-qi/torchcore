import torch
from torch import nn
from .conv_block import ConvBlock

class DarknetBottleNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 conv_cfg=None,
                 add_identity=True,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='swish'),):
        super().__init__()
        mid_channels = int(out_channels*expansion)
        self.conv1 = ConvBlock(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = ConvBlock(
            mid_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.add_identity = add_identity and in_channels==out_channels

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + x
        else:
            return out


class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 add_identity=True,
                 expand_ratio=0.5,
                 num_blocks=1,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='swish'),
                 ):
        super().__init__()
        mid_channels= int(out_channels * expand_ratio)
        self.main_conv = ConvBlock(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        
        self.short_conv = ConvBlock(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.final_conv = ConvBlock(
            mid_channels*2,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.blocks = nn.Sequential(*[
            DarknetBottleNeck(
                mid_channels,
                mid_channels,
                1.0,
                add_identity=add_identity,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ) for _ in range(num_blocks)]
        )

    def forward(self, x):
        short = self.short_conv(x)

        main = self.main_conv(x)
        main = self.blocks(main)

        out = torch.cat((short, main),dim=1)
        return self.final_conv(out)
