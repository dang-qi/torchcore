from turtle import forward
from torch import nn
from .conv import build_conv_layer
from .norm import build_norm_layer
from .activation import build_activation
from ..common import init_constant, init_kaiming
class ConvBlock(nn.Module):
    """
        conv, norm, relu module
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 conv_cfg='Conv2d', 
                 norm_cfg=dict(type='BN'), 
                 act_cfg=dict(type='relu')):
        super().__init__()
        self.conv = build_conv_layer(conv_cfg, 
            in_channels, 
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        self.act_cfg = act_cfg

        if self.with_norm:
            _, self.norm = build_norm_layer(norm_cfg, out_channels)
        if self.with_activation:
            self.act = build_activation(act_cfg)

    def init_weights(self):
        '''default weight init'''
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            init_kaiming(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            init_constant(self.norm, 1, bias=0)

    def forward(self,x):
        x = self.conv(x)
        if self.with_norm:
            x = self.norm(x)
        if self.with_activation:
            x = self.act(x)
        return x
