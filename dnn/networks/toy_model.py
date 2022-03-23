import torch
from torch import nn
from .base.conv_block import ConvBlock

class ToyModel(nn.Module):
    def __init__(self, out_channel=4, kernel_size=1):
        super().__init__()
        self.conv = ConvBlock(3, out_channel, kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv(x)
        #if self.training:
        #    loss = (torch.rand_like(x)-x).mean()
        #    return loss
        #else:
        return x
