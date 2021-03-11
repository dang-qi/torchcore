import torch.nn as nn
from torch.nn.modules import module
from ..common import init

class MultiConvHead(nn.Module):
    def __init__(self, out_channel, in_channel, head_conv_channel, middle_layer_num):
        super().__init__()
        modules = nn.Sequential()
        for i in range(middle_layer_num):
            if i == 0:
                modules.add_module('conv{}'.format(i), nn.Conv2d(in_channel, head_conv_channel,kernel_size=3, stride=1, padding=1, bias=True))
            else:
                modules.add_module('conv{}'.format(i), nn.Conv2d(head_conv_channel, head_conv_channel,kernel_size=3, stride=1, padding=1, bias=True))
            modules.add_module('relu{}'.format(i), nn.ReLU(inplace=True))
        modules.add_module('conv_last', nn.Conv2d(head_conv_channel, out_channel, kernel_size=3, padding=1, bias=True))

        self.fc = modules

        for m in self.modules():
            init(m)

    def forward(self, x):
        return self.fc(x)