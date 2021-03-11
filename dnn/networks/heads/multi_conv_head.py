import torch.nn as nn
from torch.nn.modules import module
from ..common import init, init_focal_loss_head, init_head_gaussian

class MultiConvHead(nn.Module):
    def __init__(self, out_channel, in_channel, head_conv_channel, middle_layer_num, init="gaussion"):
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

        if init == "gaussion":
            init_head_gaussian(self.fc)
        elif init == "focal_loss":
            init_focal_loss_head(self.fc)
        else:
            raise ValueError('not support init method {}'.format(init))

    def forward(self, x):
        return self.fc(x)