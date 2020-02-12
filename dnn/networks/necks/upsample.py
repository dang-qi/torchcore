import torch
import torch.nn as nn
from ..common import init

#select trans conv parameters so the out put is 2 times the input
def select_transconv_param(kernal_size): 
    assert kernal_size>1
    output_padding = kernal_size % 2
    padding = int((kernal_size + output_padding - 2) / 2)
    return padding, output_padding

class BaseUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, deconv_kernal_size):
        super(BaseUpBlock, self).__init__()
        padding, output_padding = select_transconv_param(deconv_kernal_size)

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, 
                               stride=1, padding=1, dilation=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.ConvTranspose2d(out_channel, out_channel, 
                                    kernel_size=deconv_kernal_size, 
                                    stride=2, 
                                    padding=padding, 
                                    output_padding=output_padding, 
                                    bias=False, 
                                    dilation=1)
        self.bn2 = nn.BatchNorm2d(out_channel)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.up(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class UpsampleNet(nn.Module):
    def __init__(self, filter_nums, kernel_sizes, in_channel, up_block=None):
        super(UpsampleNet, self).__init__()
        self.in_channel = in_channel
        if up_block is None:
            up_block = BaseUpBlock
        self.filter_nums = filter_nums
        self.kernel_sizes = kernel_sizes
        self.upsample_layers = self._make_upsample_layer(up_block)
        for m in self.modules():
            init(m)
        
    def _make_upsample_layer(self, up_block):
        in_channel = self.in_channel
        layers = []
        assert len(self.filter_nums) == len(self.kernel_sizes)
        for filter_num, kernal_size in zip(self.filter_nums, self.kernel_sizes) :
            layers.append(up_block(in_channel, filter_num, kernal_size))
            in_channel = filter_num
        self.out_channel = in_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.upsample_layers(x)


def upsample_basic_3(in_channel):
    filter_nums = [256, 128, 64]
    kernal_sizes = [4, 4, 4]
    model = UpsampleNet(filter_nums, kernal_sizes, in_channel, up_block=BaseUpBlock)
    return model