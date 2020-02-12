import torch.nn as nn
from ..common import init

class CommonHead(nn.Module):
    def __init__(self, class_num, in_channel, head_conv_channel):
        super().__init__()
        if head_conv_channel>0:
            fc = nn.Sequential(
                nn.Conv2d(in_channel, head_conv_channel, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv_channel, class_num, kernel_size=3, padding=1, bias=True)
            )
        else:
            fc = nn.Conv2d(in_channel, class_num, kernel_size=3, padding=1, bias=True)
        self.fc = fc

        for m in self.modules():
            init(m)

    def forward(self, x):
        return self.fc(x)
        