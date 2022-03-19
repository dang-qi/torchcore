import torch
from torch import nn

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4,1)

    def forward(self, x):
        x = self.conv(x)
        if self.training:
            loss = (torch.rand_like(x)-x).mean()
            return loss
        else:
            return x
