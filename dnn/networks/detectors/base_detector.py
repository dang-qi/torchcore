from torch import nn

class BaseDetector(nn.Module):
    def __init__(self,):
        super(BaseDetector, self).__init__()

    def forward(self):
        raise NotImplementedError

    def post_process(self):
        raise NotImplementedError

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()