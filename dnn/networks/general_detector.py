from torch import nn

class GeneralDetector(nn.Module):
    def __init__(self, backbone, neck=None, heads=None, cfg=None):
        super(GeneralDetector, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = heads
        self.cfg = cfg

    def forward(self):
        raise NotImplementedError

    def post_process(self):
        raise NotImplementedError

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()