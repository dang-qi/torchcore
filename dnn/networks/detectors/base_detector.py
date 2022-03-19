from ..base.base_module import BaseModule
from torch import nn

class BaseDetector(BaseModule):
    def __init__(self,init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg=init_cfg)

    def forward(self):
        raise NotImplementedError

    def post_process(self):
        raise NotImplementedError

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def has_neck(self):
        return self.neck is not None