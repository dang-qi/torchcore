import copy
from torch import nn
from ....util.registry import Registry
from ..common import init_kaiming
from ....util.build import build_with_config

WEIGHT_INIT_REG = Registry('WEIGHT_INIT')
def update_init_info(m, info):
    assert hasattr(m, '_init_log')
    for name, param in m.named_parameters():
        assert param in m._init_log
        mean = param.data.mean()
        var = param.data.var()
        if mean != m._init_log[param]['mean']:
            m._init_log[param]['mean'] = mean
            m._init_log[param]['var'] = var
            m._init_log[param]['info'] = info

class BaseInit():
    def __init__(self, layer, print_info=True):
        self.layer= layer
        self.print_info = print_info

@WEIGHT_INIT_REG.register(name='kaiming', force=True)
class KaimingInit(BaseInit):
    def __init__(self,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal',
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.a = a
        self.mode = mode
        self.nonliearity = nonlinearity
        self.bias = bias
        self.distribution = distribution

    def __call__(self, module):
        def init(m):
            module_name = m.__class__.__name__
            if module_name in self.layer:
                init_kaiming(m, self.a, self.mode, self.nonliearity, self.bias, self.distribution)
        module.apply(init)
        if hasattr(module,'_init_log'):
            info = '{}, a={}, mode={}, nonlinearity={}, bias={},distribution={}'.format(self.__class__.__name__, self.a,self.mode,self.nonliearity,self.bias,self.distribution)
            update_init_info(module,info)

def initialize(m, init_cfg):
    if isinstance(init_cfg, dict):
        init_cfg = [init_cfg]

    for cfg in init_cfg:
        cfg = copy.deepcopy(cfg)
        init_func = build_with_config(cfg,WEIGHT_INIT_REG)
        # The init func should init all the sub_modules
        init_func(m)
