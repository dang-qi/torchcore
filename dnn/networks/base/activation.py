import torch.nn as nn
from ....util.registry import Registry
from ....util.build import build_with_config

ACTIVATION_REG = Registry('ACTIVATION')
ACTIVATION_LAYERS = {'relu':nn.ReLU,
                     'leaky_relu':nn.LeakyReLU,
                     'prelu':nn.PReLU,
                     'rrelu':nn.RReLU,
                     'relu6':nn.ReLU6,
                     'elu':nn.ELU,
                     'sigmoid':nn.Sigmoid,
                     'tanh':nn.Tanh,
                     'swish':nn.SELU,
                     }
for k,v in ACTIVATION_LAYERS.items():
    ACTIVATION_REG.register(v, k)

def build_activation(cfg):
    activation = build_with_config(cfg, ACTIVATION_REG)
    return activation