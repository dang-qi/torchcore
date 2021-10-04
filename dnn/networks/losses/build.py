from ....util.registry import Registry
from ....util.build import build_with_config

LOSS_REG = Registry('LOSS')

def build_loss(cfg):
    loss = build_with_config(cfg, LOSS_REG)
    return loss
