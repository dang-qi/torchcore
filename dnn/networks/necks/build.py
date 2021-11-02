from ....util.registry import Registry
from ....util.build import build_with_config

NECK_REG = Registry('NECK')

def build_neck(cfg):
    neck = build_with_config(cfg, NECK_REG)
    return neck
