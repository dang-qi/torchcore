from ...util.registry import Registry
from ...util.build import build_with_config

TRANSFORM_REG = Registry('DATA TRANSFORM')

def build_transform(cfg):
    transform = build_with_config(cfg, TRANSFORM_REG)
    return transform
