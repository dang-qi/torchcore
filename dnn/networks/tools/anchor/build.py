from .....util.registry import Registry
from .....util.build import build_with_config

ANCHOR_GENERATOR_REG = Registry('ANCHOR_GENERATOR')

def build_anchor_generator(cfg):
    anchor_generator = build_with_config(cfg, ANCHOR_GENERATOR_REG)
    return anchor_generator