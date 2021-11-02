from ....util.registry import Registry
from ....util.build import build_with_config

POOLER_REG = Registry('POOLER')

def build_pooler(cfg):
    pooler = build_with_config(cfg, POOLER_REG)
    return pooler
