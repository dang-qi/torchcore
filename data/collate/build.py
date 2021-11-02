from ...util.registry import Registry
from ...util.build import build_with_config

COLLATE_REG = Registry('CollateFn')

def build_collate(cfg):
    collate = build_with_config(cfg, COLLATE_REG)
    return collate
