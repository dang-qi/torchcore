from ...util.registry import Registry
from ...util.build import build_with_config

SAMPLER_REG = Registry('DATASET SAMPLER')

def build_sampler(cfg):
    sampler = build_with_config(cfg, SAMPLER_REG)
    return sampler
