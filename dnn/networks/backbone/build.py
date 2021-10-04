from ....util.registry import Registry
from ....util.build import build_with_config

BACKBONE_REG = Registry('BACKBONE')

def build_backbone(cfg):
    backbone = build_with_config(cfg, BACKBONE_REG)
    return backbone
