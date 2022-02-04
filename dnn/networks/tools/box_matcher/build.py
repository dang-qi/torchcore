from .....util.registry import Registry
from .....util.build import build_with_config

BOX_MATCHER_REG = Registry('BOX_MATCHER')

def build_box_matcher(cfg):
    box_matcher = build_with_config(cfg, BOX_MATCHER_REG)
    return box_matcher