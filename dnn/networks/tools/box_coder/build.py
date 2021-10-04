from .....util.registry import Registry
from .....util.build import build_with_config

BOX_CODER_REG = Registry('BoxCoder')

def build_box_coder(cfg):
    box_coder = build_with_config(cfg, BOX_CODER_REG)
    return box_coder