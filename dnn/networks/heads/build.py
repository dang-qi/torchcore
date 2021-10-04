from ....util.registry import Registry
from ....util.build import build_with_config

HEAD_REG = Registry('HEAD')

def build_head(cfg):
    head = build_with_config(cfg, HEAD_REG)
    return head
