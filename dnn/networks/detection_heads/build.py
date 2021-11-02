from ....util.registry import Registry
from ....util.build import build_with_config

DETECTION_HEAD_REG = Registry('DETECTION_HEAD')

def build_detection_head(cfg):
    detection_head = build_with_config(cfg, DETECTION_HEAD_REG)
    return detection_head
