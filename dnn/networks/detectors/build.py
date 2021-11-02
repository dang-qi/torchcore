from ....util.registry import Registry
from ....util.build import build_with_config

DETECTOR_REG = Registry('DETECTOR')

def build_detector(cfg):
    detector = build_with_config(cfg, DETECTOR_REG)
    return detector
