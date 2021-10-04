from ....util.registry import Registry
from ....util.build import build_with_config

ROI_HEADS_REG = Registry('ROI HEAD')

def build_roi_head(cfg):
    roi_head = build_with_config(cfg, ROI_HEADS_REG)
    return roi_head
