import math
import torch
from .anchor_box_coder import AnchorBoxesCoder
from .build import BOX_CODER_REG

BOX_CODER_REG.register()
class BBoxesCoder(AnchorBoxesCoder):
    def encode(self, pred_boxes, pos_boxes):
        out = []
        for anchor, target_box in zip(pred_boxes, pos_boxes):
            out.append(self.encode_once(anchor, target_box))

        return out



