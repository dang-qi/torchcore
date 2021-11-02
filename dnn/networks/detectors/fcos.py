import torch
import math
import traceback
import sys
from torch import nn

from .one_stage_detector import OneStageDetector
from torchvision.ops import nms
from torchvision.ops.boxes import batched_nms, box_iou
from torch.nn import functional as F

from ..tools.anchor import build_anchor_generator
from ..tools.sampler import build_sampler
from ..tools.box_coder import build_box_coder
from ..losses import build_loss
from ..heads import build_head
from ..detection_heads import build_detection_head

from .build import DETECTOR_REG
#from .tools import AnchorBoxesCoder
#from .tools import PosNegSampler

@DETECTOR_REG.register()
class FCOS(OneStageDetector):
    def __init__(self, backbone, neck, det_head ):
        super(FCOS, self).__init__(backbone, neck=neck, det_head=det_head, training=True)

    def post_process(self, results, inputs):
        for i, (boxes, scores, labels) in enumerate(zip(results['boxes'], results['scores'], results['labels'])):

            if len(boxes)>0:
                if 'scale' in inputs:
                    scale = inputs['scale'][i]
                    if isinstance(scale, float):
                        boxes /= scale
                    else:  # scale w and scale h are seperated
                        boxes[...,0] = boxes[...,0] / scale[0]
                        boxes[...,2] = boxes[...,2] / scale[0]
                        boxes[...,1] = boxes[...,1] / scale[1]
                        boxes[...,3] = boxes[...,3] / scale[1]

            results['boxes'][i] = boxes
            results['scores'][i] = scores
            results['labels'][i] = labels
        return results
