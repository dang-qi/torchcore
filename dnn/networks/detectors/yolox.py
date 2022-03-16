import torch
import math
import time
from collections import OrderedDict
#from .general_detector import GeneralDetector
from .one_stage_detector import OneStageDetector
from torchvision.ops import roi_align, nms

from .build import DETECTOR_REG

@DETECTOR_REG.register(force=True)
class YOLOX(OneStageDetector):
    def __init__(self, backbone, neck=None, det_head=None, training=True, ):
        super(YOLOX, self).__init__(backbone, neck=neck, det_head=det_head, training=training)

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


