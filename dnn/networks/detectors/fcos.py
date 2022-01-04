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
    def __init__(self, backbone, neck, det_head, hard_grammar=None ):
        '''
        hard_grammar:
            dict(
                 'unreasonable_pairs':list(tuple(int,int),...))
        '''
        super(FCOS, self).__init__(backbone, neck=neck, det_head=det_head, training=True)
        self.hard_grammar = hard_grammar

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

            if self.hard_grammar is not None:
                boxes, scores, labels = self.remove_by_hard_grammar()

            results['boxes'][i] = boxes
            results['scores'][i] = scores
            results['labels'][i] = labels
        return results

    def remove_by_hard_grammar(self, boxes, scores, labels, ior_thresh=0.99):
        # TODO: consider the main garment score thresh
        unreasonable_pairs = self.hard_grammar['unreasonable_pairs']

        keep = torch.ones_like(labels, dtype=torch.bool)
        for pair_ind in unreasonable_pairs:
            mi, pi = pair_ind
            m_ind = (labels == mi)
            p_ind = (labels == pi)
            m_boxes = boxes[m_ind] # shape Mx4
            p_boxes = boxes[p_ind] # shape Nx4
            ior = cal_IoR_batch(m_boxes, p_boxes) # shape MxN
            v, _ = ior.max(axis=0) # shape N, get the biggest overlap
            remove_sub = v > ior_thresh
            keep[p_ind][remove_sub] = False
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        return boxes, scores, labels

            
            
        

    def get_heatmaps(self, inputs, targets=None):
        if self.training and targets is None:
            raise ValueError('targets should not be None during the training')
        features = self.backbone(inputs['data'])
        if  self.has_neck():
            features = self.neck(features)

        class_map, bbox_map, centerness_map = self.det_head.get_heatmaps(features=features)
        return class_map, bbox_map, centerness_map

def cal_IoR_batch(main_boxes, part_boxes):
    '''
        main_boxes: Mx4
        part_boxes: Nx4
        output IoR: MxN
    '''
    main_boxes = main_boxes.unsqueeze(1)
    x_min = torch.max(main_boxes[...,0], part_boxes[...,0]) # MxN
    y_min = torch.max(main_boxes[...,1], part_boxes[...,1])
    x_max = torch.min(main_boxes[...,2], part_boxes[...,2])
    y_max = torch.min(main_boxes[...,3], part_boxes[...,3])
    intersection = torch.max(x_max-x_min,0)*torch.max(y_max-y_min,0)
    part_area = (part_boxes[...,2]-part_boxes[...,0]) * (part_boxes[...,3]-part_boxes[...,1]) # N
    IoR = intersection / part_area
    return IoR

def cal_IoR(human_box, garment_box):
    '''IoR: Intersection/garment_box_area'''
    x_min = max(human_box[0], garment_box[0])
    y_min = max(human_box[1], garment_box[1])
    x_max = min(human_box[2], garment_box[2])
    y_max = min(human_box[3], garment_box[3])
    if x_max<=x_min or y_max<=y_min:
        return 0
    intersection = (x_max-x_min)*(y_max-y_min)
    garment_area = (garment_box[2]-garment_box[0])*(garment_box[3]-garment_box[1])
    IoR = intersection / garment_area
    assert IoR>0 and IoR<=1
    return IoR