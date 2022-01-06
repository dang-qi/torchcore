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

@DETECTOR_REG.register(force=True)
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
                boxes, scores, labels = self.remove_by_hard_grammar(boxes, scores, labels)
                # for debug
                #boxes, scores, labels = self.change_labels_by_hard_grammar(boxes, scores, labels)

            results['boxes'][i] = boxes
            results['scores'][i] = scores
            results['labels'][i] = labels
        return results

    def remove_by_hard_grammar(self, boxes, scores, labels, ior_thresh=0.99, main_box_score_thresh=0.1):
        unreasonable_pairs = self.hard_grammar['unreasonable_pairs']
        reasonable_pairs = self.hard_grammar['reasonable_pairs']
        #print('label min', labels.min())

        # calculate for unreasonable pairs
        keepu = torch.ones_like(labels, dtype=torch.bool)
        for pair_ind in unreasonable_pairs:
            mi, pi = pair_ind
            m_ind = (labels == mi+1)
            p_ind = (labels == pi+1)
            m_boxes = boxes[m_ind] # shape Mx4
            if main_box_score_thresh > 0:
                m_scores = scores[m_ind]
                m_boxes = m_boxes[m_scores>main_box_score_thresh]
            p_boxes = boxes[p_ind] # shape Nx4
            if len(m_boxes)==0 or len(p_boxes)==0:
                continue
            ior = cal_IoR_batch(m_boxes, p_boxes) # shape MxN
            v, _ = ior.max(axis=0) # shape N, get the biggest overlap
            remove_sub = v > ior_thresh
            keep_ind = torch.where(p_ind)[0][remove_sub]
            keepu[keep_ind] = False
            #removed = scores[p_ind][remove_sub]
            #if len(removed)>0:
            #    print('keep part after:',keep)
            #    print('remove scores: {}'.format(removed))
        # calculate for reasonable pairs
        keepr = torch.zeros_like(labels, dtype=torch.bool)
        for pair_ind in reasonable_pairs:
            mi, pi = pair_ind
            m_ind = (labels == mi+1)
            p_ind = (labels == pi+1)
            m_boxes = boxes[m_ind] # shape Mx4
            if main_box_score_thresh > 0:
                m_scores = scores[m_ind]
                m_boxes = m_boxes[m_scores>main_box_score_thresh]
            p_boxes = boxes[p_ind] # shape Nx4
            if len(m_boxes)==0 or len(p_boxes)==0:
                continue
            ior = cal_IoR_batch(m_boxes, p_boxes) # shape MxN
            v, _ = ior.max(axis=0) # shape N, get the biggest overlap
            remove_sub = v > ior_thresh
            keep_ind = torch.where(p_ind)[0][remove_sub]
            keepr[keep_ind] = True

        keep = keepu | keepr

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        #print('box num after:',len(boxes))
        return boxes, scores, labels

            
    # just for debug
    def change_labels_by_hard_grammar(self, boxes, scores, labels, ior_thresh=0.99, main_box_score_thresh=0.2):
        # TODO: consider the main garment score thresh
        unreasonable_pairs = self.hard_grammar['unreasonable_pairs']
        reasonable_pairs = self.hard_grammar['reasonable_pairs']
        #print('label min', labels.min())

        keepu = torch.ones_like(labels, dtype=torch.bool)
        for pair_ind in unreasonable_pairs:
            mi, pi = pair_ind
            m_ind = (labels == mi+1)
            p_ind = (labels == pi+1)
            m_boxes = boxes[m_ind] # shape Mx4
            if main_box_score_thresh > 0:
                m_scores = scores[m_ind]
                m_boxes = m_boxes[m_scores>main_box_score_thresh]
            p_boxes = boxes[p_ind] # shape Nx4
            if len(m_boxes)==0 or len(p_boxes)==0:
                continue
            ior = cal_IoR_batch(m_boxes, p_boxes) # shape MxN
            v, _ = ior.max(axis=0) # shape N, get the biggest overlap
            remove_sub = v > ior_thresh
            keep_ind = torch.where(p_ind)[0][remove_sub]
            keepu[keep_ind] = False
            #removed = scores[p_ind][remove_sub]
            #if len(removed)>0:
            #    print('keep part after:',keep)
            #    print('remove scores: {}'.format(removed))
        #print('box num before:',len(boxes))
        #boxes = boxes[keep]
        #scores = scores[keep]
        keepr = torch.zeros_like(labels, dtype=torch.bool)
        for pair_ind in reasonable_pairs:
            mi, pi = pair_ind
            m_ind = (labels == mi+1)
            p_ind = (labels == pi+1)
            m_boxes = boxes[m_ind] # shape Mx4
            if main_box_score_thresh > 0:
                m_scores = scores[m_ind]
                m_boxes = m_boxes[m_scores>main_box_score_thresh]
            p_boxes = boxes[p_ind] # shape Nx4
            if len(m_boxes)==0 or len(p_boxes)==0:
                continue
            ior = cal_IoR_batch(m_boxes, p_boxes) # shape MxN
            v, _ = ior.max(axis=0) # shape N, get the biggest overlap
            remove_sub = v > ior_thresh
            keep_ind = torch.where(p_ind)[0][remove_sub]
            keepr[keep_ind] = True

        abandon = ~(keepu | keepr)
        labels[abandon] = labels[abandon] + 19
        #print('box num after:',len(boxes))
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
    intersection = torch.maximum(x_max-x_min,torch.zeros_like(x_max))*torch.maximum(y_max-y_min,torch.zeros_like(y_max))
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