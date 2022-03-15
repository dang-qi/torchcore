from turtle import pos
import torch
from torch import nn

from torchcore.dnn.networks.tools.box_matcher.match_result import MatchResult
from ..base import BaseModule
from ..heads.build import build_head
from ..tools.anchor import build_anchor_generator
from ..tools.box_matcher import build_box_matcher
from mmdet.core.bbox.assigners import SimOTAAssigner
import torch.nn.functional as F

class YOLOXHead(BaseModule):
    def __init__(self,
                 num_classes=80,
                 head_cfg=dict(type='YOLOXFeatureHead',
                               in_channels=256,
                               num_classes=80,),
                 strides = [8, 16, 32],
                 prior_generator=dict(type='MultiLevelGridPointGenerator',
                        offset=0,),
                 box_matcher=dict(type='SimOTABoxMatcher',
                        center_radius=2.5,
                        candidate_topk=10,
                        iou_weight=3.,
                        class_weight=1.),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.strides = strides
        prior_generator['strides'] = strides
        self.head = build_head(head_cfg)
        self.prior_generator = build_anchor_generator(prior_generator)
        self.num_classes = num_classes
        self.box_matcher = build_box_matcher(box_matcher)
        self.use_l1 = False


    def init_weights(self):
        super().init_weights()
        if hasattr(self.head, 'init_weights'):
            self.head.init_weights()

    def forward(self,inputs,x,targets=None):
        # list((pred_class, pred_bbox, pred_obj),...)
        pred_out = self.head(x)
        self.use_l1=True

        if self.training:
            return self.get_pred_targets(pred_out, targets)

    def get_pred_targets(self, pred_multi_level, targets):    
        feature_sizes = [p[0].shape[-2:] for p in pred_multi_level]
        dtype = pred_multi_level[0][0].dtype
        device = pred_multi_level[0][0].device
        priors_with_strides = self.prior_generator.multi_level_grid(feature_sizes,dtype=dtype,device=device, with_strides=True)
        N, C, _, _= pred_multi_level[0][0].shape

        cat_pred_box = []
        cat_pred_cls = []
        cat_pred_obj = []
        cat_decode_boxes = []
        for pred, prior_with_strides,stride in zip(pred_multi_level, priors_with_strides, self.strides):
            pred_cls, pred_box, pred_obj = pred
            # flatten the preds NxCxHxW to NxH*WxC
            pred_box = pred_box.permute(0,2,3,1).reshape(N,-1,4)
            #pred_cls = pred_cls.permute(0,2,3,1).reshape(N,-1,C).sigmoid()
            #pred_obj = pred_obj.permute(0,2,3,1).reshape(N,-1).sigmoid()
            pred_cls = pred_cls.permute(0,2,3,1).reshape(N,-1,C)
            pred_obj = pred_obj.permute(0,2,3,1).reshape(N,-1)
            cat_pred_box.append(pred_box)
            cat_pred_cls.append(pred_cls)
            cat_pred_obj.append(pred_obj)

            decode_boxes = self.decode_boxes(pred_box,prior_with_strides,stride)
            cat_decode_boxes.append(decode_boxes)
        cat_pred_box = torch.cat(cat_pred_box,dim=1)
        cat_pred_obj = torch.cat(cat_pred_obj,dim=1)
        cat_pred_cls = torch.cat(cat_pred_cls,dim=1)
        cat_decode_boxes = torch.cat(cat_decode_boxes, dim=1)
        cat_priors_with_strides = torch.cat(priors_with_strides, dim=0)

        # get target for each image seperately
        for i in range(N):
            gt_boxes = targets[i]['boxes']
            gt_labels = targets[i]['labels']
            x =self.get_single_target(cat_pred_cls[i].detach(),cat_pred_obj[i].detach(),cat_pred_box[i].detach(),cat_decode_boxes[i].detach(), cat_priors_with_strides, gt_boxes, gt_labels)
            return x
            #pos_ind, cls_targets, obj_targets,box_targets, l1_targets =self.get_single_target(cat_pred_cls[i].detach(),cat_pred_obj[i].detach(),cat_pred_box[i].detach(),cat_decode_boxes[i].detach(), cat_priors_with_strides, gt_boxes, gt_labels)
            #return pos_ind, cls_targets, obj_targets,box_targets, l1_targets 

    @torch.no_grad()
    def get_single_target(self, pred_cls, pred_obj, pred_box, decode_boxes, prior_with_strides, gt_boxes, gt_labels):
        '''get pred targets for each image'''
        # all the pred are concated predictions from all the FPN level
        #return (pred_cls, pred_obj,pred_box,decode_boxes,prior_with_strides,gt_boxes,gt_labels)
        num_prior = prior_with_strides.size(0)
        gt_num = gt_boxes.size(0)
        # if there is no gt boxes
        if gt_num == 0:
            cls_targets = pred_cls.new_zeros((0, self.num_classes))
            obj_targets = pred_obj.new_zeros((num_prior,1))
            box_targets = pred_box.new_zeros((0,4))
            l1_targets = pred_box.new_zeros((0,4))
            pos_mask = pred_cls.new_zeros(num_prior).bool()
            return pos_mask, cls_targets,obj_targets,box_targets,l1_targets

        offset_priors_with_strides = torch.cat([prior_with_strides[:,:2]+0.5*prior_with_strides[:,2:], prior_with_strides[:,2:]], dim=-1)

        match = self.box_matcher.match(pred_cls.sigmoid()*pred_obj.sigmoid().unsqueeze(-1), offset_priors_with_strides, decode_boxes, gt_boxes, gt_labels )
        pos_ind = match.matched_ind>=0
        pos_num = pos_ind.sum()

        pos_ious = match.max_iou[pos_ind]
        # class targets are IoU awear 
        cls_targets = F.one_hot(gt_labels[match.matched_ind[pos_ind]],num_classes=self.num_classes) * pos_ious.unsqueeze(-1)
        obj_targets = pred_obj.new_zeros((num_prior,1))
        obj_targets[pos_ind] = 1
        box_targets = gt_boxes[match.matched_ind[pos_ind]]
        l1_targets = pred_box.new_zeros((pos_num,4))
        if self.use_l1:
            l1_targets = self._get_l1_targets(box_targets,l1_targets,prior_with_strides[pos_ind])
        return (pos_ind, cls_targets, obj_targets,box_targets, l1_targets)

    def _get_l1_targets(self,box_targets,l1_targets,priors,eps=1e-8):
        '''(xc-Pxc)/w, (yc-Pyc)/h, log(w/stride_w),log(h/stride_h)
            P means predictions, (xc, yc), w, h is ground truth center, width, height
        '''
        box_targets_cxcywh = xyxy2cxcywh(box_targets)
        l1_targets[:,:2] = (box_targets_cxcywh[:,:2]-priors[:,:2])/priors[:,2:]
        l1_targets[:,2:] = torch.log(box_targets_cxcywh[:,2:]/priors[:,2:]+eps)
        return l1_targets


    def decode_boxes(self, pred, prior, stride):
        '''pred: NxHWx4, prior: HWx2'''
        xyc = pred[...,:2]*stride + prior[...,:2]
        wh = pred[...,2:].exp()*stride

        x1 = xyc[...,0]-wh[...,0]/2
        y1 = xyc[...,1]-wh[...,1]/2
        x2 = xyc[...,0]+wh[...,0]/2
        y2 = xyc[...,1]+wh[...,1]/2

        boxes = torch.stack([x1,y1,x2,y2],dim=-1)
        return boxes

def xyxy2cxcywh(boxes):
    new_boxes = torch.empty_like(boxes)
    new_boxes[...,0] = (boxes[...,0]+boxes[...,2])/2
    new_boxes[...,1] = (boxes[...,1]+boxes[...,3])/2
    new_boxes[...,2] = (boxes[...,2]-boxes[...,0])
    new_boxes[...,3] = (boxes[...,3]-boxes[...,1])
    return new_boxes


