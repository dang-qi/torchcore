from collections import OrderedDict
from random import sample
import torch
from torch import nn
import torch.nn.functional as F
from ..pooling import RoiAliagnFPN
from ..tools import PosNegSampler
from ..tools import AnchorBoxesCoder
from torchvision.ops.boxes import batched_nms, box_iou
from ..heads import FastRCNNHead, MaskRCNNHead
from ..tools.sampler import build_sampler
from ..tools.box_coder import build_box_coder
from ..pooling.build import build_pooler
from ..detection_heads.build import build_detection_head
from ..heads.build import build_head
from ..losses.build import build_loss
from ..tools.box_matcher import build_box_matcher

from .build import ROI_HEADS_REG

#from torchcore.dnn.networks.tools import box_coder

@ROI_HEADS_REG.register(force=True)
class RoINetNew(nn.Module):
    def __init__(self, box_head, sampler, box_matcher, box_coder, roi_extractor, class_num, box_loss, class_loss, feature_strides, score_thresh=0.01, nms_thresh=0.5, detection_per_image=100, dataset_label=None, iou_low_thresh=0.5, iou_high_thresh=0.5, feature_names=None, mask_head=None):
        super().__init__()
        #self.cfg = cfg

        self.dataset_label = dataset_label
        self.class_num = class_num
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detection_per_image = detection_per_image

        self.iou_low_thresh = iou_low_thresh
        self.iou_high_thresh = iou_high_thresh

        self.feature_names = feature_names
        self.feature_strides = feature_strides

        #self.pos_neg_sampler = PosNegSampler(pos_num=cfg.pos_sample_num, neg_num=cfg.neg_sample_num)
        self.box_matcher = build_box_matcher(box_matcher)
        self.sampler = build_sampler(sampler)
        self.roi_align = build_pooler(roi_extractor)
        #self.roi_align = RoiAliagnFPN(cfg.pool_h,
        #                              cfg.pool_w,
        #                              sampling=2)
        #self.faster_rcnn_head = FastRCNNHead(cfg)
        self.faster_rcnn_head = build_head(box_head)
        if mask_head is not None:
            self.mask_rcnn_head = build_head(mask_head)
            self.mask_roi_align = build_pooler(mask_head.roi_extractor)
            self.mask_loss = build_loss(mask_head.loss)
        #self.box_coder = AnchorBoxesCoder(box_code_clip=None, weight=cfg.box_weight)
        self.box_coder = build_box_coder(box_coder)
        #self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        #self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum', beta= 1.0 / 9) 
        self.box_loss = build_loss(box_loss) 
        self.class_loss = build_loss(class_loss)
        #self.label_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, proposals, features, targets=None, inputs=None):
        if self.feature_names is not None:
            features = OrderedDict((k,features[k]) for k in self.feature_names)

        if self.training:
            with torch.no_grad():
                sample_all = self.assign_targets_to_proposals(proposals, targets)
                proposals = [torch.cat((sample_result.pos_boxes, sample_result.neg_boxes),dim=0) for sample_result in sample_all]

            rois = self.roi_align(features, proposals, self.feature_strides)

            label_pre, bbox_pre = self.faster_rcnn_head(rois)
            #if self.dataset_label is not None:
            #    # just select the stuff that equals the dataset label to compute loss
            #    losses, human_box_proposals = self.compute_partial_loss(label_pre, bbox_pre, target_labels, target_boxes, proposals, inputs)
            #    return losses, human_box_proposals

            label_loss, bbox_loss = self.compute_loss(label_pre, bbox_pre, sample_all, proposals)
            losses = {
                'loss_label': label_loss,
                'loss_roi_bbox': bbox_loss
            }
            return losses
        else:
            rois = self.roi_align(features, proposals, self.feature_strides)

            label_pre, bbox_pre = self.faster_rcnn_head(rois)
            results = self.inference_result(label_pre, bbox_pre, proposals)
            return results

    def compute_partial_loss(self, label_pre, bbox_pre, target_labels, target_boxes, proposals, inputs):
        dataset_ind = self.get_dataset_ind(inputs)
        assert len(dataset_ind) == len(target_boxes)
        box_num_per_im = [len(box) for box in target_boxes] 
        label_pre = torch.split(label_pre, box_num_per_im, dim=0)
        bbox_pre = torch.split(bbox_pre, box_num_per_im, dim=0)
        label_pre_loss =[target for target,label in zip(label_pre, dataset_ind) if label ]
        bbox_pre_loss =[target for target,label in zip(bbox_pre, dataset_ind) if label ]
        target_boxes =[target for target,label in zip(target_boxes, dataset_ind) if label ]
        target_labels =[target for target,label in zip(target_labels, dataset_ind) if label ]
        proposals_loss =[target for target,label in zip(proposals, dataset_ind) if label ]
        label_pre_loss = torch.cat(label_pre_loss, dim=0)
        bbox_pre_loss = torch.cat(bbox_pre_loss, dim=0)

        label_loss, bbox_loss = self.compute_loss(label_pre_loss, bbox_pre_loss, target_labels, target_boxes, proposals_loss)

        losses = {
            'loss_label': label_loss,
            'loss_roi_bbox': bbox_loss
        }

        label_pre_infer =[target for target,label in zip(label_pre, dataset_ind) if not label ]
        bbox_pre_infer =[target for target,label in zip(bbox_pre, dataset_ind) if not label ]
        proposals_infer =[target for target,label in zip(proposals, dataset_ind) if not label ]

        if len(label_pre_infer) > 0:
            label_pre_infer = torch.cat(label_pre_infer, dim=0)
            bbox_pre_infer = torch.cat(bbox_pre_infer, dim=0)

        results = self.inference_result(label_pre_infer, bbox_pre_infer, proposals_infer)

        return losses, results['boxes']


    def get_dataset_ind(self, inputs):
        return inputs['dataset_label'] == self.dataset_label

    def remove_small_boxes(self, boxes, min_area=0.1):
        area = (boxes[:,3]-boxes[:,1]) * (boxes[:,2]-boxes[:,0])
        keep = torch.where(area > min_area)[0]
        return keep
        
    def inference_result(self, label_pre, bbox_pre, proposals):
        results = {'boxes':[], 'labels':[], 'scores':[]}
        if len(label_pre) == 0:
            return results
        boxes_per_im = [len(proposal) for proposal in proposals]
        label_pre = label_pre.split(boxes_per_im)
        bbox_pre = bbox_pre.split(boxes_per_im)
        for label_pre_image, bbox_pre_image, proposal in zip(label_pre, bbox_pre, proposals):
            label_pre_image = F.softmax(label_pre_image, dim=1)

            # ignore the background class
            #proposal = proposal[:, 4:].view(-1,4)
            bbox_pre_image = bbox_pre_image.reshape(bbox_pre_image.shape[0],-1,4)
            proposal = proposal.unsqueeze(1).expand_as(bbox_pre_image)
            bbox_pre_image = bbox_pre_image.reshape(-1, 4)
            proposal = proposal.reshape(-1, 4)
            boxes = self.box_coder.decode_once(bbox_pre_image, proposal)
            #boxes = proposal
            scores = label_pre_image[:, 1:]
            #scores = scores[:, 1:]
            labels = torch.arange(1, self.class_num+1).expand_as(scores)

            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            #scores, labels = torch.max(label_pre_image, dim=1)
            pos_label_ind = torch.where(scores > self.score_thresh)
            labels = labels[pos_label_ind]
            scores = scores[pos_label_ind]
            boxes = boxes[pos_label_ind]

            # remove small boxes
            keep = self.remove_small_boxes(boxes, min_area=0.1)
            labels = labels[keep]
            scores = scores[keep]
            boxes = boxes[keep]

            # perform nms for each class
            keep = batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[:self.detection_per_image]
            labels = labels[keep]
            scores = scores[keep]
            boxes = boxes[keep]

            results['boxes'].append(boxes)
            results['labels'].append(labels)
            results['scores'].append(scores)
        return results


    def compute_loss(self, label_pre, bbox_pre, sample_all, proposals):
        # label_pre: sample_numx(C+1), the roi align break the batch already
        # bbox_pre: sample_numx4*(C)

        proposals_pos = []
        target_boxes = []
        target_labels = []
        for sample_result in sample_all:
            pos = sample_result.pos_boxes
            target_boxes_im = sample_result.pos_gt_boxes
            target_lables_pos = sample_result.pos_labels
            target_lables_neg = torch.zeros_like(sample_result.neg_ind)
            target_boxes.append(target_boxes_im)
            proposals_pos.append(pos)
            target_labels.append(torch.cat((target_lables_pos, target_lables_neg), dim=0))

        target_labels = torch.cat(target_labels, dim=0)
        proposals = torch.cat(proposals, dim=0)
        proposals_pos = torch.cat(proposals_pos, dim=0)
        target_boxes = torch.cat(target_boxes, dim=0)

        pos_ind = torch.where(target_labels>0)[0]
        bbox_pre_pos = bbox_pre[pos_ind]
        label_pos = target_labels[pos_ind]

        target_boxes = self.box_coder.encode_once(proposals_pos, target_boxes)

        bbox_pre_pos = bbox_pre_pos.view(bbox_pre_pos.shape[0], -1, 4)
        #print('bboxe pre pos shape:', bbox_pre_pos.shape)
        ind0 = torch.arange(bbox_pre_pos.shape[0])
        # I think I should change here to 
        # bbox_pre_pos = bbox_pre_pos[ind0, label_pos-1]
        # to make bbox pred 4*category_num instead of 4*(category_num+1) 
        #bbox_pre_pos = bbox_pre_pos[ind0, label_pos] 
        bbox_pre_pos = bbox_pre_pos[ind0, label_pos-1] 

        #bbox_pre_pos = torch.index_select(bbox_pre_pos, dim=1, index=label_pos)
        #print('label pos ', label_pos)
        #print('label pos shape:', label_pos.shape)
        #print('bboxe pre pos shape:', bbox_pre_pos.shape)


        #label_pre = torch.cat(label_pre, dim=0)
        #print('label pre', label_pre.shape)
        #print('target labels', target_labels.shape)

        #bbox_loss = self.smooth_l1_loss(bbox_pre_pos, target_boxes) / label_pos.numel()
        #weight=torch.tensor([10,10,5,5], dtype=bbox_pre_pos.dtype, device=bbox_pre_pos.device).expand_as(bbox_pre_pos)
        #bbox_loss = self.smooth_l1_loss(bbox_pre_pos*weight, target_boxes*weight) / target_labels.numel()
        bbox_loss = self.box_loss(bbox_pre_pos, target_boxes) / target_labels.numel()
        #bbox_loss = self.smooth_l1_loss(bbox_pre_pos, target_boxes) 
        label_loss = self.class_loss(label_pre, target_labels)
        return label_loss, bbox_loss

    def assign_targets_to_proposals(self, anchors, targets):
        # return the indexes of the matched anchors and matched boxes
        sample_all = []
        for anchor_image, target in zip(anchors, targets):
            boxes = target['boxes']
            labels = target['labels']
            match_result = self.box_matcher.match(boxes, anchor_image, gt_labels=labels)
            sample_result = self.sampler.sample(match_result, boxes, anchor_image, labels)
            sample_all.append(sample_result)
        return sample_all
