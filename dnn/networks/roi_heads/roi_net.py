from collections import OrderedDict
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

from .build import ROI_HEADS_REG

#from torchcore.dnn.networks.tools import box_coder

@ROI_HEADS_REG.register()
class RoINet(nn.Module):
    def __init__(self, box_head, sampler, box_coder, roi_extractor, class_num, box_loss, class_loss, feature_strides, score_thresh=0.01, nms_thresh=0.5, detection_per_image=100, dataset_label=None, iou_low_thresh=0.5, iou_high_thresh=0.5, feature_names=None, mask_head=None):
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
            proposals, target_labels, target_boxes = self.select_proposals(proposals, targets)

        rois = self.roi_align(features, proposals, self.feature_strides)

        label_pre, bbox_pre = self.faster_rcnn_head(rois)
        #return proposals, rois, label_pre, bbox_pre
        if self.training:
            if self.dataset_label is not None:
                # just select the stuff that equals the dataset label to compute loss
                losses, human_box_proposals = self.compute_partial_loss(label_pre, bbox_pre, target_labels, target_boxes, proposals, inputs)
                return losses, human_box_proposals

            label_loss, bbox_loss = self.compute_loss(label_pre, bbox_pre, target_labels, target_boxes, proposals)
            losses = {
                'loss_label': label_loss,
                'loss_roi_bbox': bbox_loss
            }
            return losses
        else:
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
            bbox_pre_image = bbox_pre_image[:, 4:].reshape(bbox_pre_image.shape[0],-1,4)
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


    def compute_loss(self, label_pre, bbox_pre, target_labels, target_boxes, proposals):
        #print('label pre:', label_pre.shape)
        #print('bbox pre:', bbox_pre.shape)
        #print('target labels:', len(target_labels))
        #print('target boxes:', len(target_boxes))
        #print('proposals:', len(proposals))
        #for i in range(len(proposals)):
        #    print('proposals {}:'.format(i), proposals[i].shape)
        target_labels = torch.cat(target_labels, dim=0)
        proposals = torch.cat(proposals, dim=0)
        target_boxes = torch.cat(target_boxes, dim=0)

        pos_ind = torch.where(target_labels>0)[0]
        proposals_pos = proposals[pos_ind]
        target_boxes_pos = target_boxes[pos_ind]
        bbox_pre_pos = bbox_pre[pos_ind]
        label_pos = target_labels[pos_ind]

        #pos_inds = [torch.where(label>0)[0] for label in target_labels]
        #proposals_pos = [proposal_im[ind] for proposal_im, ind in zip(proposals, pos_inds)]
        #target_boxes_pos = [boxes[ind] for boxes, ind in zip(target_boxes, pos_inds)]
        #bbox_pre_pos = [boxes[ind] for boxes, ind in zip(bbox_pre, pos_inds)]
        #label_pos = [label[ind] for label, ind in zip(target_labels, pos_inds)]

        target_boxes = self.box_coder.encode_once(proposals_pos, target_boxes_pos)
        #target_boxes = torch.cat(target_boxes, dim=0)
        
        #bbox_pre_pos = torch.cat(bbox_pre_pos, dim=0)
        #label_pos = torch.cat(label_pos, dim=0)
        bbox_pre_pos = bbox_pre_pos.view(bbox_pre_pos.shape[0], -1, 4)
        #print('bboxe pre pos shape:', bbox_pre_pos.shape)
        ind0 = torch.arange(bbox_pre_pos.shape[0])
        bbox_pre_pos = bbox_pre_pos[ind0, label_pos]
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


    def select_proposals(self, proposals, targets):
        # add gt boxes to proposals
        proposals = self.add_gt_boxes(proposals, targets)
        ind_pos_proposal, ind_neg_proposal, ind_pos_boxes = self.assign_targets_to_proposals(proposals, targets)
        # filter and balance the positive proposals and negtive proposals
        keep_pos, keep_neg = self.sampler.sample_batch(ind_pos_proposal, ind_neg_proposal)
        ind_pos_proposal = [ind_pos[keep] for ind_pos, keep in zip(ind_pos_proposal, keep_pos)]
        ind_neg_proposal = [ind_neg[keep] for ind_neg, keep in zip(ind_neg_proposal, keep_neg)]
        ind_pos_boxes = [ind_pos[keep] for ind_pos, keep in zip(ind_pos_boxes, keep_pos)]
        proposals_out = []
        target_boxes = []
        target_labels = []
        for ind_pos, ind_neg, ind_box, proposal_im, target in zip(
                    ind_pos_proposal, ind_neg_proposal, ind_pos_boxes, proposals, targets):
            pos = proposal_im[ind_pos]
            neg = proposal_im[ind_neg]
            boxes = target['boxes']
            labels_pos = target['labels']
            target_boxes_im = boxes[ind_box]
            boxes_neg = torch.zeros_like(neg)
            target_lables_pos = labels_pos[ind_box]
            target_lables_neg = torch.zeros_like(ind_neg)
            target_boxes.append(torch.cat((target_boxes_im, boxes_neg), dim=0))
            proposals_out.append(torch.cat((pos,neg), dim=0))
            target_labels.append(torch.cat((target_lables_pos, target_lables_neg), dim=0))
        return proposals_out, target_labels, target_boxes

    def add_gt_boxes(self, proposals, targets):
        proposal_out = []
        for proposal_image, target in zip(proposals, targets):
            boxes = target['boxes']
            proposal_image= torch.cat((proposal_image, boxes), dim=0)
            proposal_out.append(proposal_image)
        return proposal_out



    def assign_targets_to_proposals(self, proposals, targets):
        # return the indexes of the matched proposals and matched boxes
        ind_pos_proposal_all = []
        ind_neg_proposal_all = []
        ind_pos_boxes_all = []
        #ind_neg_boxes_all = []
        for proposal_image, target in zip(proposals, targets):
            boxes = target['boxes']
            if len(boxes) == 0:
                raise ValueError('there should be more than one item in each image')
            else:
                iou_mat = box_iou(proposal_image, boxes) # proposal N and target boxes M, iou mat: NxM
                # set up the max iou for each box as positive 
                # set up the iou bigger than a value as positive
                ind_pos_proposal, ind_pos_boxes, ind_neg_proposal = self.match_boxes(iou_mat, low_thresh=self.iou_low_thresh, high_thresh=self.iou_high_thresh)
                ind_pos_proposal_all.append(ind_pos_proposal)
                ind_neg_proposal_all.append(ind_neg_proposal)
                ind_pos_boxes_all.append(ind_pos_boxes)
        return ind_pos_proposal_all, ind_neg_proposal_all, ind_pos_boxes_all

    def match_boxes_old(self, iou_mat, low_thresh, high_thresh):
        # according to faster RCNN paper, the max iou and the one bigger than 
        # high_thresh are positive matches, lower than low_thresh are negtive matches
        # the iou in between are ignored during the training
        # iou mat NxM, N anchor and M target boxes
        N, M = iou_mat.shape
        _, indexes = iou_mat.max(dim=0)
        index_mat = torch.zeros_like(iou_mat)
        ind1 = torch.arange(M)
        #print('index mat shape:', index_mat.shape)
        #print('ind1 shape:', ind1.shape)
        #print('indexes shape:', indexes.shape)
        index_mat[indexes, ind1] = 1
        index_above_thre = torch.where(iou_mat>=high_thresh)
        index_mat[index_above_thre] = 1
        index_pos_anchor, index_pos_boxes = torch.where(index_mat==1)
        index_neg_anchor, index_neg_boxes = torch.where(iou_mat<low_thresh)
        print('index pos boxes:', index_pos_boxes)
        #print('index pos anchor:', index_pos_anchor)
        return index_pos_anchor, index_pos_boxes, index_neg_anchor 
        
    def match_boxes(self, iou_mat, low_thresh, high_thresh, allow_low_quality_match=False):
        # according to faster RCNN paper, the max iou and the one bigger than 
        # high_thresh are positive matches, lower than low_thresh are negtive matches
        # the iou in between are ignored during the training
        # iou mat NxM, N anchor and M target boxes
        #N, M = iou_mat.shape

        # set up the possible weak but biggest anchors that cover boxes
        # There are possible more than one anchors have same biggest 
        # value with the boxes  

        match_box_ind = torch.full_like(iou_mat[:,0], -1, dtype=torch.int64)

        # set the negtive index
        max_val_anchor, max_box_ind = iou_mat.max(dim=1)
        index_neg_anchor = torch.where(max_val_anchor<low_thresh)
        match_box_ind[index_neg_anchor] = -2 # -2 means negtive anchors


        ##TODO stuff added, maybe need to delete these
        #max_val, max_box_ind = iou_mat.max(dim=1)
        #match_box_ind[inds_anchor] = max_box_ind[inds_anchor]

        # set the vague ones
        # if a anchor can match two boxes, still keep the biggest one!!!!
        if  allow_low_quality_match:
            max_val, _ = iou_mat.max(dim=0)
            inds_anchor, ind_box = torch.where(iou_mat==max_val.expand_as(iou_mat))
            #match_box_ind[inds_anchor] = ind_box
            match_box_ind[inds_anchor] = max_box_ind[inds_anchor]

        #index_mat = torch.zeros_like(iou_mat)
        #ind1 = torch.arange(M)
        #print('index mat shape:', index_mat.shape)
        #print('ind1 shape:', ind1.shape)
        #print('indexes shape:', indexes.shape)
        #print('iou mat', iou_mat[inds_max])
        #index_mat[inds_max] = 1
        inds_anchor_above = max_val_anchor>=high_thresh
        match_box_ind[inds_anchor_above] = max_box_ind[inds_anchor_above]
        #index_mat[index_above_thre] = 1


        # the whole thing here is to make sure each anchor only map to one box (not two or more)
        # otherwise we can use: index_pos_anchor, index_pos_boxes = torch.where(index_mat==1)
        #max_val_new, index_pos_boxes = index_mat.max(dim=1)
        #max_val_new = max_val_new > 0
        #index_pos_anchor = index_pos_anchor[max_val_new]
        #index_pos_boxes = index_pos_boxes[max_val_new]
       
        pos_ind = match_box_ind>= 0
        index_pos_boxes = match_box_ind[pos_ind]
        index_pos_anchor = torch.where(pos_ind)[0]
        index_neg_anchor = torch.where(match_box_ind==-2)[0]
        #index_pos_anchor, index_pos_boxes = torch.where(index_mat==1)
        #print('index pos boxes:', index_pos_boxes)
        #print('index pos anchor:', index_pos_anchor)
        return index_pos_anchor, index_pos_boxes, index_neg_anchor