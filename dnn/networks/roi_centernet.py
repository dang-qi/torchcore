import torch
from torch import nn
import torch.nn.functional as F
from .pooling import RoiAliagnFPN
from .tools import PosNegSampler
from .tools import AnchorBoxesCoder
from torchvision.ops.boxes import batched_nms, box_iou
from .heads import FastRCNNHead

class RoICenterNet(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg

        self.pos_neg_sampler = PosNegSampler(pos_num=127, neg_num=127)
        self.roi_align = RoiAliagnFPN(cfg.roi_pool.pool_h,
                                      cfg.roi_pool.pool_w,
                                      sampling=-1)
        self.faster_rcnn_head = FastRCNNHead(cfg)
        self.box_coder = AnchorBoxesCoder(box_code_clip=None)
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.label_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, proposals, features, strides, targets=None):
        if self.training:
            proposals, target_labels, target_boxes = self.select_proposals(proposals, targets)

        rois, proposals, inds= self.roi_align(features, proposals, strides)

        label_pre, bbox_pre = self.faster_rcnn_head(rois)
        if self.training:
            target_labels = [target_label[ind] for target_label, ind in zip(target_labels, inds)]
            target_boxes = [target_box[ind] for target_box, ind in zip(target_boxes, inds)]

            label_loss, bbox_loss = self.compute_loss(label_pre, bbox_pre, target_labels, target_boxes, proposals)
            losses = {
                'loss_label': label_loss,
                'loss_roi_bbox': bbox_loss
            }
            return losses
        else:
            results = self.inference_result(label_pre, bbox_pre, proposals)
            return results

    def remove_small_boxes(self, boxes, min_area=0.1):
        area = (boxes[:,3]-boxes[:,1]) * (boxes[:,2]-boxes[:,0])
        keep = torch.where(area > min_area)[0]
        return keep
        
    def inference_result(self, label_pre, bbox_pre, proposals):
        results = {'boxes':[], 'labels':[], 'scores':[]}
        for label_pre_image, bbox_pre_image, proposal in zip(label_pre, bbox_pre, proposals):
            label_pre_image = F.softmax(label_pre_image, dim=1)

            # ignore the background class
            #proposal = proposal[:, 4:].view(-1,4)
            bbox_pre_image = bbox_pre_image[:, 4:].reshape(bbox_pre_image.shape[0],-1,4)
            proposal = proposal.unsqueeze(1).expand_as(bbox_pre_image)
            bbox_pre_image = bbox_pre_image.reshape(-1, 4)
            proposal = proposal.reshape(-1, 4)
            boxes = self.box_coder.decode_once(bbox_pre_image, proposal)
            boxes = proposal
            scores = label_pre_image[:, 1:]
            #scores = scores[:, 1:]
            labels = torch.arange(1, self.cfg.class_num+1).expand_as(scores)

            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            #scores, labels = torch.max(label_pre_image, dim=1)
            pos_label_ind = torch.where(scores > self.cfg.roi_head.score_thre)
            labels = labels[pos_label_ind]
            scores = scores[pos_label_ind]
            boxes = boxes[pos_label_ind]

            # remove small boxes
            keep = self.remove_small_boxes(boxes, min_area=0.1)
            labels = labels[keep]
            scores = scores[keep]
            boxes = boxes[keep]

            # perform nms for each class
            keep = batched_nms(boxes, scores, labels, self.cfg.nms_thresh)
            keep = keep[:300]
            labels = labels[keep]
            scores = scores[keep]
            boxes = boxes[keep]

            results['boxes'].append(boxes)
            results['labels'].append(labels)
            results['scores'].append(scores)
        return results


    def compute_loss(self, label_pre, bbox_pre, target_labels, target_boxes, proposals):
        pos_inds = [torch.where(label>0)[0] for label in target_labels]
        proposals_pos = [proposal_im[ind] for proposal_im, ind in zip(proposals, pos_inds)]
        target_boxes_pos = [boxes[ind] for boxes, ind in zip(target_boxes, pos_inds)]
        bbox_pre_pos = [boxes[ind] for boxes, ind in zip(bbox_pre, pos_inds)]
        label_pos = [label[ind] for label, ind in zip(target_labels, pos_inds)]

        target_boxes = self.box_coder.encode(proposals_pos, target_boxes_pos)
        target_boxes = torch.cat(target_boxes, dim=0)
        
        bbox_pre_pos = torch.cat(bbox_pre_pos, dim=0)
        label_pos = torch.cat(label_pos, dim=0)
        bbox_pre_pos = bbox_pre_pos.view(bbox_pre_pos.shape[0], -1, 4)
        #print('bboxe pre pos shape:', bbox_pre_pos.shape)
        ind0 = torch.arange(bbox_pre_pos.shape[0])
        bbox_pre_pos = bbox_pre_pos[ind0, label_pos]
        #bbox_pre_pos = torch.index_select(bbox_pre_pos, dim=1, index=label_pos)
        #print('label pos ', label_pos)
        #print('label pos shape:', label_pos.shape)
        #print('bboxe pre pos shape:', bbox_pre_pos.shape)


        target_labels = torch.cat(target_labels, dim=0)
        label_pre = torch.cat(label_pre, dim=0)
        #print('label pre', label_pre.shape)
        #print('target labels', target_labels.shape)

        bbox_loss = self.smooth_l1_loss(bbox_pre_pos, target_boxes)
        label_loss = self.label_loss(label_pre, target_labels)
        return label_loss, bbox_loss


    def select_proposals(self, proposals, targets):
        # add gt boxes to proposals
        proposals = self.add_gt_boxes(proposals, targets)
        ind_pos_proposal, ind_neg_proposal, ind_pos_boxes = self.assign_targets_to_proposals(proposals, targets)
        # filter and balance the positive proposals and negtive proposals
        keep_pos, keep_neg = self.pos_neg_sampler.sample_batch(ind_pos_proposal, ind_neg_proposal)
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
                ind_pos_proposal, ind_pos_boxes, ind_neg_proposal, _ = self.match_boxes(iou_mat, low_thresh=0.3, high_thresh=0.7)
                ind_pos_proposal_all.append(ind_pos_proposal)
                ind_neg_proposal_all.append(ind_neg_proposal)
                ind_pos_boxes_all.append(ind_pos_boxes)
        return ind_pos_proposal_all, ind_neg_proposal_all, ind_pos_boxes_all

    def match_boxes(self, iou_mat, low_thresh, high_thresh):
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
        index_neg_anchor, index_neg_boxes = torch.where(iou_mat<=low_thresh)
        #print('index pos boxes:', index_pos_boxes)
        #print('index pos anchor:', index_pos_anchor)
        return index_pos_anchor, index_pos_boxes, index_neg_anchor, index_neg_boxes
        