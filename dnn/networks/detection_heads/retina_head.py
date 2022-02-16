import torch
import math
from torch import nn
#from ..heads.retina_head import RetinaHead
from ..tools import AnchorBoxesCoder
from ..tools.box_coder.build import build_box_coder
#from ..rpn import MyRegionProposalNetwork, RegionProposalNetwork
from torchvision.ops.boxes import batched_nms, box_area, box_iou, nms
from ..losses import FocalLossSigmoid,FocalLoss,SigmoidFocalLoss
from ..losses.build import build_loss
from ..heads.build import build_head 
from ..tools.anchor.build import build_anchor_generator
from .build import DETECTION_HEAD_REG
from ..tools.box_matcher.build import build_box_matcher

# for debug
from ....tools.memory_tools import print_mem
import torch.distributed as dist

@DETECTION_HEAD_REG.register(force=True)
class RetinaNetHead(nn.Module):
    def __init__(self, 
                 head, 
                 anchor_generator, 
                 nms_thresh, 
                 score_thresh,
                 class_loss=dict(
                     type='SigmoidFocalLoss',
                     gamma=2.0,
                     alpha=0.25,
                     reduction='sum'
                     ),
                 box_matcher=dict(
                     type='MaxIoUBoxMatcher',
                     high_thresh=0.5,
                     low_thresh=0.4,
                     allow_low_quality_match=True,
                     assign_all_gt_max=True,
                     keep_max_iou_in_low_quality=True
                     ),
                 #loss_bbox=dict(type='SmoothL1Loss', beta=1.0/9),
                 box_loss=dict(type='L1Loss',reduction='sum'),
                 box_coder=dict(type='AnchorBoxesCoder',box_code_clip=math.log(1000./16)),
                 post_clip=True,
                 before_nms_top_n_train=1000,
                 before_nms_top_n_test=1000,
                 post_nms_top_n_train=1000,
                 post_nms_top_n_test=100):
        super(RetinaNetHead, self).__init__()
        #self.cfg = cfg
        self.head = build_head(head)
        #self.anchor_generater = anchor_generator
        self.anchor_generater = build_anchor_generator(anchor_generator)
        #self.box_coder = AnchorBoxesCoder(box_code_clip=math.log(1000./16))
        self.box_coder = build_box_coder(box_coder)
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.box_matcher = build_box_matcher(box_matcher)
        #self.class_loss = FocalLossSigmoid(alpha=0.25, gamma=2)
        #self.class_loss = SigmoidFocalLoss(alpha=0.25, gamma=2, reduction='sum')
        self.loss_class = build_loss(class_loss)
        self.loss_bbox = build_loss(box_loss)
        #self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum', beta= 1.0 / 9) 
        self.post_clip = post_clip
        self.post_nms_top_n_train = post_nms_top_n_train
        self.post_nms_top_n_test = post_nms_top_n_test

        self.before_nms_top_n_train = before_nms_top_n_train
        self.before_nms_top_n_test = before_nms_top_n_test

    def forward_old(self, inputs, features, targets=None):
        # convert features to list
        if isinstance(features, dict):
            features = list(features.values())
        elif torch.is_tensor(features):
            features = [features]

        pred_out = self.head(features)
        anchors = self.anchor_generater(inputs, features)
        
        num_images = len(anchors)
        num_anchors_per_level = [pred[1][0].numel()//4 for pred in pred_out]
        pred_class, pred_bbox_deltas = self.combine_and_permute_predictions(pred_out)
        #return features,pred_class, pred_bbox_deltas

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        #boxes, scores = self.filter_proposals(proposals, pred_class.detach(), inputs['image_sizes'], num_anchors_per_level)

        if not self.training:
            proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
            proposals = proposals.view(num_images, -1, 4)

            image_shapes = inputs['image_sizes']
            results = self.post_detection(pred_class, proposals, image_shapes, num_anchors_per_level)
            return results
        else:
            ind_pos_anchor, ind_neg_anchor, ind_pos_boxes = self.assign_targets_to_anchors(anchors, targets)
            #print('matched pos anchor num is:', sum([len(ind) for ind in ind_pos_anchor]))
            #boxes = [target['boxes'] for target in targets]
            #print('all the boxes are:', boxes)
            #generate regression target
            pos_boxes = [target['boxes'][ind] for target, ind in zip(targets, ind_pos_boxes)]
            pos_anchor = [target[ind] for target, ind in zip(anchors, ind_pos_anchor)]
            with torch.no_grad():
                regression_targets  = self.box_coder.encode(pos_anchor, pos_boxes )

            #generate classification target
            pos_labels = [target['labels'][ind] for target, ind in zip(targets, ind_pos_boxes)]
            #pos_labels = torch.cat(pos_labels, dim=0)
            #print('regression targets shapes',[t.shape for t in regression_targets])
            #print('regression targets:', regression_targets)

            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                pred_class, pred_bbox_deltas, ind_pos_anchor, ind_neg_anchor, regression_targets, pos_labels)
            #print('loss')
            losses = {
                "loss_objectness": loss_objectness,
                "loss_box_reg": loss_rpn_box_reg,
            }
            return losses
    def forward(self, inputs, features, targets=None):
        # convert features to list
        if isinstance(features, dict):
            features = list(features.values())
        elif torch.is_tensor(features):
            features = [features]

        #rank = dist.get_rank()
        #print('rank:', rank,inputs['data'].shape)

        #print_mem(msg='before head')
        pred_out = self.head(features)
        #print_mem(msg='after head')
        #print_mem(msg='after anchor')
        
        #num_images = len(anchors)
        #num_anchors_per_level = [pred[1][0].numel()//4 for pred in pred_out]
        #print_mem(msg='after permute')
        #return features,pred_class, pred_bbox_deltas

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        #boxes, scores = self.filter_proposals(proposals, pred_class.detach(), inputs['image_sizes'], num_anchors_per_level)

        if not self.training:
            #proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
            #proposals = proposals.view(num_images, -1, 4)

            anchors = self.anchor_generater(inputs, features, keep_multi_level=True)
            image_shapes = inputs['image_sizes']
            pred_class, pred_bbox_deltas = self.combine_and_permute_predictions(pred_out, keep_multi_level=True)
            results = self.post_detection(pred_class, pred_bbox_deltas, anchors, image_shapes )
            return results
        else:
            # list(MatchResult)
            anchors = self.anchor_generater(inputs, features)
            matches = self.assign_targets_to_anchors(anchors, targets)
            #print_mem(msg='after match')

            #anchor_by_gt = [len(a)*len(t['boxes']) for a, t in zip(anchors, targets)]
            #max_match = [(m.matched_ind>=-1).sum() for m in matches]
            #anchor_by_gt = max(anchor_by_gt)
            #max_match = sum(max_match)
            #if anchor_by_gt> self.max_anchor_by_gt_num:
            #    print('max anchor by gt change from to', self.max_anchor_by_gt_num, anchor_by_gt)
            #    self.max_anchor_by_gt_num = anchor_by_gt
            #if max_match > self.max_match:
            #    print('max match change from to',self.max_match, max_match)
            #    self.max_match = max_match
            pred_class, pred_bbox_deltas = self.combine_and_permute_predictions(pred_out, keep_multi_level=False)

            loss_objectness, loss_rpn_box_reg = self.compute_loss( targets, pred_class, pred_bbox_deltas, matches, anchors)
            #print_mem(msg='after loss')
            #print('loss')
            losses = {
                "loss_objectness": loss_objectness,
                "loss_box_reg": loss_rpn_box_reg,
            }
            return losses

    def compute_loss(self, targets, pred_class, pred_bbox_deltas, matches, anchors):
        loss_class_all = []
        loss_box_all = []
        # computer for per image and merge in the end
        for target_per_im,  pred_class_per_im, pred_bbox_delta_per_im, match_per_im, anchors_per_im in zip(targets, pred_class, pred_bbox_deltas, matches, anchors):
            matched_ind = match_per_im.matched_ind
            matched_labels = match_per_im.labels
            valid_ind = matched_ind >= match_per_im.NEGATIVE_MATCH
            pos_ind = matched_ind >= 0
            pos_num = pos_ind.sum()
            valid_num = valid_ind.sum()
            #neg_ind = matched_ind == match_per_im.NEGATIVE_MATCH
            pred_class_valid = pred_class_per_im[valid_ind]

            # to use mm detection focal loss the label of negative sample should be class_num+1, class_label should start from ZERO
            class_num = pred_class_per_im.shape[-1]
            gt_label_per_im = matched_labels.new_full((valid_num,), class_num)
            gt_label_per_im[pos_ind[valid_ind]] = matched_labels[pos_ind]-1
            #gt_label_per_im = torch.zeros_like(pred_class_valid,dtype=torch.long)
            #gt_pos_ind = matched_ind[valid_ind]>=0
            #gt_label_per_im[gt_pos_ind]=gt_label_per_im.new_zeros((pos_num,pred_class_valid.shape[1])).scatter_(1, (matched_labels[pos_ind]-1).view(-1,1), 1)
            #print(gt_label_per_im[matched_ind[valid_ind]<0][0])
            #print('pos num', pos_num)
            #print('all num', valid_ind.sum())
            #loss_class_all.append(self.loss_class(pred_class_valid, gt_label_per_im)/pos_num)
            loss_class_all.append(self.loss_class(pred_class_valid, gt_label_per_im, avg_factor=pos_num))

            pred_boxes_pos = pred_bbox_delta_per_im[pos_ind]
            pos_anchor = anchors_per_im[pos_ind]
            pos_boxes = target_per_im['boxes'][matched_ind[pos_ind]]
            with torch.no_grad():
                regression_targets  = self.box_coder.encode_once(pos_anchor, pos_boxes )

            loss_box_all.append(self.loss_bbox(
                pred_boxes_pos,
                regression_targets,
            ) / max(1, pred_boxes_pos.shape[0]))

            
        loss_class = sum(loss_class_all) / len(loss_class_all)
        loss_box = sum(loss_box_all)/len(loss_box_all)

        return loss_class, loss_box

    def compute_loss_old(self, pred_class, pred_bbox_deltas, ind_pos_anchor, ind_neg_anchor, regression_targets, pos_labels):

        #keep_pos, keep_neg = self.pos_neg_sampler.sample_batch(ind_pos_anchor, ind_neg_anchor)

        ######DEBUG
        #matched_idxs = [torch.full((len(pred_class_single),), -1, dtype=torch.float32, device=pred_class_single.device) for pred_class_single in pred_class]
        #for i in range(len(matched_idxs)):
        #    matched_idxs[i][ind_pos_anchor[i]]=1
        #    matched_idxs[i][ind_neg_anchor[i]]=0
        #torch.manual_seed(0)
        #sampled_pos, sampled_neg = self.tv_sampler(matched_idxs)
        #print('keep pos', ind_pos_anchor[0][keep_pos[0]])
        #print('tv keep pos', torch.where(sampled_pos[0]))
        #seta = set(ind_pos_anchor[0][keep_pos[0]].numpy().tolist())
        #setb = set(torch.where(sampled_pos[0])[0].numpy().tolist())
        #print(seta-setb)
        #print(setb-seta)
        #print(seta.difference(setb))
        #######

        loss_class_list = []
        for pred_class_per_im, ind_pos_anchor_per_im, pos_labels_per_im, ind_neg_anchor_per_im in zip(pred_class,ind_pos_anchor, pos_labels, ind_neg_anchor):
            pred_class_pos = pred_class_per_im[ind_pos_anchor_per_im]
            #pred_class_pos = [pred_pos[ind] for pred_pos, ind in zip(pred_class, ind_pos_anchor)]
            #pred_class_pos = torch.cat(pred_class_pos, dim=0)

            label_pos = torch.zeros_like(pred_class_pos)
            label_pos.scatter_(1,(pos_labels_per_im-1).view(-1,1), 1.0)
            pred_class_neg = pred_class_per_im[ind_neg_anchor_per_im]
            #pred_class_neg = [pred_neg[ind] for pred_neg, ind in zip(pred_class, ind_neg_anchor)]
            #pred_class_neg = torch.cat(pred_class_neg, dim=0)
            label_neg = torch.zeros_like(pred_class_neg)
            label_pred = torch.cat((pred_class_pos, pred_class_neg), dim=0)
            label_target = torch.cat((label_pos, label_neg), dim=0)

            #loss_class = F.binary_cross_entropy_with_logits(label_pred, label_target)
            #label_pred = torch.sigmoid_(label_pred)
            loss_class_list.append(self.loss_class(label_pred, label_target) / label_pos.size()[0])
        loss_class = sum(loss_class_list) / len(loss_class_list)
        #print('label pos shape:', label_pos.shape)
        #print('targets shape:', regression_targets.shape)
        #loss_box = self.smooth_l1_loss(pred_bbox_deltas, regression_targets) / label_pos.numel()

        loss_box_list = []
        #regression_targets = torch.cat(regression_targets, dim=0)
        for pred_bbox_per_im, ind_pos_per_im, regression_targets_per_im in zip(pred_bbox_deltas, ind_pos_anchor, regression_targets):
            pred_bbox_delta = pred_bbox_per_im[ind_pos_per_im]

        #pred_bbox_deltas = [pred_box[ind] for pred_box, ind in zip(pred_bbox_deltas, ind_pos_anchor)]
        #pred_bbox_deltas = torch.cat(pred_bbox_deltas, dim=0)
        # TODO this is the loss from torchvision, reduced the wight of box loss
            #loss_box_list.append(self.smooth_l1_loss(pred_bbox_delta, regression_targets_per_im) / pred_bbox_delta.shape[0])
            loss_box_list.append(torch.nn.functional.l1_loss(
                pred_bbox_delta,
                regression_targets_per_im,
                reduction='sum'
            ) / max(1, pred_bbox_delta.shape[0]))
        loss_box = sum(loss_box_list)/len(loss_box_list)
        #return label_pred, label_target
        #loss_box = self.smooth_l1_loss(pred_bbox_deltas, regression_targets) 
        #loss_box = det_utils.smooth_l1_loss(
        #    pred_bbox_deltas,
        #    regression_targets,
        #    beta=1 / 9,
        #    size_average=False,
        #) / (ind_pos_anchor.numel())
        return loss_class, loss_box

    def post_detection(self, pred_class, pred_bbox, anchors, image_shapes):
        '''
            pred_class:[level_1(shape:NxAxC), level2, ...]
            N is batch size, A is anchor size, C is category number
            pred_bbox:[level_1(shape:NxAx4), level2, ...]
            anchors: [image1[level1, level2,...], image2,...]
            anchors:[anchors_per_image([anchors_per_level])]'''
        pred_class = [torch.sigmoid_(p) for p in pred_class]
        
        batch_size = pred_class[0].shape[0]

        boxes_all = []
        labels_all = []
        scores_all = []
        # do post_detection for each image:
        for i in range(batch_size):
            pred_class_image = [p[i] for p in pred_class]
            pred_bbox_image = [p[i] for p in pred_bbox]
            anchor = anchors[i]
            image_shape= image_shapes[i]

            boxes, scores, labels = self.post_detection_single_image(pred_class_image, pred_bbox_image, anchor, image_shape)

            boxes_all.append(boxes)
            scores_all.append(scores)
            labels_all.append(labels+1) # labels start from 1

        results = dict()
        results['boxes'] = boxes_all
        results['scores'] = scores_all
        results['labels'] = labels_all
        return results


    def post_detection_single_image(self, pred_class_image, pred_bbox_image, anchor, image_shape):

        scores_all = []
        labels_all = []
        boxes_all = []
        # do it per level
        for pred_class_per_level, pred_bbox_per_level, anchor_per_level in zip(pred_class_image, pred_bbox_image, anchor):
            keep = pred_class_per_level > self.score_thresh
            scores_per_level = pred_class_per_level[keep]
            keep_inds = torch.nonzero(keep)

            keep_num = min(len(scores_per_level), self.get_before_nms_top_n())
            scores_per_level, scores_ind = scores_per_level.sort(descending=True)
            scores_per_level = scores_per_level[:keep_num]
            keep_inds = keep_inds[scores_ind[:keep_num]]
            keep_inds, labels_per_level = keep_inds.unbind(dim=1)

            pred_bbox_per_level = pred_bbox_per_level[keep_inds]
            anchor_per_level = anchor_per_level[keep_inds]
            boxes_per_level = self.box_coder.decode_once(pred_bbox_per_level, anchor_per_level)

            # crop the box inside the image
            if self.post_clip:
                boxes_per_level = self.crop_boxes(boxes_per_level, image_shape)

            ## remove super small boxes
            #keep = self.remove_small_boxes(boxes_per_level, min_size=1e-2)
            #boxes_per_level = boxes_per_level[keep]
            #scores_per_level = scores_per_level[keep]
            #labels_per_level = labels_per_level[keep]

            scores_all.append(scores_per_level)
            labels_all.append(labels_per_level)
            boxes_all.append(boxes_per_level)
            
        # do it with multi level merged
        boxes = torch.cat(boxes_all)
        labels = torch.cat(labels_all)
        scores = torch.cat(scores_all)
        keep = batched_nms(boxes, scores,labels, self.nms_thresh)
        keep = keep[:self.get_post_nms_top_n()]

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        return boxes, scores, labels



    def post_detection_old(self, pred_class, pred_bbox, image_shapes, num_anchors_per_level):
        # pre nms for each feature layers in each image
        # crop the boxes so they are inside the image
        # delete very small boxes
        # post nms for all the proposals for each image
        # proposal shape: N * Num_anchor_all * 4
        # pred_class shape: N * Num_anchor_all * C


        pred_class = torch.sigmoid_(pred_class)

        class_num = pred_class.shape[-1]

        batch_size, num_anchors_all, C = pred_class.shape

        boxes_all = []
        scores_all = []
        labels_all = []
        for i in range(batch_size):
            pred_class_image = pred_class[i]
            pred_bbox_image = pred_bbox[i]
            image_shape = image_shapes[i]

            # crop the box inside the image
            if self.post_clip:
                pred_bbox_image = self.crop_boxes(pred_bbox_image, image_shape)

            boxes_image = []
            scores_image = []
            labels_image = []
            for class_ind in range(class_num):
                pred_class_image_the_class= pred_class_image[:,class_ind]

                # remove detection with low score
                keep = pred_class_image_the_class > self.score_thresh
                scores_class = pred_class_image_the_class[keep]
                boxes_class = pred_bbox_image[keep]

                # remove too small detections
                keep = self.remove_small_boxes(boxes_class, min_size=1e-2)
                boxes_class = boxes_class[keep]
                scores_class = scores_class[keep]

                # doing non maximum surpress
                keep = nms(boxes_class, scores_class, self.nms_thresh)

                # get topk score boxes for the level
                keep = keep[:self.get_post_nms_top_n()]
                boxes_class = boxes_class[keep]
                scores_class = scores_class[keep]
                labels_class = torch.full_like(scores_class, class_ind, dtype=int)

                boxes_image.append(boxes_class)
                scores_image.append(scores_class)
                labels_image.append(labels_class)
            
            boxes_image = torch.cat(boxes_image,dim=0)
            scores_image = torch.cat(scores_image,dim=0)
            labels_image = torch.cat(labels_image,dim=0)

            
            boxes_all.append(boxes_image)
            scores_all.append(scores_image)
            labels_all.append(labels_image+1)

        results = dict()
        results['boxes'] = boxes_all
        results['scores'] = scores_all
        results['labels'] = labels_all
        return results


    def remove_small_boxes(self, boxes, min_size):
        #area = box_area(boxes)
        w = boxes[:,2]-boxes[:,0]
        h = boxes[:,3]-boxes[:,1]
        keep = torch.minimum(w,h) >= min_size
        #keep = area >= min_size
        return keep

    def crop_boxes(self, boxes, image_size):
        # boxes: N * 4 tensor, x1, y1, x2, y2 format
        # image_size: height, width
        height, width = image_size
        boxes[...,0] = boxes[...,0].clamp(min=0, max=width)
        boxes[...,1] = boxes[...,1].clamp(min=0, max=height)
        boxes[...,2] = boxes[...,2].clamp(min=0, max=width)
        boxes[...,3] = boxes[...,3].clamp(min=0, max=height)
        return boxes

    #def get_pre_nms_top_n(self):
    #    if self.training:
    #        return self.pre_nms_top_n_train
    #    else:
    #        return self.pre_nms_top_n_test

    def get_post_nms_top_n(self):
        if self.training:
            return self.post_nms_top_n_train
        else:
            return self.post_nms_top_n_test

    def get_before_nms_top_n(self):
        if self.training:
            return self.before_nms_top_n_train
        else:
            return self.before_nms_top_n_test

    @torch.no_grad()
    def assign_targets_to_anchors(self, anchors, targets):
        # return the matched result
        match_all = []
        for anchor_image, target in zip(anchors, targets):
            boxes = target['boxes']
            labels = target['labels']
            if len(boxes) == 0:
                raise ValueError('there should be more than one item in each image')
            else:
                match = self.box_matcher.match(boxes, anchor_image, labels)
                match_all.append(match)
        return match_all
    

    def assign_targets_to_anchors_old(self, anchors, targets):
        # return the indexes of the matched anchors and matched boxes
        ind_pos_anchor_all = []
        ind_neg_anchor_all = []
        ind_pos_boxes_all = []
        for anchor_image, target in zip(anchors, targets):
            boxes = target['boxes']
            if len(boxes) == 0:
                raise ValueError('there should be more than one item in each image')
            else:
                match = self.box_matcher.match(boxes, anchor_image)
                match_box_ind = match.matched_ind
                # try to release memory
                #del match
                pos_ind = match_box_ind>= 0
                ind_pos_boxes = match_box_ind[pos_ind]
                ind_pos_anchor = torch.where(pos_ind)[0]
                ind_neg_anchor = torch.where(match_box_ind==-2)[0]
                ## set up the max iou for each box as positive 
                ## set up the iou bigger than a value as positive
                #iou_mat = box_iou(anchor_image, boxes) # anchor N and target boxes M, iou mat: NxM
                #ind_pos_anchor, ind_pos_boxes, ind_neg_anchor = self.match_boxes(iou_mat, low_thresh=0.4, high_thresh=0.5, allow_weak_match=True)
                ind_pos_anchor_all.append(ind_pos_anchor)
                ind_neg_anchor_all.append(ind_neg_anchor)
                ind_pos_boxes_all.append(ind_pos_boxes)
        return ind_pos_anchor_all, ind_neg_anchor_all, ind_pos_boxes_all

    def match_boxes(self, iou_mat, low_thresh, high_thresh, allow_weak_match=True):
        # according to faster RCNN paper, the max iou and the one bigger than 
        # high_thresh are positive matches, lower than low_thresh are negtive matches
        # the iou in between are ignored during the training
        # iou mat NxM, N anchor and M target boxes
        #N, M = iou_mat.shape
        # if one groundtruth box cannot be matched by the thresh, we allow weak match by set allow_weak_match=True

        # set up the possible weak but biggest anchors that cover boxes
        # There are possible more than one anchors have same biggest 
        # value with the boxes  

        # The box ind for each anchor
        match_box_ind = torch.full_like(iou_mat[:,0], -1, dtype=torch.int64)

        # set the negtive index, the box ind will be overwrite later by the weak match if it is allowed
        max_val_anchor, max_box_ind = iou_mat.max(dim=1)
        index_neg_anchor = torch.where(max_val_anchor<low_thresh)
        match_box_ind[index_neg_anchor] = -2 # -2 means negtive anchors


        ##TODO stuff added, maybe need to delete these
        #max_val, max_box_ind = iou_mat.max(dim=1)
        #match_box_ind[inds_anchor] = max_box_ind[inds_anchor]
        

        # set the vague ones
        # if a anchor can match two boxes, still keep the biggest one!!!!
        if allow_weak_match:
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

    def combine_and_permute_predictions(self, predictions, keep_multi_level=False):
        pred_class_all = []
        pred_bbox_deltas_all = []
        # for each feature map, potentially have multiple level output (Such as FPN output), do the process
        for pred_class, pred_bbox_deltas in predictions:
            # the shape of original pred_class: N * Ax2 * H * W
            # the shape of original pred_bbox_deltas: N * Ax4 * H * W
            # target shape should be same with anchor N * HxW * A * 4 
            N, AxC, H, W = pred_class.shape
            N, Ax4, H, W = pred_bbox_deltas.shape
            A = Ax4 // 4
            C = AxC // A

            # keep the batch and last C, the output should be N * HxWxA * C
            pred_class = permute_and_flatten(pred_class, N, C, A, H, W)
            pred_bbox_deltas = permute_and_flatten(pred_bbox_deltas, N, 4, A, H, W)

            pred_class_all.append(pred_class)
            pred_bbox_deltas_all.append(pred_bbox_deltas)
        if keep_multi_level:
            return pred_class_all, pred_bbox_deltas_all
        else:
            pred_class = torch.cat(pred_class_all, dim=1)
            pred_bbox_deltas = torch.cat(pred_bbox_deltas_all, dim=1)
            return pred_class, pred_bbox_deltas



            
def permute_and_flatten(pred, N, C, A, H, W):
    pred = pred.view(N, A, C, H, W)
    pred = pred.permute(0, 3, 4, 1, 2)
    pred = pred.reshape(N, -1, C)
    return pred