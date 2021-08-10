import torch
import math
from torch import nn
#from ..heads.retina_head import RetinaHead
from ..tools import AnchorBoxesCoder
#from ..rpn import MyRegionProposalNetwork, RegionProposalNetwork
from torchvision.ops.boxes import batched_nms, box_iou
from ..losses import FocalLossSigmoid,FocalLoss,SigmoidFocalLoss

class RetinaNetHead(nn.Module):
    def __init__(self, head, anchor_generator, cfg):
        super(RetinaNetHead, self).__init__()
        self.cfg = cfg
        self.head = head
        self.anchor_generater = anchor_generator
        self.box_coder = AnchorBoxesCoder(box_code_clip=math.log(1000./16))
        self.nms_thresh = cfg.nms_thresh
        self.score_thresh = cfg.score_thresh
        #self.class_loss = FocalLossSigmoid(alpha=0.25, gamma=2)
        self.class_loss = SigmoidFocalLoss(alpha=0.25, gamma=2, reduction='sum')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum', beta= 1.0 / 9) 

    def forward(self, inputs, features, targets=None):
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

    def compute_loss(self, pred_class, pred_bbox_deltas, ind_pos_anchor, ind_neg_anchor, regression_targets, pos_labels):

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
            loss_class_list.append(self.class_loss(label_pred, label_target) / label_pos.size()[0])
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
            loss_box_list.append(self.smooth_l1_loss(pred_bbox_delta, regression_targets_per_im) / pred_bbox_delta.shape[0])
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

    def post_detection(self, pred_class, pred_bbox, image_shapes, num_anchors_per_level):
        # pre nms for each feature layers in each image
        # crop the boxes so they are inside the image
        # delete very small boxes
        # post nms for all the proposals for each image
        # proposal shape: N * Num_anchor_all * 4
        # pred_class shape: N * Num_anchor_all * C


        pred_class = torch.sigmoid_(pred_class)

        batch_size, num_anchors_all, C = pred_class.shape

        boxes_all = []
        scores_all = []
        labels_all = []
        for i in range(batch_size):
            pred_class_image = pred_class[i]
            pred_bbox_image = pred_bbox[i]
            image_shape = image_shapes[i]

            boxes_image = []
            scores_image = []
            labels_image = []
            off_set_image = 0
            for num_anchor in num_anchors_per_level:
                pred_class_image_level = pred_class_image[off_set_image:off_set_image+num_anchor]
                pred_bbox_image_level = pred_bbox_image[off_set_image:off_set_image+num_anchor]
                # remove detection with low score
                #print('pred_class level ', pred_class_image_level)
                keep = pred_class_image_level > self.score_thresh
                score_keep = pred_class_image_level[keep]
                inds, inds_class = torch.where(keep)

                # get topk score boxes for the level
                num_topk = min(self.get_pre_nms_top_n(), len(inds))
                scores_topk, ind_topk = score_keep.topk(num_topk)
                inds = inds[ind_topk]
                inds_class = inds_class[ind_topk]
                boxes_topk = pred_bbox_image_level[inds]

                self.crop_boxes(boxes_topk, image_shape)

                boxes_image.append(boxes_topk)
                scores_image.append(scores_topk)
                labels_image.append(inds_class)

                off_set_image += num_anchor
            
            boxes_image = torch.cat(boxes_image,dim=0)
            scores_image = torch.cat(scores_image,dim=0)
            labels_image = torch.cat(labels_image,dim=0)

            # do nms for the image
            keep = batched_nms(boxes_image, scores_image, labels_image, self.nms_thresh) 
            # only keep the top k candidate
            keep = keep[:self.get_post_nms_top_n()]
            boxes_image = boxes_image[keep]
            scores_image = scores_image[keep]
            labels_image = labels_image[keep]
            #print('scores shape', scores.shape)
            
            boxes_all.append(boxes_image)
            scores_all.append(scores_image)
            labels_all.append(labels_image+1)

        results = dict()
        results['boxes'] = boxes_all
        results['scores'] = scores_all
        results['labels'] = labels_all
        return results


    def crop_boxes(self, boxes, image_size):
        # boxes: N * 4 tensor, x1, y1, x2, y2 format
        # image_size: height, width
        height, width = image_size
        boxes[...,0] = boxes[...,0].clamp(min=0, max=width)
        boxes[...,1] = boxes[...,1].clamp(min=0, max=height)
        boxes[...,2] = boxes[...,2].clamp(min=0, max=width)
        boxes[...,3] = boxes[...,3].clamp(min=0, max=height)
        return boxes

    def get_pre_nms_top_n(self):
        if self.training:
            return self.cfg.pre_nms_top_n_train
        else:
            return self.cfg.pre_nms_top_n_test

    def get_post_nms_top_n(self):
        if self.training:
            return self.cfg.post_nms_top_n_train
        else:
            return self.cfg.post_nms_top_n_test

    def assign_targets_to_anchors(self, anchors, targets):
        # return the indexes of the matched anchors and matched boxes
        ind_pos_anchor_all = []
        ind_neg_anchor_all = []
        ind_pos_boxes_all = []
        for anchor_image, target in zip(anchors, targets):
            boxes = target['boxes']
            if len(boxes) == 0:
                raise ValueError('there should be more than one item in each image')
            else:
                iou_mat = box_iou(anchor_image, boxes) # anchor N and target boxes M, iou mat: NxM
                # set up the max iou for each box as positive 
                # set up the iou bigger than a value as positive
                ind_pos_anchor, ind_pos_boxes, ind_neg_anchor = self.match_boxes(iou_mat, low_thresh=0.4, high_thresh=0.5, allow_weak_match=True)
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

    def combine_and_permute_predictions(self, predictions):
        pred_class_all = []
        pred_bbox_deltas_all = []
        # for each feature map, potentially have multiple batch, do the process
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
        pred_class = torch.cat(pred_class_all, dim=1)
        pred_bbox_deltas = torch.cat(pred_bbox_deltas_all, dim=1)
        return pred_class, pred_bbox_deltas



            
def permute_and_flatten(pred, N, C, A, H, W):
    pred = pred.view(N, A, C, H, W)
    pred = pred.permute(0, 3, 4, 1, 2)
    pred = pred.reshape(N, -1, C)
    return pred