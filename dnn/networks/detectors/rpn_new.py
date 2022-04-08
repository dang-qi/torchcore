import torch
import math
import traceback
import sys
from torch import nn

#from torchvision.models.detection.rpn import RegionProposalNetwork

from torchvision.ops import nms
from torchvision.ops.boxes import batched_nms, box_iou
from torch.nn import functional as F

from ..tools.anchor import build_anchor_generator
from ..tools.sampler import build_sampler
from ..tools.box_coder import build_box_coder
from ..tools.box_matcher import build_box_matcher
from ..losses import build_loss
from ..heads import build_head

from .build import DETECTOR_REG
#from .tools import AnchorBoxesCoder
#from .tools import PosNegSampler

@DETECTOR_REG.register(force=True)
class RPNNew(nn.Module):
    def __init__(self, anchor_generator, rpn_head, bbox_coder, sampler, box_matcher, box_loss=None, class_loss=None, nms_thresh=0.7, dataset_label=None, pre_nms_top_n_train=2000, pre_nms_top_n_test=2000, post_nms_top_n_train=1000, post_nms_top_n_test=100):
        super().__init__()
        self.anchor_generator = build_anchor_generator(anchor_generator)
        rpn_head.num_anchors = self.anchor_generator.num_anchors_per_location()[0]
        self.head = build_head(rpn_head)
        self.box_coder = build_box_coder(bbox_coder)
        self.box_matcher= build_box_matcher(box_matcher)
        #self.pos_neg_sampler = PosNegSampler(pos_num=128, neg_num=128)
        self.sampler = build_sampler(sampler)
        #self.pos_neg_sampler = PosNegSampler(pos_num=256, neg_num=256)
        #self.tv_sampler = BalancedPositiveNegativeSampler(512, 0.25)

        self.nms_thresh = nms_thresh
        self.dataset_label = dataset_label
        self._pre_nms_top_n_train = pre_nms_top_n_train
        self._pre_nms_top_n_test = pre_nms_top_n_test
        self._post_nms_top_n_train = post_nms_top_n_train
        self._post_nms_top_n_test = post_nms_top_n_test

        # used to remove the boxes not in the image
        self.min_size = 1e-3
        if box_loss is None:
            self.box_loss = nn.SmoothL1Loss(reduction='sum', beta= 1.0 / 9) 
        else:
            self.box_loss = build_loss(box_loss)
        if class_loss is None:
            self.class_loss = nn.BCEWithLogitsLoss()
        else:
            self.class_loss = build_loss(class_loss)
        #self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean') 

    def forward(self, inputs, features, targets):
        """
        Arguments:
            inputs (dict with 'data', 'image_sizes'): images for which we want to compute the predictions
            features (List[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        if isinstance(features, dict):
            features = list(features.values())
        elif torch.is_tensor(features):
            features = [features]
        pred_out = self.head(features) # return list[(pred_obj, pred_bbox_deltas),...for each feature maps]
        anchors = self.anchor_generator(inputs, features) # [anchors_layer_one_all(shape:(anchor_num_all,4)), anchors_layer_two_all, ...],
        #for anchor in anchors:
        #    print('anchor shape,', anchor.shape)
        #for preo, preb in pred_out:
        #    print('pre_cla shape: {}, pre_box shape: {}'.format(preo.shape, preb.shape))

        num_images = len(anchors)
        num_anchors_per_level = [pred[0][0].numel() for pred in pred_out]
        pred_class, pred_bbox_deltas = self.combine_and_permute_predictions(pred_out)
        #print('pred box shape:', pred_bbox_deltas.shape)

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, pred_class.detach(), inputs['image_sizes'], num_anchors_per_level)

        if not self.training:
            #return anchors, boxes, scores
            return boxes, scores
        else:
            losses = {}
            if self.dataset_label is not None:
                dataset_ind = self.get_dataset_ind(inputs)
                anchors = [anchor for anchor,label in zip(anchors, dataset_ind) if label]
                targets =[target for target,label in zip(targets, dataset_ind) if label ]
                pred_bbox_deltas = pred_bbox_deltas[dataset_ind]
                pred_class = pred_class[dataset_ind]

            
            #print('regression targets shapes',[t.shape for t in regression_targets])
            #print('regression targets:', regression_targets)

            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                anchors, targets, pred_class, pred_bbox_deltas)
            #return loss_objectness, loss_rpn_box_reg
            #print('loss')
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
            return boxes, losses

    def get_dataset_ind(self, inputs):
        return inputs['dataset_label'] == self.dataset_label

    def compute_loss(self, anchors, targets, pred_class, pred_bbox_deltas):
        sample_images = self.assign_targets_to_anchors(anchors, targets)
        #return sample_images[0], sample_images[1]
        pos_anchor = [sample.pos_boxes for sample in sample_images]
        pos_boxes = [sample.pos_gt_boxes for sample in sample_images]
        regression_targets  = self.box_coder.encode(pos_anchor, pos_boxes )
        regression_targets = torch.cat(regression_targets, dim=0)
        pred_class_pos = [pred[sample.pos_ind] for pred,sample in zip(pred_class,sample_images)]
        pred_class_pos = torch.cat(pred_class_pos, dim=0).flatten()
        pred_class_neg = [pred[sample.neg_ind] for pred,sample in zip(pred_class,sample_images)]
        pred_class_neg = torch.cat(pred_class_neg, dim=0).flatten()
        pred_bbox_deltas = [pred[sample.pos_ind] for pred,sample in zip(pred_bbox_deltas,sample_images)]
        pred_bbox_deltas = torch.cat(pred_bbox_deltas, dim=0)


        label_pos = torch.ones_like(pred_class_pos)
        label_neg = torch.zeros_like(pred_class_neg)
        label_pred = torch.cat((pred_class_pos, pred_class_neg), dim=0)
        label_target = torch.cat((label_pos, label_neg), dim=0)

        #loss_class = F.binary_cross_entropy_with_logits(label_pred, label_target)
        loss_class = self.class_loss(label_pred, label_target)
        # TODO this is the loss from torchvision, reduced the wight of box loss
        loss_box = self.box_loss(pred_bbox_deltas, regression_targets) / label_target.numel()
        #loss_box = self.smooth_l1_loss(pred_bbox_deltas, regression_targets) 
        #loss_box = det_utils.smooth_l1_loss(
        #    pred_bbox_deltas,
        #    regression_targets,
        #    beta=1 / 9,
        #    size_average=False,
        #) / (ind_pos_anchor.numel())
        return loss_class, loss_box


    def assign_targets_to_anchors(self, anchors, targets):
        # return the indexes of the matched anchors and matched boxes
        sample_all = []
        for anchor_image, target in zip(anchors, targets):
            boxes = target['boxes']
            labels = target['labels']
            match_result = self.box_matcher.match(boxes, anchor_image, gt_labels=labels)
            sample_result = self.sampler.sample(match_result, boxes, anchor_image, labels)
            sample_all.append(sample_result)
        return sample_all

    def filter_proposals(self, proposals, pred_class, image_sizes, num_anchors_per_level ):
        # pre nms for each feature layers in each image
        # crop the boxes so they are inside the image
        # delete very small boxes
        # post nms for all the proposals for each image
        # proposal shape: N * Num_anchor_all * 4
        # pred_class shape: N * Num_anchor_all * C
        device = proposals.device
        batch_size, num_anchors_all, C = pred_class.shape
        indexes = [torch.full((n,), num, dtype=torch.int64, device=device) for num, n in enumerate(num_anchors_per_level)]
        indexes = torch.cat(indexes, dim=0)
        indexes = indexes.expand((batch_size, num_anchors_all))
        #print('indexes shape:', indexes.shape)
        #print('num anchors per level', num_anchors_per_level)

        # do a pre top k score proposal selection for each feature map
        keep = self.get_top_k_ind(pred_class[:,:,0], self.get_pre_nms_top_n(), num_anchors_per_level)
        #print('keep shape', keep.shape)
        #print('proposals shape', proposals.shape)
        image_ind = torch.arange(batch_size, device=device)[:,None]
        proposals = proposals[image_ind, keep]
        indexes = indexes[image_ind, keep]
        pred_class = pred_class[image_ind, keep]
        #proposals = proposals.view(-1, 4)[keep].reshape(batch_size, -1, 4)
        #indexes = indexes.reshape(-1)[keep].reshape(batch_size, -1)
        #print('indexes shape:', indexes.shape)
        #pred_class = pred_class.view(-1, C)[keep].reshape(batch_size, -1, C)

        # perform nms on each layer seperately on each image
        boxes_all = []
        scores_all = []
        for proposal, pred_class_image, image_size, idxs in zip(proposals, pred_class, image_sizes, indexes):
            # first crop the boxes inside the image
            proposal = self.crop_boxes(proposal, image_size)
            if len(proposal) == 0:
                print('proposal are 0')
            #print('proposal shape:', len(proposal))
            keep = self.remove_small_boxes(proposal, self.min_size)

            scores = pred_class_image[:,0]
            proposal = proposal[keep]
            scores = scores[keep]
            idxs = idxs[keep]
            #print('indx shape', idxs.shape)
            if len(proposal) == 0:
                print('proposal after small box removal are 0')
                traceback.print_stack(file=sys.stdout)

            # perform nms
            keep = batched_nms(proposal, scores, idxs, self.nms_thresh) 
            # only keep the top k candidate
            keep = keep[:self.get_post_nms_top_n()]
            proposal = proposal[keep]
            scores = scores[keep]
            #print('proposal shape', proposal.shape)
            #print('scores shape', scores.shape)
            
            boxes_all.append(proposal)
            scores_all.append(scores)
        return boxes_all, scores_all

    def get_pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n_train
        else:
            return self._pre_nms_top_n_test

    def get_post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n_train
        else:
            return self._post_nms_top_n_test

    def get_top_k_ind(self, scores, k, num_anchors_per_level):
        # scores: Tensor[N,NumAnchors]
        keep = []
        #for score_image in scores:
        offset_image = 0
        for num_anchor in num_anchors_per_level:
            score_level = scores[:,offset_image:offset_image+num_anchor]
            k = min(k, num_anchor)
            _, i = torch.topk(score_level, k, dim=1)
            keep.append(i+offset_image)
            offset_image += num_anchor
        keep = torch.cat(keep, dim=1)
        return keep

    def remove_small_boxes(self, boxes, min_size):
        boxes_width = boxes[:,2] - boxes[:,0]
        boxes_height = boxes[:,3] - boxes[:,1]
        keep = (boxes_width >= min_size) & (boxes_height >= min_size)
        keep = torch.where(keep)[0]
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

