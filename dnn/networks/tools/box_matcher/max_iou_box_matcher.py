import torch
from torchvision.ops.boxes import box_iou
from .match_result import MatchResult
from .build import BOX_MATCHER_REG


@BOX_MATCHER_REG.register(force=True)
class MaxIoUBoxMatcher():
    def __init__(self, high_thresh:float, low_thresh:float, allow_low_quality_match=True, assign_all_gt_max=True, keep_max_iou_in_low_quality=True) -> None:
        '''assign_all_gt_max: only works when allow_low_quality_match. If there are more than one anchor box is assigned to max value, should we assign all the matching to gt box or just match one to the gt box'''
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.allow_low_quality_match = allow_low_quality_match
        self.assign_all_gt_max = assign_all_gt_max
        self.keep_max_iou_in_low_quality = keep_max_iou_in_low_quality
        self.NEGATIVE_MATCH = MatchResult.NEGATIVE_MATCH
        self.BETWEEN_POS_NEG_MATCH = MatchResult.IGNORE_MATCH

    def match(self, gt_boxes, anchor_boxes, gt_lables=None):
        
        '''match the gt_boxes to the anchor boxes

        This method can assign the gt boxes to anchor boxes,
        if the Iou(gt_box, anchor_box) >= high_thresh it should be positive, if IoU(gt_box, anchor_box) < low_thresh, it should be assigned as negative.
        if allow low quality assign, the gt_box can be assign to the biggest IoU anchor box even if the iou < low_thresh    

        match_box_ind: the array that indicate which gt_box is assigned to the anchor box, -1 is not assigned, -2 is negativve,
        '''

        # calculate IoU between gt_boxes and anchor boxes
        # iou mat NxM, N anchor and M target boxes
        # N, M = iou_mat.shape
        iou_mat = box_iou(anchor_boxes, gt_boxes)

        # The box ind for each anchor
        match_box_ind = torch.full_like(iou_mat[:,0], self.BETWEEN_POS_NEG_MATCH, dtype=torch.int64)

        anchor_num, box_num = iou_mat.shape
        if anchor_num ==0 or box_num==0:
            # if no gt box, everything are background
            if box_num == 0:
                match_box_ind = self.NEGATIVE_MATCH
                max_iou = torch.zeros_like(match_box_ind)
            return MatchResult(box_num, match_box_ind, max_iou )
        # set the negtive index, the box ind will be overwrite later by the weak match if it is allowed
        max_val_anchor, max_box_ind = iou_mat.max(dim=1)
        index_neg_anchor = torch.where(max_val_anchor<self.low_thresh)
        match_box_ind[index_neg_anchor] = self.NEGATIVE_MATCH # -2 means negtive anchors

        inds_anchor_above = max_val_anchor>=self.high_thresh
        match_box_ind[inds_anchor_above] = max_box_ind[inds_anchor_above]

        if self.allow_low_quality_match:
            max_val, max_anchor_ind = iou_mat.max(dim=0)
            # when there is two gt boxes A,B has IoU with Anchor box C 0.8 and 0.9 respectively. A want to get assigned during this low quality match, detectron still assign C to B, which has bigger IoU, while mmdetection assign C to A. It has no significant impact according to https://github.com/facebookresearch/detectron2/blob/7cad0a7d95cc8b0c7974cc19e50bded742183555/detectron2/modeling/matcher.py#L124
            if self.assign_all_gt_max:
                inds_anchor, ind_box = torch.where(iou_mat==max_val.expand_as(iou_mat))
                if self.keep_max_iou_in_low_quality:
                    match_box_ind[inds_anchor] = max_box_ind[inds_anchor] # torchvision/detectron implementation
                else:
                    match_box_ind[inds_anchor] = ind_box # mm detection implementation
            else:
                if self.keep_max_iou_in_low_quality:
                    match_box_ind[max_anchor_ind] = max_box_ind[max_anchor_ind] # detectron
                else:
                    match_box_ind[max_anchor_ind] = torch.arange(box_num) # mmdetection
        return MatchResult(box_num, match_box_ind, max_val_anchor)