import torch
from torch.nn import functional as F
#from torchvision.ops.boxes import box_iou
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps as box_iou
from .match_result import MatchResult
from .build import BOX_MATCHER_REG

@BOX_MATCHER_REG.register(force=True)
class SimOTABoxMatcher():
    def __init__(self,
                 center_radius=2.5,
                 candidate_topk=10,
                 iou_weight=3.,
                 class_weight=1.) -> None:
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.class_weight = class_weight

    def match(self, 
              pred_scores, 
              priors_with_strides, 
              pred_bbox, 
              gt_boxes, 
              gt_labels,
              pred_obj=None,
              eps=1e-7):
        '''
            Suppose we have M gt_boxes and N pred_bboxes, C is class_num
            pred_scores: NxC
            priors: Nx4, [x,y,stride_w,stride_h]
            pred_bbox: Nx4
            gt_boxes: Mx4
            gt_labels: M

            warning: here gt labels start with 1!!!
        '''
        INF=100000000
        gt_nums = gt_boxes.size(0)
        pred_nums = pred_bbox.size(0)
        num_classes = pred_scores.size(1)
        matched_ind = pred_bbox.new_zeros((pred_nums,),dtype=torch.long)
        # select the prior grids that is inside or in the center of gt_box
        center_ind, in_box_and_in_center =self.select_center_candidates(priors_with_strides, gt_boxes)
        valid_num = center_ind.sum()

        if not center_ind.any():
            max_iou = pred_bbox.new_zeros((pred_nums,))
            if gt_labels is None:
                labels = None
            else:
                labels = gt_labels.new_full((pred_nums,),-1)
            return MatchResult(gt_nums, matched_ind, max_iou, labels)

        # calculate Reg cost
        pred_bbox_valid = pred_bbox[center_ind]
        iou_mat = box_iou(pred_bbox_valid, gt_boxes) # VxM
        iou_cost = -torch.log(iou_mat+eps) # VxM
        #return gt_boxes, pred_bbox_valid

        gt_labels_one_hot = F.one_hot((gt_labels-1).to(torch.int64),num_classes=num_classes) # MxC
        gt_labels_one_hot = gt_labels_one_hot.float().unsqueeze(0).repeat(valid_num,1,1)

        # calculate cls cost
        with torch.cuda.amp.autocast(enabled=False):
            if pred_obj is not None:
                pred_obj = pred_obj[center_ind].float().sigmoid().unsqueeze(-1)
                pred_scores_valid = pred_scores[center_ind].float().sigmoid()*pred_obj
            else:
                pred_scores_valid = pred_scores[center_ind].float().sigmoid()
            pred_scores_valid = pred_scores_valid.unsqueeze(1).repeat(1,gt_nums,1) # Vx1xC

            cls_cost = F.binary_cross_entropy(pred_scores_valid.sqrt_(), gt_labels_one_hot, reduction='none').sum(-1)

        # generate cost matrix
        # VxM
        cost_mat = iou_cost*self.iou_weight + cls_cost*self.class_weight + (~in_box_and_in_center*INF) # we just want to match the one that both inside the box and in the center???

        # get dynamic top k candidates
        k_all = self.dynamic_topk(iou_mat)

        # do the match and remove the grid that assigned to multiple gt_boxes
        matched_mask, matched_ind = self.match_topk(cost_mat, k_all)

        matched_ious = iou_mat[matched_mask,matched_ind]

        matched_ind_all = pred_bbox.new_full((pred_nums,),MatchResult.NEGATIVE_MATCH, dtype=torch.long)
        #matched_ious_all = pred_bbox.new_zeros((pred_nums,))
        matched_ious_all = pred_bbox.new_full((pred_nums,),-INF)

        # now center ind become positive mask
        center_ind[center_ind.clone()] = matched_mask
        pos_ind = center_ind
        matched_ind_all[pos_ind] = matched_ind
        matched_ious_all[pos_ind]=matched_ious
        matched_labels = matched_ind.new_full((pred_nums,),-1)
        matched_labels[pos_ind] = gt_labels[matched_ind]

        return MatchResult(gt_nums, matched_ind=matched_ind_all, max_iou=matched_ious_all, labels=matched_labels)

    def dynamic_topk(self, pair_wise_ious):
        # first get the k
        candidate_k = min(self.candidate_topk, pair_wise_ious.size(0))
        topk_ious, _ = torch.topk(pair_wise_ious, candidate_k, dim=0)
        k_all = topk_ious.sum(0).int().clamp(1)
        return k_all

    def match_topk(self, cost, k_all):
        # for each gt, we find the index of positive and negative samples by cost
        match_matrix = torch.zeros_like(cost)
        for i,k in enumerate(k_all):
            _, ind = torch.topk(cost[:,i],k.item(),largest=False) 
            match_matrix[:,i][ind] = 1

        # what happens when one prediction match more than one gt?
        # we just assign this anchor to the gt that have smallest cost
        multi_match_ind = match_matrix.sum(dim=1)>1
        if multi_match_ind.any():
            min_cost, min_ind = torch.min(cost[multi_match_ind,:],dim=1)
            match_matrix[multi_match_ind,:] *= 0
            match_matrix[multi_match_ind,min_ind] = 1

        # get the valid ind with match matrix
        matched_mask = match_matrix.any(dim=1)
        matched_box_ind = match_matrix[matched_mask].argmax(1)
        return matched_mask, matched_box_ind





    def select_center_candidates(self, prior_with_strides, gt_boxes):
        # N prior, M gt boxes
        # the prior center should be in gt_boxes
        # shape Nx1, can be broadcast to NxM
        prior_x = prior_with_strides[:,0].unsqueeze(1) 
        prior_y = prior_with_strides[:,1].unsqueeze(1) 
        stride_w = prior_with_strides[:,2].unsqueeze(1)
        stride_h = prior_with_strides[:,3].unsqueeze(1)

        # shape NxM
        x1_diff = prior_x - gt_boxes[:,0] 
        y1_diff = prior_y - gt_boxes[:,1]
        x2_diff = gt_boxes[:,2] - prior_x
        y2_diff = gt_boxes[:,3] - prior_y

        # shape NxM
        inside_box_all = torch.stack([x1_diff,y1_diff,x2_diff,y2_diff], dim=-1).min(dim=-1).values >0 
        inside_box = inside_box_all.any(dim=1) # N

        # the prior center should be in gt_boxes center
        # shape NxM
        gt_cx = (gt_boxes[:,2]+gt_boxes[:,0])/2
        gt_cy = (gt_boxes[:,3]+gt_boxes[:,1])/2

        gt_center_x1 = gt_cx - self.center_radius*stride_w 
        gt_center_y1 = gt_cy - self.center_radius*stride_h 
        gt_center_x2 = gt_cx + self.center_radius*stride_w
        gt_center_y2 = gt_cy + self.center_radius*stride_h 
        x1_diff = prior_x - gt_center_x1
        y1_diff = prior_y - gt_center_y1
        x2_diff = gt_center_x2 - prior_x
        y2_diff = gt_center_y2 - prior_y

        # shape NxM
        inside_center_box_all = torch.stack([x1_diff,y1_diff,x2_diff,y2_diff], dim=-1).min(dim=-1).values >0 
        inside_center_box = inside_center_box_all.any(dim=1) # N

        # grid bool index for grid inside gt box or center box
        inside_box_or_center_box = inside_box | inside_center_box # N

        # grid/gt bool index matrix for grid inside gt box and center box
        inside_box_and_center_box = inside_box_all[inside_box_or_center_box,:] & inside_center_box_all[inside_box_or_center_box,:] # NxM

        return inside_box_or_center_box, inside_box_and_center_box

