import torch
import random
from torch import nn
from .roi_align_batch import ROIAlignBatch
from torchvision.ops.boxes import box_area

class TargetInRoIPool(nn.Module):
    '''
        This class is to select and add the human rois and then do the roi align
        cfg.min_area: The minimum area requirement for the human proposals
        cfg.roi_pool_h/cfg.roi_pool_w: The roi pool size
    '''
    def __init__(self, cfg):
        super(TargetInRoIPool, self).__init__()
        self.roi_align = ROIAlignBatch(cfg.roi_pool_h,
                                      cfg.roi_pool_w,
                                      sampling=-1)
        self.dataset_label = cfg.dataset_label
        self.min_area = cfg.min_area
        self.train_box_iou_thre = cfg.train_box_iou_thre

    def forward(self, proposals, features, stride, inputs=None, targets=None):
        '''
            features: features from one output layer: NxCxHxW
            proposals: list[N1x4, N2x4...]
            stride: int or float
        '''
        image_sizes = inputs['image_sizes']
        if self.training and self.dataset_label is not None:
            dataset_ind = inputs['dataset_label'] == self.dataset_label
            features = features[dataset_ind]
            image_sizes =[im_size for im_size,label in zip(image_sizes, dataset_ind) if label ]
            if targets is not None:
                targets =[target for target,label in zip(targets, dataset_ind) if label ]

        if self.training or proposals is None:
            # remove the proposals from other branch if there are any
            if self.dataset_label is not None and proposals is not None:
                non_human_ind = inputs['dataset_label'] != 0
                other_labels = inputs['dataset_label'][non_human_ind]
                proposal_ind = other_labels == self.dataset_label
                assert len(proposal_ind) == len(proposals)
                proposals = [prop for prop, label in zip(proposals, proposal_ind) if label]

            assert len(proposals) == len(targets)

            proposals = self.select_proposals(proposals, targets, image_sizes)
        
        # rois: tuple(Tensor_image1, Tensor_image2)
        rois = self.roi_align(features, proposals, stride)
        roi_features = torch.cat(rois, dim=0)
        return proposals, roi_features, targets

    @torch.no_grad()
    def select_proposals(self, proposals, targets, image_sizes):
        if proposals is not None:
            # remove small human proposals
            proposals = self.remove_small_boxes_batch(proposals, min_area=self.min_area)

            # add gt human box
            proposals = self.add_gt_boxes( proposals, targets)
        else:
            proposals = self.gen_gt_boxes(targets)

        ## TODO this is for debug and need to be deleted!!!
        #image_sizes = [(h,w-1) for h,w in image_sizes]

        # make proposals inside the image
        proposals = [self.crop_boxes(proposal, image_size) for proposal, image_size in zip(proposals, image_sizes)]

        # select the human box that contain at least several targets
        proposals = self.get_valid_proposal(proposals, targets)
        #print('valid proposals', proposals)

        return proposals

    def get_valid_proposal(self, proposals, targets):
        proposal_out = []
        for proposal_image, target in zip(proposals, targets):
            boxes = target['boxes']
            iou_mat = box_cover(proposal_image, boxes)
            iou_mat = iou_mat > self.train_box_iou_thre
            keep = torch.sum(iou_mat, 1, dtype=torch.float64) / len(boxes) > 0.5
            proposal_out.append(proposal_image[keep])
        return proposal_out
                

    def add_gt_boxes(self, proposals, targets):
        proposal_out = []
        for proposal_image, target in zip(proposals, targets):
            if 'human_box' in target:
                human_box = target['human_box']
            else:
                human_box = gen_human_box(target['boxes'])
            proposal_image= torch.cat((proposal_image, human_box), dim=0)
            proposal_out.append(proposal_image)
        return proposal_out

    def gen_gt_boxes(self, targets):
        proposal_out = []
        for target in targets:
            if 'human_box' in target:
                human_box = target['human_box']
            else:
                human_box = gen_human_box(target['boxes'])
            proposal_out.append(human_box)
        return proposal_out

    def crop_boxes(self, boxes, image_size):
        # boxes: N * 4 tensor, x1, y1, x2, y2 format
        # image_size: height, width
        height, width = image_size
        boxes[...,0] = boxes[...,0].clamp(min=0, max=width)
        boxes[...,1] = boxes[...,1].clamp(min=0, max=height)
        boxes[...,2] = boxes[...,2].clamp(min=0, max=width)
        boxes[...,3] = boxes[...,3].clamp(min=0, max=height)
        return boxes

    def remove_small_boxes(self, boxes, min_area=0.1):
        area = (boxes[:,3]-boxes[:,1]) * (boxes[:,2]-boxes[:,0])
        keep = torch.where(area > min_area)[0]
        return keep

    def remove_small_boxes_batch(self, batch_boxes, min_area):
        out_boxes = []
        for boxes in batch_boxes:
            keep = self.remove_small_boxes(boxes, min_area=min_area)
            out_boxes.append(boxes[keep])
        return out_boxes

def box_cover(boxes, covered_boxes):
    #area1 = box_area(boxes)
    area2 = box_area(covered_boxes)

    lt = torch.max(boxes[:, None, :2], covered_boxes[:, :2])  # [N,M,2]
    rb = torch.min(boxes[:, None, 2:], covered_boxes[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    covered_ratio = inter /  area2 
    return covered_ratio

def gen_human_box( boxes, x_margin_min=5, x_margin_max=20, y_margin_min=20, y_margin_max=40):
    x_margin = random.randint(x_margin_min, x_margin_max)
    y_margin = random.randint(y_margin_min, y_margin_max)
    x1 = min(boxes[:,0]) - x_margin
    y1 = min(boxes[:,1]) - y_margin
    x2 = max(boxes[:,2]) + x_margin
    y2 = max(boxes[:,3]) + y_margin
    human_box = torch.tensor([[x1,y1,x2,y2]], dtype=boxes.dtype, device=boxes.device)
    return human_box