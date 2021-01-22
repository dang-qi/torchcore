import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from .pooling import ROIAlignBatch
from .tools import PosNegSampler
from .tools import AnchorBoxesCoder
from torchvision.ops.boxes import batched_nms, box_iou, box_area
from torchvision.ops import roi_align
from .heads import FastRCNNHead
from .center_net import get_center_head
from ...data.datasets.coco_center import generate_ellipse_gaussian_heatmap, generate_ind, generate_offset, generate_width_height
from .center_net import CenterNetLoss, decode_by_ind, point_nms, topk_ind

def box_cover(boxes, covered_boxes):
    #area1 = box_area(boxes)
    area2 = box_area(covered_boxes)

    lt = torch.max(boxes[:, None, :2], covered_boxes[:, :2])  # [N,M,2]
    rb = torch.min(boxes[:, None, 2:], covered_boxes[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    covered_ratio = inter /  area2 
    return covered_ratio

class RoIFrcnn(nn.Module):
    def __init__(self,cfg, rpn, roi_head):
        super().__init__()
        self.cfg = cfg
        self.train_box_iou_thre = cfg.train_box_iou_thre
        self.min_area = cfg.min_area

        self.roi_align = ROIAlignBatch(cfg.roi_pool_h,
                                      cfg.roi_pool_w,
                                      sampling=-1)
        #self.faster_rcnn_head = FastRCNNHead(cfg)
        self.rpn = rpn
        self.roi_head = roi_head

        self.strides = None

        #self.box_coder = AnchorBoxesCoder(box_code_clip=None)
        #self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        #self.label_loss = nn.CrossEntropyLoss(reduction='mean')

        if hasattr(cfg, 'dataset_label'):
            self.dataset_label = cfg.dataset_label
        else:
            self.dataset_label = None


    def forward(self, proposals, features, stride, inputs=None, targets=None):
        '''
            proposals: only contain human proposals, list[Tensor[nx4]...]
            features: Tensor NxCxHxW
        '''

        image_sizes = inputs['image_sizes']
        self.stride = stride
        if self.training and self.dataset_label is not None:
            dataset_ind = inputs['dataset_label'] == self.dataset_label
            features = features[dataset_ind]
            #image_sizes = image_sizes[dataset_ind]
            image_sizes =[im_size for im_size,label in zip(image_sizes, dataset_ind) if label ]
            if targets is not None:
                targets =[target for target,label in zip(targets, dataset_ind) if label ]

        #print('feature shape',features.shape)
        #print('targets len', len(targets))

        if self.training:
            proposals = self.select_proposals(proposals, targets, image_sizes)
            new_targets = self.generate_targets(proposals, targets, stride)
            #return proposals, centernet_targets
        elif proposals is None:
            proposals = self.select_proposals(proposals, targets, image_sizes)
        #print('proposals:', proposals)
        #print('targes:', targets)
        #print('new targes:', new_targets)
        
        proposal_per_im = [len(proposal) for proposal in proposals]

        # rois: tuple(Tensor_image1, Tensor_image2)
        rois = self.roi_align(features, proposals, stride)
        roi_features = torch.cat(rois, dim=0)

        rpn_inputs = self.generate_inputs(len(roi_features))

        features_new = OrderedDict()
        features_new['0'] = roi_features
        roi_features = features_new

        #print('proposals',proposals)
        #print('rois',rois)
        #print('features',features)
        #print('rois-features',(rois-features[:,:,:,:69]).sum())
        #print('rois-features',rois-features)
        if self.training:
            second_proposals, losses_rpn = self.rpn(rpn_inputs, roi_features, new_targets)
        else:
            second_proposals, scores = self.rpn(rpn_inputs, roi_features, targets)


        #for k in self.feature_names:
        #    features_new[k] = features[k]
        #if self.strides is None:
        #    strides = self.get_strides()
        #else:
        #    strides = self.strides
        strides = [stride]
        #for proposal in proposals:
        #    print('proposal shape', proposal.shape)

        if self.training:
            losses_roi = self.roi_head(second_proposals, roi_features, strides, targets=new_targets)
            losses = {**losses_rpn, **losses_roi}
            return losses
        else:
            results = self.roi_head(second_proposals, roi_features, strides )
            results = self.post_process(results, proposals, proposal_per_im, stride)
            return results

    @torch.no_grad()
    def post_process(self, results, human_boxes_batch, human_per_im, stride):
        roi_boxes = torch.cat(human_boxes_batch, dim=0) # ROI_NUM_N x 4
        roi_w = roi_boxes[:,2] - roi_boxes[:,0] # ROI_NUM_N
        roi_h = roi_boxes[:,3] - roi_boxes[:,1]
        w_scale = (roi_w / self.cfg.roi_pool_w / stride) # ROI_NUM_N 
        h_scale = (roi_h / self.cfg.roi_pool_h / stride)
        boxes_batch = results['boxes'] # list[DET_PER_IM x 4,...]
        for i, boxes in enumerate(boxes_batch):
            boxes[:,0] *= w_scale[i]
            boxes[:,2] *= w_scale[i]
            boxes[:,1] *= h_scale[i]
            boxes[:,3] *= h_scale[i]

            boxes[:,0] += roi_boxes[i,0]
            boxes[:,2] += roi_boxes[i,0]
            boxes[:,1] += roi_boxes[i,1]
            boxes[:,3] += roi_boxes[i,1]

        #results['boxes'] = [torch.cat(boxes[im_num:im_num+i], dim=0) for i, im_num in zip(accumulate_list,human_per_im)]
        scores = []
        labels = []
        boxes = []

        ind = 0
        for human_num in human_per_im:
            boxes.append(torch.cat(results['boxes'][ind:ind+human_num], dim=0))
            labels.append(torch.cat(results['labels'][ind:ind+human_num], dim=0))
            scores.append(torch.cat(results['scores'][ind:ind+human_num], dim=0))
            ind+=human_num
        results['boxes'] = boxes
        results['labels'] = labels
        results['scores'] = scores
        return results

    @torch.no_grad()
    def post_process_old(self, results, human_boxes_batch, human_per_im, stride):
        roi_boxes = torch.cat(human_boxes_batch, dim=0) # ROI_NUM_N x 4
        roi_w = roi_boxes[:,2] - roi_boxes[:,0] # ROI_NUM_N x 1
        roi_h = roi_boxes[:,3] - roi_boxes[:,1]
        w_scale = (roi_w / self.cfg.roi_pool_w / stride)[:,None] # ROI_NUM_N x 1
        h_scale = (roi_h / self.cfg.roi_pool_h / stride)[:,None]
        boxes_batch = results['boxes'] # list[DET_PER_IM x 4,...]
        boxes_all = torch.stack(boxes_batch, dim=0) # ROI_NUM_N x DET_PER_IM x 4

        boxes_all[:,:,0] *= w_scale
        boxes_all[:,:,2] *= w_scale
        boxes_all[:,:,1] *= h_scale
        boxes_all[:,:,3] *= h_scale

        boxes_all[:,:,0] += roi_boxes[:,0][:,None]
        boxes_all[:,:,2] += roi_boxes[:,0][:,None]
        boxes_all[:,:,1] += roi_boxes[:,1][:,None]
        boxes_all[:,:,3] += roi_boxes[:,1][:,None]
        
        results['boxes'] = torch.split(boxes_all, human_per_im)
        results['boxes'] = [boxes_im.reshape(-1,4) for boxes_im in results['boxes']]
        scores = []
        labels = []
        ind = 0
        for human_num in human_per_im:
            labels.append(torch.cat(results['labels'][ind:ind+human_num], dim=0))
            scores.append(torch.cat(results['scores'][ind:ind+human_num], dim=0))
            ind+=human_num
        results['labels'] = labels
        results['scores'] = scores
        return results


    
    def generate_inputs(self, batch_size):
        '''
            return input_size: (h,w), image_sizes: list[(h,w)...] and batch_size: int
        '''
        inputs = {}
        h = self.stride * self.cfg.roi_pool_h
        w = self.stride * self.cfg.roi_pool_w

        inputs['input_size'] = (h, w)
        inputs['image_sizes'] = [(h, w)]*batch_size
        inputs['batch_size'] = batch_size
        return inputs

    @torch.no_grad()
    def generate_targets(self, human_boxes_batch, targets, stride):
        new_targets = []
        for human_boxes, target in zip(human_boxes_batch, targets):
            for human_box in human_boxes:
                boxes = target['boxes']
                boxes = self.normalize_boxes(human_box, boxes, stride)
                #boxes = boxes.detach().cpu().numpy()
                keep = self.find_valid_boxes(boxes)
                boxes  = boxes[keep]
                labels = target['labels'][keep]
                new_target = {}
                new_target['boxes'] = boxes
                new_target['labels'] = labels
                new_targets.append(new_target)
        
        return new_targets

    def get_strides(self):
        return self.cfg.feature_strides
            

    def find_valid_boxes(self, boxes):
        if torch.is_tensor(boxes):
            keep = torch.logical_and(boxes[:,2] > boxes[:,0], boxes[:,3] > boxes[:,1])
        elif isinstance(boxes, (np.ndarray, np.generic) ):
            keep = np.logical_and(boxes[:,2] > boxes[:,0], boxes[:,3] > boxes[:,1])
        return keep

    def normalize_boxes(self, human_boxes, boxes, stride):
        # clone the boxes and not change the original input boxes
        boxes = boxes.clone().detach()

        boxes[:,0] = boxes[:,0]-human_boxes[0]
        boxes[:,1] = boxes[:,1]-human_boxes[1]
        boxes[:,2] = boxes[:,2]-human_boxes[0]
        boxes[:,3] = boxes[:,3]-human_boxes[1]
        human_width = human_boxes[2] - human_boxes[0]
        human_height = human_boxes[3] - human_boxes[1]
        boxes = self.crop_boxes(boxes, (int(human_height), int(human_width)))

        # normalize the boxes using the roi size
        roi_h = self.cfg.roi_pool_h
        roi_w = self.cfg.roi_pool_w

        boxes[:,0] = boxes[:,0] / human_width * roi_w * stride
        boxes[:,1] = boxes[:,1] / human_height * roi_h * stride
        boxes[:,2] = boxes[:,2] / human_width * roi_w * stride
        boxes[:,3] = boxes[:,3] / human_height * roi_h * stride
        return boxes

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
        
    def postprocess(self, pred, roi_boxes_batch):
        heatmap = pred['heatmap'].sigmoid_()
        heatmap_out = heatmap.clone()

        k=100
        heatmap = point_nms(heatmap)
        scores, categories, ys, xs, inds = topk_ind(heatmap, k=k)
        #mask = topk_mask(heatmap, k=k)
        batch_size = heatmap.size(0)
        #scores= heatmap.masked_select(mask).view(batch_size, k)
        #ys, xs, categories = decode_mask(mask) # (batch_size, k)
        if 'offset' in self.parts:
            offset = pred['offset']
            offset = decode_by_ind(offset, inds)
            #offset = offset.masked_select(mask)
            
        if 'width_height' in self.parts:
            width_height = pred['width_height']
            width_height = decode_by_ind(width_height, inds)
            #width_height = width_height.masked_select(mask)
        #print('xs shape', xs.shape)
        #print('ys shape', ys.shape)
        #print('offset shape', offset.shape)
        #print('width height shape', width_height.shape)
        roi_per_pic = [len(roi) for roi in roi_boxes_batch]
        boxes = recover_roi_boxes(xs, ys, offset, width_height, self.stride, roi_boxes_batch, self.cfg.roi_pool_h, self.cfg.roi_pool_w)
        boxes = torch.split(boxes, roi_per_pic)
        categories = categories+1
        categories = torch.split(categories, roi_per_pic)
        scores = torch.split(scores, roi_per_pic)
        boxes, categories, scores = merge_roi_boxes(boxes, categories, scores, iou_threshold=self.cfg.nms_thresh)
        result = {}
        result['heatmap'] = heatmap_out
        result['offset'] = offset
        result['width_height'] = width_height
        result['boxes'] = boxes
        result['scores'] = scores
        result['labels'] = categories

        return result

    @torch.no_grad()
    def select_proposals(self, proposals, targets, image_sizes):
        if proposals is not None:
            # remove small human proposals
            proposals = self.remove_small_boxes_batch(proposals, min_area=self.min_area)

            # add gt human box
            proposals = self.add_gt_boxes( proposals, targets)
        else:
            proposals = self.gen_gt_boxes(targets)
            #proposals = self.gen_gt_boxes_debug(targets)

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

    def gen_gt_boxes_debug(self, targets):
        proposal_out = []
        for target in targets:
            human_box = torch.tensor([[0, 0, 416,  416]], dtype=target['boxes'].dtype, device=target['boxes'].device)
            proposal_out.append(human_box)
        return proposal_out

        
def gen_human_box( boxes, x_margin_min=5, x_margin_max=20, y_margin_min=20, y_margin_max=40):
    x_margin = random.randint(x_margin_min, x_margin_max)
    y_margin = random.randint(y_margin_min, y_margin_max)
    x1 = min(boxes[:,0]) - x_margin
    y1 = min(boxes[:,1]) - y_margin
    x2 = max(boxes[:,2]) + x_margin
    y2 = max(boxes[:,3]) + y_margin
    human_box = torch.tensor([[x1,y1,x2,y2]], dtype=boxes.dtype, device=boxes.device)
    return human_box

def recover_roi_boxes(xs, ys, offset, width_height, down_stride, roi_boxes_batch, roi_pool_h, roi_pool_w):

    roi_boxes = torch.cat(roi_boxes_batch, dim=0)
    roi_w = roi_boxes[:,2] - roi_boxes[:,0]
    roi_h = roi_boxes[:,3] - roi_boxes[:,1]
    w_scale = (roi_w / roi_pool_w)[:,None]
    h_scale = (roi_h / roi_pool_h)[:,None]
    #xs = (xs + offset[:,0,:])*down_stride*w_scale
    #ys = (ys + offset[:,1,:])*down_stride*h_scale
    xs = (xs + offset[:,0,:])*w_scale
    ys = (ys + offset[:,1,:])*h_scale
    #xs = (xs )*down_stride*w_scale
    #ys = (ys )*down_stride*h_scale
    width = width_height[:,0,:]*w_scale
    height = width_height[:,1,:]*h_scale
    #width = width_height[:,0,:]*w_scale*down_stride
    #height = width_height[:,1,:]*h_scale*down_stride
    x1 = xs - width/2 + roi_boxes[:,0][:,None]
    x2 = xs + width/2 + roi_boxes[:,0][:,None]
    y1 = ys - height/2 + roi_boxes[:,1][:,None]
    y2 = ys + height/2 + roi_boxes[:,1][:,None]
    boxes = torch.stack([x1, y1, x2, y2], dim=2)
    return  boxes

def merge_roi_boxes(boxes, labels, scores, iou_threshold=0.5):
    boxes_all = []
    labels_all = []
    scores_all = []
    for boxes_im, labels_im, scores_im in zip(boxes, labels, scores):
        boxes_im = boxes_im.reshape((-1,4))
        labels_im = labels_im.reshape(-1)
        scores_im = scores_im.reshape(-1)
        keep = batched_nms(boxes_im, scores_im, labels_im, iou_threshold=iou_threshold)
        boxes_im = boxes_im[keep]
        labels_im = labels_im[keep]
        scores_im = scores_im[keep]

        boxes_all.append(boxes_im)
        labels_all.append(labels_im)
        scores_all.append(scores_im)
    return boxes_all, labels_all, scores_all
