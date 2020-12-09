import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
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

class RoICenterNet(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.train_box_iou_thre = cfg.centernet_head.train_box_iou_thre
        self.min_area = cfg.centernet_head.min_area

        self.roi_align = ROIAlignBatch(cfg.centernet_head.roi_pool_h,
                                      cfg.centernet_head.roi_pool_w,
                                      sampling=-1)
        #self.faster_rcnn_head = FastRCNNHead(cfg)
        self.centernet_heads = get_center_head(cfg.out_feature_num, cfg.class_num)
        self._max_obj = cfg.centernet_head.max_obj

        loss_parts = cfg.centernet_head.loss_parts
        loss_weight = cfg.centernet_head.loss_weight
        self.centernet_loss = CenterNetLoss(loss_parts, loss_weight=loss_weight)
        self.parts = loss_parts

        #self.box_coder = AnchorBoxesCoder(box_code_clip=None)
        #self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        #self.label_loss = nn.CrossEntropyLoss(reduction='mean')

        if hasattr(cfg.centernet_head, 'dataset_label'):
            self.dataset_label = cfg.centernet_head.dataset_label
        else:
            self.dataset_label = None

    def forward(self, proposals, features, stride, inputs=None, targets=None):
        image_sizes = inputs['image_sizes']
        self.stride = stride
        if self.dataset_label is not None:
            dataset_ind = inputs['dataset_label'] == self.dataset_label
            if self.training:
                features = features[dataset_ind]
            #image_sizes = image_sizes[dataset_ind]
            image_sizes =[im_size for im_size,label in zip(image_sizes, dataset_ind) if label ]
            if targets is not None:
                targets =[target for target,label in zip(targets, dataset_ind) if label ]

            ## TODO: Maybe I should change this to somewhere else
            #if not self.training:
            #    print(inputs['dataset_label'])
            #    proposals =[proposal for proposal,label in zip(proposals, dataset_ind) if label ]
            #    print('proposals:', proposals)

        if self.training:
            proposals = self.select_proposals(proposals, targets, image_sizes)
            centernet_targets = self.generate_targets(proposals, targets)
            #return proposals, centernet_targets
            #return proposals
            


        # rois: tuple(Tensor_image1, Tensor_image2)
        #print('feature size', features.shape)
        #print('proposals', len(proposals))
        rois = self.roi_align(features, proposals, stride)
        rois = torch.cat(rois, dim=0)

        pred = self.centernet_heads(rois)
        #label_pre, bbox_pre = self.faster_rcnn_head(rois)
        if self.training:
            device = rois.device
            for k,v in centernet_targets.items():
                centernet_targets[k] = torch.from_numpy(v).to(device)
            loss = self.centernet_loss(pred, centernet_targets)
            return loss
        else:
            #results = self.inference_result(label_pre, bbox_pre, proposals)
            results = self.postprocess(pred, proposals)
            return results

    @torch.no_grad()
    def generate_targets(self, human_boxes_batch, targets):
        heatmaps_all = []
        offset_all = []
        width_height_all = []
        ind_all = []
        ind_mask_all = []
        centernet_targets = {}
        for human_boxes, target in zip(human_boxes_batch, targets):
            for human_box in human_boxes:
                boxes = target['boxes']
                boxes = self.normalize_boxes(human_box, boxes)
                boxes = boxes.detach().cpu().numpy()
                keep = self.find_valid_boxes(boxes)
                boxes  = boxes[keep]
                labels = target['labels'][keep]

                center_x = (boxes[:,0] + boxes[:,2])/2 
                center_y = (boxes[:,1] + boxes[:,3])/2 
                boxes_w = boxes[:,2] - boxes[:,0]
                boxes_h = boxes[:,3] - boxes[:,1]
                boxes_w = boxes_w * self.stride
                boxes_h = boxes_h * self.stride

                heatmaps = generate_ellipse_gaussian_heatmap(self.cfg.class_num, self.cfg.centernet_head.roi_pool_w ,self.cfg.centernet_head.roi_pool_h, center_x, center_y, boxes_w, boxes_h, labels)
            
                offset = generate_offset(center_x, center_y, self._max_obj)
                width_height = generate_width_height(boxes, self._max_obj)

                ind = np.zeros(self._max_obj, dtype=int)
                ind = generate_ind(ind, center_x, center_y, self.cfg.centernet_head.roi_pool_w)
                ind_mask = np.zeros(self._max_obj, dtype=int)
                ind_mask[:len(center_x)] = 1

                heatmaps_all.append(heatmaps)
                offset_all.append(offset)
                width_height_all.append(width_height)
                ind_all.append(ind)
                ind_mask_all.append(ind_mask)
        centernet_targets['heatmap'] = np.stack(heatmaps_all)
        centernet_targets['offset'] = np.stack(offset_all)
        #targets['offset_map'] = offset_map
        centernet_targets['width_height'] = np.stack(width_height_all)
        #targets['width_height_map'] = width_height_map
        centernet_targets['ind'] = np.stack(ind_all)
        centernet_targets['ind_mask'] = np.stack(ind_mask_all)
        return centernet_targets
            

    def find_valid_boxes(self, boxes):
        if torch.is_tensor(boxes):
            keep = torch.logical_and(boxes[:,2] > boxes[:,0], boxes[:,3] > boxes[:,1])
        elif isinstance(boxes, (np.ndarray, np.generic) ):
            keep = np.logical_and(boxes[:,2] > boxes[:,0], boxes[:,3] > boxes[:,1])
        return keep

    def normalize_boxes(self, human_boxes, boxes):
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
        roi_h = self.cfg.centernet_head.roi_pool_h
        roi_w = self.cfg.centernet_head.roi_pool_w

        boxes[:,0] = boxes[:,0] / human_width * roi_w
        boxes[:,1] = boxes[:,1] / human_height * roi_h
        boxes[:,2] = boxes[:,2] / human_width * roi_w
        boxes[:,3] = boxes[:,3] / human_height * roi_h
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
        boxes = recover_roi_boxes(xs, ys, offset, width_height, self.stride, roi_boxes_batch, self.cfg.centernet_head.roi_pool_h, self.cfg.centernet_head.roi_pool_w)
        boxes = torch.split(boxes, roi_per_pic)
        categories = torch.split(categories, roi_per_pic)
        scores = torch.split(scores, roi_per_pic)
        boxes, categories, scores = merge_roi_boxes(boxes, categories, scores, iou_threshold=self.cfg.centernet_head.nms_thresh)
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
        # remove small human proposals
        proposals = self.remove_small_boxes_batch(proposals, min_area=self.min_area)

        # add gt human box
        proposals = self.add_gt_boxes( proposals, targets)

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

        
def gen_human_box( boxes, x_margin=5, y_margin=20):
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
