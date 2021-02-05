import torch
from torch import nn
import numpy as np
from torchvision.ops.boxes import batched_nms 

from ..center_net import get_center_head
from ..center_net import CenterNetLoss, decode_by_ind, point_nms, topk_ind
from ....data.datasets.coco_center import generate_ellipse_gaussian_heatmap, generate_ind, generate_offset, generate_width_height

class RoICenterNetHeads(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg

        self.centernet_heads = get_center_head(cfg.out_feature_num, cfg.class_num)
        self._max_obj = cfg.max_obj
        self.class_num = cfg.class_num
        self.roi_pool_h = cfg.roi_pool_h
        self.roi_pool_w = cfg.roi_pool_w

        loss_parts = cfg.loss_parts
        loss_weight = cfg.loss_weight
        self.centernet_loss = CenterNetLoss(loss_parts, loss_weight=loss_weight)
        self.parts = loss_parts

        if hasattr(cfg, 'dataset_label'):
            self.dataset_label = cfg.dataset_label
        else:
            self.dataset_label = None

    def forward(self, roi_boxes, features, stride, inputs, targets=None):
        '''
            features: dict(key:tensor)
        '''
        assert len(features)==1
        features = list(features.values())[0]
        pred = self.centernet_heads(features)
        #return pred['heatmap'].sigmoid_()
        #label_pre, bbox_pre = self.faster_rcnn_head(rois)
        if self.training:
            centernet_targets = self.generate_targets(targets, stride, self.class_num, self.roi_pool_w, self.roi_pool_h)
            device = features.device
            for k,v in centernet_targets.items():
                centernet_targets[k] = torch.from_numpy(v).to(device)
            loss = self.centernet_loss(pred, centernet_targets)
            return loss
        else:
            #results = self.inference_result(label_pre, bbox_pre, proposals)
            results = self.postprocess(pred, roi_boxes, stride)
            # for debug
            #return proposals, results
            return results

    @torch.no_grad()
    def generate_targets(self, targets, stride, class_num, width, height):
        '''
            width: heatmap width
            height: heatmap height
        '''
        heatmaps_all = []
        offset_all = []
        width_height_all = []
        ind_all = []
        ind_mask_all = []
        centernet_targets = {}
        for target in targets:
            boxes = target['boxes'].cpu().detach().numpy()
            labels = target['labels'].cpu().detach().numpy()

            center_x = (boxes[:,0] + boxes[:,2])/2 /stride
            center_y = (boxes[:,1] + boxes[:,3])/2 /stride
            boxes_w = boxes[:,2] - boxes[:,0]
            boxes_h = boxes[:,3] - boxes[:,1]
            boxes_w = boxes_w
            boxes_h = boxes_h

            heatmaps = generate_ellipse_gaussian_heatmap(class_num, width, height, center_x, center_y, boxes_w, boxes_h, labels)
        
            offset = generate_offset(center_x, center_y, self._max_obj)
            width_height = generate_width_height(boxes, self._max_obj)

            ind = np.zeros(self._max_obj, dtype=int)
            ind = generate_ind(ind, center_x, center_y, width)
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

    def postprocess(self, pred, roi_boxes_batch, stride):
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
        boxes = recover_roi_boxes(xs, ys, offset, width_height, stride, roi_boxes_batch, self.cfg.roi_pool_h, self.cfg.roi_pool_w)
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
    width = width_height[:,0,:]*w_scale / down_stride
    height = width_height[:,1,:]*h_scale / down_stride
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
