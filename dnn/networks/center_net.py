import torch
import torch.nn as nn
import numpy as np

from .one_stage_detector import OneStageDetector
from .heads import CommonHead, ComposedHead, CommonHeadWithSigmoid
from .losses import FocalLossHeatmap, L1LossWithMask, L1LossWithInd
from .common import init_focal_loss_head, init_head_gaussian
from ...data.datasets.coco_center import generate_ellipse_gaussian_heatmap, generate_ind, generate_offset, generate_width_height

class CenterNet(OneStageDetector):
    def __init__(self, backbone, num_classes, parts=['heatmap'], cfg=None, neck=None, pred_heads=None, loss_weight=None, max_obj=128):
        #super(OneStageDetector,self).__init__()
        if pred_heads is None:
            if neck is None:
                in_channel = backbone.out_channel
            else:
                in_channel = neck.out_channel
            heads = get_center_head(in_channel, num_classes)
        self.parts = parts
        self.down_stride = 4
        self.num_classes = num_classes
        self._max_obj = max_obj

        losses = CenterNetLoss(parts, loss_weight=loss_weight)

        #self.backbone = backbone
        #self.neck = neck
        #self.pred_heads = pred_heads
        #self.losses = losses
        super().__init__(backbone, heads, losses, neck=neck)

    def forward(self, inputs, targets=None):
        if self.training and targets is None:
            raise ValueError('targets should not be None during the training')

        features = self.backbone(inputs['data'])
        #print(features.keys())
        if self.neck is not None:
            features = self.neck(features)
        pred = self.pred_heads(features)
        #return pred

        if self.training:
            device = pred['heatmap'].device
            centernet_targets = self.generate_targets(inputs, targets, device)
            #return centernet_targets
            loss = self.losses(pred, centernet_targets)
            # for debug
            return loss
        
        output = self.postprocess(pred, inputs)
        return output

    def postprocess(self, pred, inputs):
        heatmap = pred['heatmap'].sigmoid_()
        #heatmap = pred['heatmap']

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
        boxes = recover_boxes(xs, ys, offset, width_height, self.down_stride)
        result = {}
        result['offset'] = offset
        result['width_height'] = width_height
        result['boxes'] = boxes
        result['scores'] = scores
        result['category'] = categories

        return result

    @torch.no_grad()
    def generate_targets(self, inputs, targets, device):
        heatmaps_all = []
        offset_all = []
        width_height_all = []
        ind_all = []
        ind_mask_all = []
        centernet_targets = {}

        _,_, height, width = inputs['data'].shape
        heatmap_w = width // self.down_stride
        heatmap_h = height // self.down_stride

        for boxes, labels in zip(targets['boxes'], targets['cat_labels']):
            #boxes = target['boxes']
            #boxes = self.normalize_boxes(human_box, boxes)
            boxes = boxes.detach().cpu().numpy() / self.down_stride
            #labels = target['labels']
            keep = self.find_valid_boxes(boxes)
            boxes  = boxes[keep]
            labels = labels[keep]

            center_x = (boxes[:,0] + boxes[:,2])/2 
            center_y = (boxes[:,1] + boxes[:,3])/2 
            boxes_w = boxes[:,2] - boxes[:,0]
            boxes_h = boxes[:,3] - boxes[:,1]

            # Follow the source code from centernet, radius calculate by original box size
            boxes_w = boxes_w * self.down_stride
            boxes_h = boxes_h * self.down_stride

            heatmaps = generate_ellipse_gaussian_heatmap(self.num_classes, heatmap_w, heatmap_h, center_x, center_y, boxes_w, boxes_h, labels)
        
            offset = generate_offset(center_x, center_y, self._max_obj)
            width_height = generate_width_height(boxes, self._max_obj)

            ind = np.zeros(self._max_obj, dtype=int)
            ind = generate_ind(ind, center_x, center_y, heatmap_w)
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

        for k,v in centernet_targets.items():
            centernet_targets[k] = torch.from_numpy(v).to(device)
        return centernet_targets

    def find_valid_boxes(self, boxes):
        if torch.is_tensor(boxes):
            keep = torch.logical_and(boxes[:,2] > boxes[:,0], boxes[:,3] > boxes[:,1])
        elif isinstance(boxes, (np.ndarray, np.generic) ):
            keep = np.logical_and(boxes[:,2] > boxes[:,0], boxes[:,3] > boxes[:,1])
        return keep

class CenterNetLoss(nn.Module):
    def __init__(self, loss_parts, loss_weight=None):
        super().__init__()
        self.loss_parts = loss_parts
        if loss_weight is None:
            loss_weight = {'heatmap':1., 'offset':0.5, 'width_height':0.01}
        self.loss_weight = loss_weight
        if 'heatmap' in loss_parts:
            self.heatmap_loss = FocalLossHeatmap(alpha=0.5, gamma=2)
        if 'offset' in loss_parts:
            #self.offset_loss = L1LossWithMask()
            self.offset_loss = L1LossWithInd()
        if 'width_height' in loss_parts:
            #self.width_height_loss = L1LossWithMask() 
            self.width_height_loss = L1LossWithInd() 
    
    def forward(self, pred, targets ):
        losses = {}
        if 'heatmap' in self.loss_parts:
            losses['heatmap'] = self.heatmap_loss(pred['heatmap'], targets['heatmap']) * self.loss_weight['heatmap']
        if 'offset' in self.loss_parts:
            #losses['offset'] = self.offset_loss(pred['offset'], targets['mask'], targets['offset']) * self.loss_weight['offset']
            losses['offset'] = self.offset_loss(pred['offset'], targets['ind'],targets['ind_mask'], targets['offset']) * self.loss_weight['offset']
        if 'width_height' in self.loss_parts:
            #losses['width_height'] = self.width_height_loss(pred['width_height'], targets['mask'], targets['width_height']) * self.loss_weight['width_height']
            losses['width_height'] = self.width_height_loss(pred['width_height'], targets['ind'], targets['ind_mask'], targets['width_height']) * self.loss_weight['width_height']
        return losses



def get_center_head(in_channel, num_classes):
    head_names = ['heatmap', 'offset', 'width_height']
    heatmap_head = CommonHead(num_classes, in_channel, head_conv_channel=64)
    init_focal_loss_head(heatmap_head)

    offset_head = CommonHead(2, in_channel, head_conv_channel=64)
    init_head_gaussian(offset_head)

    witdh_height_head = CommonHead(2, in_channel, head_conv_channel=64)
    init_head_gaussian(witdh_height_head)

    heads = [heatmap_head, offset_head, witdh_height_head]
    return ComposedHead(head_names, heads)

def point_nms(heatmap, kernel_size=3):
    padding = (kernel_size - 1) // 2
    heatout = nn.functional.max_pool2d(heatmap, kernel_size=kernel_size, padding=padding, stride=1 )
    #print((heatout==1).nonzero())
    mask = (heatout == heatmap).float()
    return mask*heatmap

def topk_ind(heatmap, k=100):
    n, c, h, w = heatmap.shape
    topk, inds = torch.topk(heatmap.view(n,-1), k=k)
    scores = topk

    categories = inds // (w*h)
    topk_inds = inds % (w*h)
    ys = topk_inds // w
    xs = topk_inds % w
    return scores, categories, ys, xs, topk_inds

def decode_by_ind(offset, ind):
    n,c,h,w = offset.shape
    offset = offset.view(n,c, -1)
    ind = ind.unsqueeze(1).expand(n,c, ind.size(1))
    offset = offset.gather(2, ind)
    return offset


def topk_mask(heatmap, k=100):
    n, c, h, w = heatmap.shape
    # getting the topk in 2 demension
    topk, inds = torch.topk(heatmap.view(n,-1), k=k)
    # set the mask in 2 demension and transfer to desired shape
    mask = torch.zeros((n, c*h*w), dtype=bool)
    mask = mask.scatter(1, inds, True)
    mask = mask.view(n,c,h,w)
    return mask
    
def decode_mask(mask):
    # decode other stuff from the mask indexes
    n, c, h, w = mask.shape
    indexes = mask.nonzero()
    categories = indexes[:,1].view(n,-1)
    ys = indexes[:,2].view(n,-1)
    xs = indexes[:,1].view(n,-1)
    return ys, xs, categories

def recover_boxes(xs, ys, offset, width_height, down_stride):
    xs = (xs + offset[:,0,:])*down_stride
    ys = (ys + offset[:,1,:])*down_stride
    width = width_height[:,0,:]
    height = width_height[:,1,:]
    x1 = xs - width/2
    x2 = xs + width/2
    y1 = ys - height/2
    y2 = ys + height/2
    boxes = torch.stack([x1, y1, x2, y2], dim=2)
    return  boxes
    
