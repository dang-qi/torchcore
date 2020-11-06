from collections import OrderedDict
import torch
import math
from torch import nn
from torchvision.ops import roi_align, RoIAlign

class RoiAliagnFPN(nn.Module):
    def __init__(self, pool_h, pool_w, sampling=-1):
        super().__init__()
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.out_size = [pool_h, pool_w]
        # scale is the spacial scale, which is equal to 1/stride
        # here we use list of scale as scales
        self.sampling = sampling
        self.k_max = None
        self.k_min = None

    def forward_old(self, features, boxes, strides):
        feature_keys = list(features.keys())
        boxes_level, ind_level = self.split_boxes_by_size(boxes, feature_keys)
        assert len(feature_keys) == len(strides)
        rois = OrderedDict()
        for level, stride in zip(feature_keys,strides):
            feature_level = features[level]
            boxes_in_level = boxes_level[level]
            num_per_level = [len(box) for box in boxes_in_level]
            #print('level {}: stride{}'.format(level, stride))
            #print('feature level shape',feature_level.shape)
            #print('boxes in level', boxes_in_level)
            roi_level = roi_align(feature_level, boxes_in_level, self.out_size, spatial_scale=1.0/stride, sampling_ratio=self.sampling)
            rois[level] = torch.split(roi_level, num_per_level, dim=0)
        rois = self.split_to_batch(rois)
        boxes = self.split_to_batch(boxes_level)
        #print('ind_level:', ind_level.keys())
        #for ind_t in ind_level[0]:
        #    print('ind_level[0]:', len(ind_t))
        inds = self.split_to_batch(ind_level)
        return rois, boxes, inds

    def forward(self, features, boxes, strides):
        feature_keys = list(features.keys())
        if self.k_min is None:
            self.setup_scale(strides)
        inds = self.split_boxes_by_size(boxes, feature_keys)
        assert len(feature_keys) == len(strides)
        rois = self.add_batch_number_to_boxes(boxes)
        roi_num = len(rois)
        roi_channel = features[feature_keys[0]].shape[1]
        dtype = features[feature_keys[0]].dtype
        device = features[feature_keys[0]].device
        rois_out = torch.zeros((roi_num, roi_channel, self.pool_h, self.pool_w), dtype=dtype, device=device)
        for i, (level, stride) in enumerate(zip(feature_keys,strides)):
            feature_level = features[level]
            ind_level = torch.where(inds==i)[0]
            boxes_level = rois[ind_level]
            #num_per_level = [len(box) for box in boxes_in_level]
            #print('level {}: stride{}'.format(level, stride))
            #print('feature level shape',feature_level.shape)
            #print('boxes in level', boxes_in_level)
            roi_level = roi_align(feature_level, boxes_level, self.out_size, spatial_scale=1.0/stride, sampling_ratio=self.sampling)
            rois_out[ind_level] = roi_level
        return rois_out

    def split_boxes_by_size(self, boxes, feature_keys, k0=4):
        boxes = torch.cat(boxes, dim=0)
        area = (boxes[:,3]-boxes[:,1]) * (boxes[:,2]-boxes[:,0])
        k = torch.floor(k0 + torch.log2(torch.sqrt(area) / 224))
        k = k.clamp(min=self.k_min, max=self.k_max)
        k = k-self.k_min
        return k

    def add_batch_number_to_boxes(self, boxes):
        cat_boxes = torch.cat(boxes, dim=0)
        device = cat_boxes.device
        dtype = cat_boxes.dtype
        inds = [torch.full((len(box), 1), i,dtype=dtype, device=device) for i, box in enumerate(boxes)]
        inds = torch.cat(inds)

        out = torch.cat([inds, cat_boxes], dim=1)
        return out

    def split_boxes_by_size_old(self, boxes, feature_keys, k0=4):
        feature_num = len(feature_keys)
        boxes_level = OrderedDict([(k,[]) for k in feature_keys])
        ind_level = OrderedDict([(k,[]) for k in feature_keys])
        for im_id, boxes_im in enumerate(boxes):
            area = (boxes_im[:,3]-boxes_im[:,1]) * (boxes_im[:,2]-boxes_im[:,0])
            k = torch.floor(k0 + torch.log2(torch.sqrt(area) / 224))
            k = k.clamp(min=0, max=feature_num-1)
            for i in range(feature_num):
                feature_key = feature_keys[i]
                ind = torch.where(k==i)
                boxes_level[feature_key].append(boxes_im[ind])
                # ind is tuple(ind,), so we just need ind[0]
                ind_level[feature_key].append(ind[0])
        return boxes_level, ind_level

    def split_to_batch(self, rois):
        batch_size = len(list(rois.values())[0])
        rois_batch = [[] for i in range(batch_size)]
        for k, v in rois.items():
            for i, v_level in enumerate(v):
                rois_batch[i].append(v_level)
        rois_batch = [torch.cat(v) for v in rois_batch]
        return rois_batch

    def setup_scale(self, strides):
        max_stride = max(strides)
        min_stride = min(strides)
        self.k_min = round(math.log2(min_stride))
        self.k_max = round(math.log2(max_stride))