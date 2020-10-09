from collections import OrderedDict
import torch
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

    def forward(self, features, boxes, strides):
        feature_keys = list(features.keys())
        boxes_level, ind_level = self.split_boxes_by_size(boxes, feature_keys)
        assert len(feature_keys) == len(strides)
        rois = OrderedDict()
        for level, stride in zip(feature_keys,strides):
            feature_level = features[level]
            boxes_in_level = boxes_level[level]
            num_per_level = [len(box) for box in boxes_in_level]
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

    def split_boxes_by_size(self, boxes, feature_keys, k0=4):
        feature_num = len(feature_keys)
        boxes_level = OrderedDict([(k,[]) for k in feature_keys])
        ind_level = OrderedDict([(k,[]) for k in feature_keys])
        for im_id, boxes_im in enumerate(boxes):
            area = (boxes_im[:,3]-boxes_im[:,1]) * (boxes_im[:,2]-boxes_im[:,0])
            k = torch.floor_(k0 + torch.log2_(torch.sqrt_(area) / 224))
            k = k.clamp_(min=0, max=feature_num-1)
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
