import torch
import math
import time
from collections import OrderedDict
from .pooling import RoiAliagnFPN
from .general_detector import GeneralDetector
from torchvision.ops import roi_align, nms

class RetinaNet(GeneralDetector):
    def __init__(self, backbone, neck=None, heads=None, cfg=None, training=True, feature_names=['0','1','2','p6', 'p7'], debug_time=False, just_rpn=False):
        super(GeneralDetector, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.retina_head = heads['retina_head']
        self.training = training
        self.feature_names = feature_names
        self.strides = None
        self.just_rpn=just_rpn

        if debug_time:
            self.total_time = {'feature':0.0, 'neck':0.0, 'retina_head':0.0}
        self.debug_time = debug_time
        

    def forward(self, inputs, targets=None):
        debug_time = self.debug_time
        if debug_time:
            start = time.time()
        features = self.backbone(inputs['data'])
        if debug_time:
            feature_time = time.time()
            self.total_time['feature'] += feature_time - start
        #print('feature keys:', features.keys())
        #for k,v in features.items():
        #    print('{}:{}'.format(k, v.shape))
        if self.neck is not None:
            features = self.neck(features)
        if debug_time:
            neck_time = time.time()
            self.total_time['neck'] += neck_time - feature_time

        #print('strides', strides)
        # This is new place to try
        rpn_features = OrderedDict()
        for k in self.feature_names:
            rpn_features[k] = features[k]

        if self.training:
            losses = self.retina_head(inputs, rpn_features, targets)
            return losses
        else:
            results = self.retina_head(inputs, rpn_features, targets)
            results = self.post_process(results, inputs)
            if debug_time:
                head_time = time.time()
                self.total_time['retina_head'] += head_time - neck_time
            return results


    def post_process(self, results, inputs):
        for i, (boxes, scores, labels) in enumerate(zip(results['boxes'], results['scores'], results['labels'])):

            if len(boxes)>0:
                if 'scale' in inputs:
                    scale = inputs['scale'][i]
                    if isinstance(scale, float):
                        boxes /= scale
                    else:  # scale w and scale h are seperated
                        boxes[...,0] = boxes[...,0] / scale[0]
                        boxes[...,2] = boxes[...,2] / scale[0]
                        boxes[...,1] = boxes[...,1] / scale[1]
                        boxes[...,3] = boxes[...,3] / scale[1]

            results['boxes'][i] = boxes
            results['scores'][i] = scores
            results['labels'][i] = labels
        return results


    def combine_dict(self, rois):
        rois = list(rois.values())
        roi_level_all = []
        if isinstance(rois[0], list):
            for roi_level in rois:
                roi_level = torch.cat(roi_level, dim=0)
                roi_level_all.append(roi_level)
        else:
            roi_level_all = rois
        rois = torch.cat(roi_level_all, dim=0)
        return rois

    def get_strides(self, inputs, features):
        if isinstance(features, dict):
            features = list(features.values())
        grid_sizes = tuple([feature_map.shape[-2:] for feature_map in features])
        image_size = inputs['data'].shape[-2:]
        #strides = tuple((image_size[0] / g[0] + image_size[1] / g[1])/2 for g in grid_sizes)
        strides = tuple(math.pow(2,math.ceil(math.log2((image_size[0] / g[0] + image_size[1] / g[1])/2))) for g in grid_sizes)
        return strides

    def reset_time(self):
        self.total_time = {'feature':0.0, 'rpn':0.0, 'roi_head':0.0}

