import torch
import math
import time
from collections import OrderedDict
from .pooling import RoiAliagnFPN
from .general_detector import GeneralDetector
from torchvision.ops import roi_align, nms

class RoiCenterNetWithBackbone(GeneralDetector):
    def __init__(self, backbone, neck=None, heads=None, cfg=None, training=True, debug_time=False):
        super(GeneralDetector, self).__init__()
        self.backbone = backbone
        self.neck = neck
        #self.rpn = heads['rpn']
        #self.roi_head = heads['bbox']
        self.centernet_head = heads['roi_centernet']
        self.training = training
        self.feature_names = ['0' ]
        self.strides = None

        if debug_time:
            self.total_time = {'feature':0.0, 'rpn':0.0, 'roi_head':0.0}
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

        #print('strides', strides)

        features_new = OrderedDict()
        for k in self.feature_names:
            features_new[k] = features[k]
        features = features_new
        if self.strides is None:
            strides = self.get_strides(inputs, features)
            self.strides = strides
        else:
            strides = self.strides
        #for proposal in proposals:
        #    print('proposal shape', proposal.shape)
        feature_second = features['0']
        stride_second = self.strides[0]
        human_proposal = None

        if self.training:
            losses_centernet = self.centernet_head(human_proposal, feature_second, stride_second, inputs=inputs, targets=targets )
            #return losses_second_roi

            losses = losses_centernet
            return losses
        else:
            results = self.centernet_head(human_proposal, feature_second, stride_second, inputs=inputs, targets=targets)
            # for debug
            #human_proposal, results = self.centernet_head(human_proposal, feature_second, stride_second, inputs=inputs, targets=targets)
            results = self.post_process(results, inputs)
            # for debug
            #return human_proposal, results 
            return results 

    def post_process(self, results, inputs):
        for i, (boxes, scores, labels) in enumerate(zip(results['boxes'], results['scores'], results['labels'])):

            if len(boxes)>0:
                if 'scale' in inputs:
                    scale = inputs['scale'][i]
                    boxes = boxes / scale

            results['boxes'][i] = boxes
            results['scores'][i] = scores
            results['labels'][i] = labels
        return results

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

