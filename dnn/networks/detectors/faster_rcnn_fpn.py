import torch
import math
import time
from collections import OrderedDict
from torchvision.ops import roi_align, nms
from .two_stage_detector import TwoStageDetector

from .build import DETECTOR_REG

@DETECTOR_REG.register()
class FasterRCNNFPN(TwoStageDetector):
    def __init__(self, backbone, rpn, roi_head, neck=None, training=True, debug_time=False, just_rpn=False):
        super(FasterRCNNFPN, self).__init__(backbone=backbone, neck=neck, rpn=rpn, roi_head=roi_head, training=training )
        #self.strides = None
        #self.just_rpn=just_rpn

        #if debug_time:
        #    self.total_time = {'feature':0.0, 'rpn':0.0, 'roi_head':0.0}
        #self.debug_time = debug_time
        

    def forward_old(self, inputs, targets=None):
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
        # This is new place to try
        rpn_features = OrderedDict()
        for k in self.rpn_feature_names:
            rpn_features[k] = features[k]

        if self.training:
            proposals, losses_rpn = self.rpn(inputs, rpn_features, targets)
        else:
            proposals, scores = self.rpn(inputs, rpn_features, targets)

        if debug_time:
            rpn_time = time.time()
            self.total_time['rpn'] += rpn_time - feature_time 


        roi_input_features = OrderedDict()
        for k in self.roi_feature_names:
            roi_input_features[k] = features[k]

        if self.strides is None:
            strides = self.get_strides(inputs, roi_input_features)
        else:
            strides = self.strides
        #for proposal in proposals:
        #    print('proposal shape', proposal.shape)

        if self.training:
            losses_roi = self.roi_head(proposals, roi_input_features, strides, targets=targets)
            losses = {**losses_rpn, **losses_roi}
            if debug_time:
                roi_head_time = time.time()
                self.total_time['roi_head'] += roi_head_time - rpn_time 
            return losses
        else:
            if self.just_rpn:
                results = {'boxes':proposals, 'scores':scores, 'labels':[torch.ones_like(score) for score in scores]}
                return results
            results = self.roi_head(proposals, roi_input_features, strides, targets=targets)
            results = self.post_process(results, inputs)
            if debug_time:
                roi_head_time = time.time()
                self.total_time['roi_head'] += roi_head_time - rpn_time 
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

