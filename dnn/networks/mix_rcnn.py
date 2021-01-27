import torch
import time
from collections import OrderedDict
from .pooling import RoiAliagnFPN
from .general_detector import GeneralDetector
from torchvision.ops import roi_align, nms

class MixRCNN(GeneralDetector):
    def __init__(self, backbone, neck=None, heads=None, cfg=None, training=True, second_loss_weight=None, test_mode='both', debug_time=False):
        super(GeneralDetector, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn = heads['rpn']
        self.roi_head = heads['bbox']
        self.second_roi_head = heads['second_box']
        self.training = training
        self.feature_names = ['0', '1', '2', '3']
        self.strides = None
        self.second_loss_weight = second_loss_weight
        self.test_mode = test_mode

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

        if self.training:
            proposals, losses_rpn = self.rpn(inputs, features, targets)
        else:
            proposals, scores = self.rpn(inputs, features, targets)

        if debug_time:
            rpn_time = time.time()
            self.total_time['rpn'] += rpn_time - feature_time 


        if self.strides is None:
            strides = self.get_strides(inputs, features)
            self.strides = strides
        else:
            strides = self.strides
        #for proposal in proposals:
        #    print('proposal shape', proposal.shape)
        feature_second = features['0']
        stride_second = self.strides[0]

        if self.training:
            losses_roi, human_proposal = self.roi_head(proposals, features, strides, targets=targets, inputs=inputs)
            #return {**losses_rpn, **losses_roi}
            
            losses_second_roi = self.second_roi_head(human_proposal, feature_second, stride_second, inputs=inputs, targets=targets )
            #return losses_second_roi
            losses_second_roi_new = {}
            for k,v in losses_second_roi.items():
                if self.second_loss_weight is not None:
                    v *= self.second_loss_weight
                losses_second_roi_new['second_'+k] = v

            losses = {**losses_rpn, **losses_roi, **losses_second_roi_new}
            #losses = {**losses_rpn, **losses_roi }
            if debug_time:
                roi_head_time = time.time()
                self.total_time['roi_head'] += roi_head_time - rpn_time 
            return losses
        else:
            human_results = self.roi_head(proposals, features, strides, targets=targets)
            if self.test_mode == 'first':
                human_results_out = self.post_process(human_results, inputs)
                return human_results_out
            else:
                human_boxes = human_results['boxes']
                human_scores = human_results['scores']
                #human_boxes = [human_box[torch.argmax(human_score)][None,:].clone() for human_box, human_score in zip(human_boxes, human_scores)]
                human_boxes = [human_box[human_score>0.5].clone() if (human_score>0.5).any() else human_box[torch.argmax(human_score)][None,:].clone() for human_box, human_score in zip(human_boxes, human_scores)]
                results = self.second_roi_head(human_boxes, feature_second, stride_second, inputs=inputs, targets=targets)
                results = self.post_process(results, inputs)
            if self.test_mode == 'second':
                return results
            elif self.test_mode == 'both':
                human_results_out = self.post_process(human_results, inputs)
                return human_results_out, results 
            else:
                raise ValueError('Unknow test mode {}'.format(self.test_mode))
            #human_results['boxes'] = [human_box[torch.argmax(human_score)][None,:] for human_box, human_score in zip(human_results['boxes'], human_scores)]
            #return human_results_out
            #return results
            # for debug

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
        strides = tuple((image_size[0] / g[0] + image_size[1] / g[1])/2 for g in grid_sizes)
        return strides

    def reset_time(self):
        self.total_time = {'feature':0.0, 'rpn':0.0, 'roi_head':0.0}

