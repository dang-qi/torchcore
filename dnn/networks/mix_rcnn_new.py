import torch
import time
from collections import OrderedDict
from .pooling import RoiAliagnFPN
from .general_detector import GeneralDetector
from torchvision.ops import roi_align, nms

class MixRCNN(torch.nn.Module):
    def __init__(self, backbone, heads, roi_pooler, neck=None, targets_converter=None, inputs_converter=None, cfg=None, training=True, second_loss_weight=None, debug_time=False):
        super(MixRCNN, self).__init__()
        self.backbone = backbone
        self.neck = neck
        #self.rpn = heads['rpn']
        #self.roi_head = heads['bbox']
        self.human_detection_head = heads['first_head']
        self.roi_detection_head = heads['second_head']
        self.roi_pooler = roi_pooler
        self.targets_converter = targets_converter
        self.inputs_converter = inputs_converter
        self.training = training
        self.feature_names = ['0', '1', '2', '3']
        self.strides = None
        self.second_loss_weight = second_loss_weight

        if debug_time:
            self.total_time = {'feature':0.0, 'rpn':0.0, 'roi_head':0.0}
        self.debug_time = debug_time
        

    def forward(self, inputs, targets=None):
        '''
            get features -> first head -> roi pooler -> second head
        '''
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

        features_new = OrderedDict()
        for k in self.feature_names:
            features_new[k] = features[k]
        features = features_new

        self.set_strides(inputs, features)

        feature_second = features['0']
        stride_second = self.strides[0]

        if self.training:
            losses_first, human_proposal = self.human_detection_head(features, inputs, targets)

            human_proposal, roi_features, targets_for_second = self.roi_pooler(human_proposal, feature_second, stride_second, inputs, targets)
            if self.inputs_converter is not None:
                second_inputs = self.inputs_converter(stride_second, len(roi_features))
            else:
                second_inputs = inputs
            
            if self.targets_converter is not None:
                second_targets = self.targets_converter(human_proposal, targets_for_second, stride_second)
            else:
                second_targets = targets
            
            roi_features = {'0':roi_features}
            losses_second_roi = self.roi_detection_head(human_proposal, roi_features, stride_second, inputs=second_inputs, targets=second_targets )
            #return losses_second_roi
            losses_second= {}
            for k,v in losses_second_roi.items():
                if self.second_loss_weight is not None:
                    v *= self.second_loss_weight
                losses_second['second_'+k] = v

            losses = {**losses_first, **losses_second}
            return losses
        else:
            human_results = self.human_detection_head(features, inputs, targets=targets)
            human_boxes = human_results['boxes']
            human_scores = human_results['scores']
            #human_boxes = [human_box[torch.argmax(human_score)][None,:].clone() for human_box, human_score in zip(human_boxes, human_scores)]
            human_boxes = [human_box[human_score>0.5].clone() if (human_score>0.5).any() else human_box[torch.argmax(human_score)][None,:].clone() for human_box, human_score in zip(human_boxes, human_scores)]
            human_proposal, roi_features = self.roi_pooler(human_boxes, feature_second, stride_second, inputs, targets)
            roi_features = {'0':roi_features}
            results = self.roi_detection_head(human_proposal, roi_features, stride_second, inputs=inputs, targets=targets)
            results = self.post_process(results, inputs)
            #human_results['boxes'] = [human_box[torch.argmax(human_score)][None,:] for human_box, human_score in zip(human_results['boxes'], human_scores)]
            #human_results_out = self.post_process(human_results, inputs)
            return human_results
            #return results
            # for debug
            return human_results, results 
            return human_results

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

    def set_strides(self, inputs, features):
        if self.strides is None:
            self.strides = self.get_strides(inputs, features)

    def reset_time(self):
        self.total_time = {'feature':0.0, 'rpn':0.0, 'roi_head':0.0}

