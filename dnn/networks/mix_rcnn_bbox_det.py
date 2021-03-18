import torch
import math
import time
from collections import OrderedDict
from .pooling import RoiAliagnFPN
from .general_detector import GeneralDetector
from torchvision.ops import roi_align, nms

class MixRCNNBBoxDetector(torch.nn.Module):
    def __init__(self, backbone, heads, roi_pooler, neck=None, targets_converter=None, inputs_converter=None, cfg=None, second_loss_weight=None, third_loss_weight=None,expand_ratio=None, test_mode='both', debug_time=False):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        #self.rpn = heads['rpn']
        #self.roi_head = heads['bbox']
        self.human_detection_head = heads['first_head']
        self.bbox_head = heads['second_head']
        self.roi_detection_head = heads['third_head']
        self.roi_pooler = roi_pooler
        self.targets_converter = targets_converter
        self.inputs_converter = inputs_converter
        self.feature_names = ['0', '1', '2', '3']
        self.strides = None
        self.second_loss_weight = second_loss_weight
        self.third_loss_weight = third_loss_weight
        self.test_mode = test_mode
        self.expand_ratio = expand_ratio

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

            losses_second_bbox, outfit_boxes = self.bbox_head(features, human_proposal, self.strides, inputs=inputs, targets=targets )

            if self.second_loss_weight is not None:
                for k,v in losses_second_bbox.items():
                    v *= self.second_loss_weight
                    losses_second_bbox['second_'+k] = v
                losses_second_bbox = self.second_loss_weight * losses_second_bbox

            outfit_boxes, roi_features, targets_for_second = self.roi_pooler(outfit_boxes, feature_second, stride_second, inputs, targets)
            if self.inputs_converter is not None:
                second_inputs = self.inputs_converter(stride_second, len(roi_features))
            else:
                second_inputs = inputs
            
            # convert the targets according to the roi size and proposal
            if self.targets_converter is not None:
                second_targets = self.targets_converter(outfit_boxes, targets_for_second, stride_second)
            else:
                second_targets = targets
            
            roi_features = {'0':roi_features}
            losses_second_roi = self.roi_detection_head(outfit_boxes, roi_features, stride_second, inputs=second_inputs, targets=second_targets )
            #return losses_second_roi
            losses_second= {}
            for k,v in losses_second_roi.items():
                if self.third_loss_weight is not None:
                    v *= self.third_loss_weight
                losses_second['third_'+k] = v

            losses = {**losses_first, **losses_second, **losses_second_bbox}
            return losses
        else:
            human_results = self.human_detection_head(features, inputs, targets=targets)
            if self.test_mode == 'first':
                return human_results
            human_boxes = human_results['boxes']
            human_scores = human_results['scores']
            human_boxes = [human_box[torch.argmax(human_score)][None,:].clone() for human_box, human_score in zip(human_boxes, human_scores)]
            human_thresh = 0.8
            #human_boxes = [human_box[human_score>human_thresh].clone() if (human_score>human_thresh).any() else human_box[torch.argmax(human_score)][None,:].clone() for human_box, human_score in zip(human_boxes, human_scores)]
            outfit_boxes = self.bbox_head(features, human_boxes, self.strides, inputs=inputs, targets=targets )

            #if self.test_mode =='second':
            #    return bbox_results
            #elif self.test_mode == 'both':
            #    return human_results, bbox_results 
            #else:
            #    raise ValueError('Unknow test mode {}'.format(self.test_mode))

            if self.expand_ratio is not None:
                outfit_boxes = self.expand_boxes_batch(outfit_boxes, inputs['image_sizes'], self.expand_ratio)
            outfit_proposal, roi_features, targets_for_second = self.roi_pooler(outfit_boxes, feature_second, stride_second, inputs, targets)

            if self.inputs_converter is not None:
                second_inputs = self.inputs_converter(stride_second, len(roi_features))
            else:
                second_inputs = inputs

            roi_features = {'0':roi_features}
            results = self.roi_detection_head(outfit_proposal, roi_features, stride_second, inputs=second_inputs, targets=targets)
            results = self.post_process(results, inputs)
            if self.test_mode == 'second':
                return results
            elif self.test_mode == 'both':
                return human_results, results 
            else:
                raise ValueError('Unknow test mode {}'.format(self.test_mode))

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

    def add_human_box_to_gt(self, human_proposal, inputs, targets, human_dataset_label):
        valid_ind = inputs['dataset_label'] != human_dataset_label
        valid_targets = [target for target, ind in zip(targets, valid_ind) if ind ]
        assert len(valid_targets) == len(human_proposal)
        for target, proposal in zip(valid_targets, human_proposal):
            proposal




    @torch.no_grad()
    def expand_boxes_batch(self, boxes_batch, image_sizes, ratio=0.2):
        new_boxes_batch = [self.expand_boxes(boxes, image_size, ratio) for boxes, image_size in zip(boxes_batch, image_sizes)]
        return new_boxes_batch

    @torch.no_grad()
    def expand_boxes(self, boxes, image_size, ratio=0.2):
        boxes_w = boxes[:,2] - boxes[:,0]
        boxes_h = boxes[:,3] - boxes[:,1]

        boxes_w_half = boxes_w * (ratio + 1) / 2
        boxes_h_half = boxes_h * (ratio + 1) / 2

        boxes_xc = (boxes[:,2] + boxes[:,0]) / 2
        boxes_yc = (boxes[:,3] + boxes[:,1]) / 2

        boxes_x1 = boxes_xc - boxes_w_half
        boxes_y1 = boxes_yc - boxes_h_half
        boxes_x2 = boxes_xc + boxes_w_half
        boxes_y2 = boxes_yc + boxes_h_half

        height, width = image_size
        boxes_x1 = boxes_x1.clamp(min=0, max=width)
        boxes_y1 = boxes_y1.clamp(min=0, max=height)
        boxes_x2 = boxes_x2.clamp(min=0, max=width)
        boxes_y2 = boxes_y2.clamp(min=0, max=height)

        return torch.stack([boxes_x1, boxes_y1, boxes_x2, boxes_y2], dim=1)


    def get_strides(self, inputs, features):
        if isinstance(features, dict):
            features = list(features.values())
        grid_sizes = tuple([feature_map.shape[-2:] for feature_map in features])
        image_size = inputs['data'].shape[-2:]
        #strides = tuple((image_size[0] / g[0] + image_size[1] / g[1])/2 for g in grid_sizes)
        strides = tuple(math.pow(2,math.ceil(math.log2((image_size[0] / g[0] + image_size[1] / g[1])/2))) for g in grid_sizes)
        return strides

    def set_strides(self, inputs, features):
        if self.strides is None:
            self.strides = self.get_strides(inputs, features)

    def reset_time(self):
        self.total_time = {'feature':0.0, 'rpn':0.0, 'roi_head':0.0}

