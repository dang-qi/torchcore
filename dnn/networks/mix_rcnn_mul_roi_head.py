import torch
import time
from collections import OrderedDict
from .pooling import RoiAliagnFPN
from .general_detector import GeneralDetector
from torchvision.ops import roi_align, nms

class MixRCNNMulRoIHead(torch.nn.Module):
    def __init__(self, backbone, heads, roi_pooler, neck=None, targets_converter=None, inputs_converter=None, cfg=None, training=True, second_loss_weight=None,expand_ratio=None, test_mode='both', debug_time=False):
        super(MixRCNNMulRoIHead, self).__init__()
        self.backbone = backbone
        self.neck = neck
        #self.rpn = heads['rpn']
        #self.roi_head = heads['bbox']
        self.human_detection_head = heads['first_head']
        self.roi_detection_head = heads['second_head']
        if isinstance(self.roi_detection_head, list):
            for i, head in enumerate(self.roi_detection_head):
                self.__setattr__('roi_detection_head_{}'.format(i), head)
        self.roi_pooler = roi_pooler
        self.targets_converter = targets_converter
        self.inputs_converter = inputs_converter
        self.training = training
        self.feature_names = ['0', '1', '2', '3']
        self.strides = None
        self.second_loss_weight = second_loss_weight
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
            losses_first, human_proposal_out = self.human_detection_head(features, inputs, targets)
            #return losses_first

            if isinstance(self.roi_pooler, list):
                human_proposals = []
                roi_features = []
                targets_for_second = []
                for roi_pooler in self.roi_pooler:
                    human_proposal, roi_feature, targets_temp = roi_pooler(human_proposal_out, feature_second, stride_second, inputs, targets)
                    human_proposals.append(human_proposal)
                    roi_features.append(roi_feature)
                    targets_for_second.append(targets_temp)
            else:
                human_proposals, roi_features, targets_for_second = self.roi_pooler(human_proposal_out, feature_second, stride_second, inputs, targets)

            if self.inputs_converter is not None:
                if torch.is_tensor(roi_features):
                    second_inputs = self.inputs_converter(stride_second, len(roi_features))
                elif isinstance(roi_features, list):
                    second_inputs = [self.inputs_converter(stride_second, len(roi_feature)) for roi_feature in roi_features]
                else:
                    raise ValueError('unsupported roi features type!')
            else:
                second_inputs = inputs
            
            # convert the targets according to the roi size and proposal
            if self.targets_converter is not None:
                # if it has multiple head
                if isinstance(self.roi_detection_head, list):
                    second_targets = [self.targets_converter(human_proposal, targets_temp, stride_second) for human_proposal, targets_temp in zip(human_proposals, targets_for_second)]
                else:
                    second_targets = self.targets_converter(human_proposals, targets_for_second, stride_second)
            else:
                second_targets = targets
            
            if torch.is_tensor(roi_features):
                roi_features = {'0':roi_features}
            elif isinstance(roi_features, list):
                roi_features = [{'0':roi_feature} for roi_feature in roi_features]

            losses_second= {}
            if isinstance(self.roi_detection_head, list):
                for i, (head, human_proposal, roi_feature, second_input, second_target) in enumerate(zip(self.roi_detection_head, human_proposals, roi_features, second_inputs, second_targets)):
                    losses_second_roi = head(human_proposal, roi_feature, stride_second, inputs=second_input, targets=second_target )
                    losses_second_roi = self.update_loss_name(losses_second_roi, str(i+1))
                    losses_second.update(losses_second_roi)
            else:
                losses_second_roi = self.roi_detection_head(human_proposals, roi_features, stride_second, inputs=second_inputs, targets=second_targets )
                losses_second_roi = self.update_loss_name(losses_second_roi, str(1))
                losses_second.update(losses_second_roi)

            losses = {**losses_first, **losses_second}
            return losses
        else:
            human_results = self.human_detection_head(features, inputs, targets=targets)
            if self.test_mode == 'first':
                return human_results
            human_boxes = human_results['boxes']
            human_scores = human_results['scores']
            #human_boxes = [human_box[torch.argmax(human_score)][None,:].clone() for human_box, human_score in zip(human_boxes, human_scores)]
            human_boxes = [human_box[human_score>0.5].clone() if (human_score>0.5).any() else human_box[torch.argmax(human_score)][None,:].clone() for human_box, human_score in zip(human_boxes, human_scores)]
            if self.expand_ratio is not None:
                human_boxes = self.expand_boxes_batch(human_boxes, inputs['image_sizes'], self.expand_ratio)

            if isinstance(self.roi_pooler, list):
                human_proposals = []
                roi_features = []
                targets_for_second = []
                for roi_pooler in self.roi_pooler:
                    human_proposal, roi_feature, targets_temp = roi_pooler(human_boxes, feature_second, stride_second, inputs, targets)
                    human_proposals.append(human_proposal)
                    roi_features.append(roi_feature)
                    targets_for_second.append(targets_temp)
            else:
                human_proposals, roi_features, targets_for_second = self.roi_pooler(human_boxes, feature_second, stride_second, inputs, targets)
            #human_proposal, roi_features, targets_for_second = self.roi_pooler(human_boxes, feature_second, stride_second, inputs, targets)
            if self.inputs_converter is not None:
                if torch.is_tensor(roi_features):
                    second_inputs = self.inputs_converter(stride_second, len(roi_features))
                elif isinstance(roi_features, list):
                    second_inputs = [self.inputs_converter(stride_second, len(roi_feature)) for roi_feature in roi_features]
                else:
                    raise ValueError('unsupported roi features type!')
            else:
                second_inputs = inputs
            
            if torch.is_tensor(roi_features):
                roi_features = {'0':roi_features}
            elif isinstance(roi_features, list):
                roi_features = [{'0':roi_feature} for roi_feature in roi_features]

            if isinstance(self.roi_detection_head, list):
                results = []
                for i, (head, human_proposal,roi_feature, second_input) in enumerate(zip(self.roi_detection_head, human_proposals, roi_features, second_inputs)):
                    result = self.head(human_proposal, roi_feature, stride_second, inputs=second_input, targets=targets)
                    result = self.post_process(result, inputs)
                    results.append(result)
            else:
                results = self.roi_detection_head(human_proposals, roi_features, stride_second, inputs=second_inputs, targets=targets)
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

    @torch.no_grad()
    def expand_boxes_batch(self, boxes_batch, image_sizes, ratio=0.2):
        new_boxes_batch = [expand_boxes(boxes, image_size, ratio) for boxes, image_size in zip(boxes_batch, image_sizes)]
        return new_boxes_batch



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

    def update_loss_name(self, losses, prefix):
        losses_new = {}
        for k,v in losses.items():
            losses_new[prefix+k] = v
        return losses_new


@torch.no_grad()
def expand_boxes(boxes, image_size, ratio=0.2):
    '''
    if image_size is None, no cliping conducted
    '''
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

    if image_size is not None:
        height, width = image_size
        boxes_x1 = boxes_x1.clamp(min=0, max=width)
        boxes_y1 = boxes_y1.clamp(min=0, max=height)
        boxes_x2 = boxes_x2.clamp(min=0, max=width)
        boxes_y2 = boxes_y2.clamp(min=0, max=height)

    return torch.stack([boxes_x1, boxes_y1, boxes_x2, boxes_y2], dim=1)