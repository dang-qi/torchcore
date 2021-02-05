import torch
import time
from collections import OrderedDict
from .pooling import RoiAliagnFPN
from .general_detector import GeneralDetector
from torchvision.ops import roi_align, nms

class MultiHeadCNN(torch.nn.Module):
    def __init__(self, backbone, heads, neck=None, cfg=None, training=True, loss_weights=None, test_mode='both', debug_time=False):
        '''
            heads: list[head1, head2...]
        '''
        super(MultiHeadCNN, self).__init__()
        self.backbone = backbone
        self.neck = neck
        #self.rpn = heads['rpn']
        #self.roi_head = heads['bbox']
        self.heads = heads
        self.training = training
        self.feature_names = [['0', '1', '2', '3'], ['0','1','2','3']]
        self.strides = None
        self.loss_weights = loss_weights
        self.test_mode = test_mode
        self.dataset_labels = list(range(len(heads)))

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

        features_all = []
        for feature_name in self.feature_names:
            features_new = OrderedDict()
            for k in feature_name:
                features_new[k] = features[k]
            features = features_new
            features_all.append(features)

        #self.set_strides(inputs, features)
        #feature_second = features['0']
        #stride_second = self.strides[0]

        if self.training:
            dataset_labels = inputs['dataset_label']
            for i, head, features in enumerate(zip(self.heads, features_all)):
                indexs = dataset_labels == i
                # if there is no given label
                if not indexs.any():
                    continue

                # select the inputs, targets according to the dataset label 
                for k,v in features:
                    features[k] = v[indexs]

                loss = head(features, inputs, targets)
            losses_first, human_proposal = self.human_detection_head(features, inputs, targets)
            #return losses_first

            human_proposal, roi_features, targets_for_second = self.roi_pooler(human_proposal, feature_second, stride_second, inputs, targets)
            if self.inputs_converter is not None:
                second_inputs = self.inputs_converter(stride_second, len(roi_features))
            else:
                second_inputs = inputs
            
            # convert the targets according to the roi size and proposal
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
            if self.test_mode == 'first':
                return human_results
            human_boxes = human_results['boxes']
            human_scores = human_results['scores']
            #human_boxes = [human_box[torch.argmax(human_score)][None,:].clone() for human_box, human_score in zip(human_boxes, human_scores)]
            human_boxes = [human_box[human_score>0.5].clone() if (human_score>0.5).any() else human_box[torch.argmax(human_score)][None,:].clone() for human_box, human_score in zip(human_boxes, human_scores)]
            if self.expand_ratio is not None:
                human_boxes = self.expand_boxes_batch(human_boxes, inputs['image_sizes'], self.expand_ratio)
            human_proposal, roi_features, targets_for_second = self.roi_pooler(human_boxes, feature_second, stride_second, inputs, targets)

            if self.inputs_converter is not None:
                second_inputs = self.inputs_converter(stride_second, len(roi_features))
            else:
                second_inputs = inputs

            roi_features = {'0':roi_features}
            results = self.roi_detection_head(human_proposal, roi_features, stride_second, inputs=second_inputs, targets=targets)
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
        strides = tuple((image_size[0] / g[0] + image_size[1] / g[1])/2 for g in grid_sizes)
        return strides

    def set_strides(self, inputs, features):
        if self.strides is None:
            self.strides = self.get_strides(inputs, features)

    def reset_time(self):
        self.total_time = {'feature':0.0, 'rpn':0.0, 'roi_head':0.0}

