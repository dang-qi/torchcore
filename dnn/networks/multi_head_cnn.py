import torch
import time
import numpy as np
from collections import OrderedDict
from .pooling import RoiAliagnFPN
from .general_detector import GeneralDetector
from torchvision.ops import roi_align, nms

class MultiHeadCNN(torch.nn.Module):
    def __init__(self, backbone, heads, feature_names, neck=None, cfg=None, training=True, loss_weights=None, test_mode=[0], debug_time=False):
        '''
            heads: list[head1, head2...]
            test_mode: int or list[0,1,2...], the index of head that wanted to be tested
        '''
        super(MultiHeadCNN, self).__init__()
        self.backbone = backbone
        self.neck = neck
        #self.rpn = heads['rpn']
        #self.roi_head = heads['bbox']
        self.heads = heads
        self.set_heads(heads)
        self.training = training
        self.feature_names = feature_names
        self.strides = None
        self.loss_weights = loss_weights
        self.set_test_mode(test_mode)
        self.dataset_labels = list(range(len(heads)))

        if debug_time:
            self.total_time = {'feature':0.0, 'rpn':0.0, 'roi_head':0.0}
        self.debug_time = debug_time

    def set_heads(self, heads):
        for i, head in enumerate(heads):
            self.__setattr__('head{}'.format(i+1), head)
        

    def forward(self, inputs, targets=None):
        '''
            get features -> first head -> roi pooler -> second head
            targets should be list if not None
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
            loss_all = {}
            for i, (head, features) in enumerate(zip(self.heads, features_all)):
                indexs = dataset_labels == i
                # if there is no given label
                if not indexs.any():
                    continue

                # select the inputs, targets according to the dataset label 
                for k,v in features.items():
                    features[k] = v[indexs]
                
                head_inputs = self.input_sampling(inputs, indexs)
                head_targets = [target for target, label in zip(targets, indexs) if label]
                

                loss = head(features, head_inputs, head_targets)
                loss = convert_loss_name(loss, i+1)
                loss_all.update(loss)
            return loss_all
        else:
            resutls_all = []
            for i, (head, features) in enumerate(zip(self.heads, features_all)):
                if i not in self.test_mode:
                    continue
                result = head(features, inputs, targets=targets)
                #result = self.post_process(result, inputs)
                resutls_all.append(result)
            if len(resutls_all) == 1:
                return resutls_all[0]
            else:
                return resutls_all

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

    def input_sampling(self, inputs, indexs):
        '''
            indexs: torch.tensor(bool)
        '''
        if isinstance(inputs, dict):
            out_inputs = {}
            for k,v in inputs.items():
                out_inputs[k] = self.variable_sampling(v, indexs)
        elif isinstance(inputs, list):
            out_inputs = self.variable_sampling(inputs, indexs)
        else:
            raise ValueError('Not support inputs type: {}'.format(type(inputs)))
        return out_inputs
    
    def variable_sampling(self, variable, indexs):
        if isinstance(variable, list):
            out_variable = [v for v,ind in zip(variable, indexs) if ind]
        elif torch.is_tensor(variable):
            out_variable = variable[indexs]
        elif isinstance(variable, np.ndarray):
            out_variable = variable[indexs.cpu().numpy()]
        else:
            raise ValueError('Not support sampling for {} variable'.format(type(variable)))
        return out_variable
        
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

    def set_test_mode(self, mode):
        if isinstance(mode, int):
            self.test_mode = [mode]
        elif isinstance(mode, (list, tuple)):
            self.test_mode = mode
        else:
            raise ValueError('Not support test mode type {}'.format(type(mode)))


def convert_loss_name(loss, i):
    out_loss = {}
    for k,v in loss.items():
        out_loss[str(i)+k] = v
    return out_loss
