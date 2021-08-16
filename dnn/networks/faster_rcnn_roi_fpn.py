import torch
import math
import time
from collections import OrderedDict
from .pooling import RoiAliagnFPN
from .general_detector import GeneralDetector
from torchvision.ops import roi_align, nms

class FasterRCNNRoIFPN(GeneralDetector):
    def __init__(self, backbone, base_stride, target_converter_cfg, neck=None, heads=None, cfg=None, training=True, rpn_feature_names=['0','1','2','3', 'pool'], roi_feature_names=['0','1','2','3'],roi_box_key='input_box', debug_time=False, just_rpn=False):
        super(GeneralDetector, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn = heads['rpn']
        self.roi_head = heads['bbox']
        self.training = training
        self.roi_box_key = roi_box_key
        self.rpn_feature_names = rpn_feature_names
        self.roi_feature_names = roi_feature_names
        self.strides = None
        self.just_rpn=just_rpn
        self.base_stride = base_stride
        self.target_converter = RoITargetConverter(target_converter_cfg.roi_pool_w, target_converter_cfg.roi_pool_h, target_converter_cfg.stride, target_converter_cfg.boxes_key, target_converter_cfg.keep_key, target_converter_cfg.mask_key, allow_box_outside=target_converter_cfg.allow_box_outside)

        if debug_time:
            self.total_time = {'feature':0.0, 'rpn':0.0, 'roi_head':0.0}
        self.debug_time = debug_time
        

    def forward(self, inputs, targets=None):
        debug_time = self.debug_time
        if debug_time:
            start = time.time()
        roi_boxes = [target[self.roi_box_key] for target in targets]
        features = self.backbone(inputs['data'], roi_boxes )
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
        
        targets = self.target_converter.convert(roi_boxes, targets)

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
            results = self.target_converter.convert_back(roi_boxes, results)
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
                    boxes /= scale

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
        #image_size = inputs['data'].shape[-2:]
        #strides = tuple((image_size[0] / g[0] + image_size[1] / g[1])/2 for g in grid_sizes)
        strides = tuple(((grid_sizes[0][0] // g[0])+(grid_sizes[0][1] // g[1]))/2 * self.base_stride for g in grid_sizes)
        #strides = tuple(math.pow(2,math.ceil(math.log2((image_size[0] / g[0] + image_size[1] / g[1])/2))) for g in grid_sizes)
        return strides

    def reset_time(self):
        self.total_time = {'feature':0.0, 'rpn':0.0, 'roi_head':0.0}


class RoITargetConverter():
    def __init__(self, roi_pool_w, roi_pool_h, stride, boxes_key='boxes', keep_key=['labels'],mask_key=None, allow_box_outside=False ) -> None:
        self.boxes_key = boxes_key
        self.mask_key = mask_key
        self.keep_key = keep_key
        self.roi_pool_w = roi_pool_w
        self.roi_pool_h = roi_pool_h
        self.stride = stride
        self.allow_box_outside = allow_box_outside
        
    @torch.no_grad()
    def convert(self, roi_boxes_batch, targets):
        targets_new = []
        for roi_boxes, target in zip(roi_boxes_batch, targets):
            for roi_box in roi_boxes:
                roi_w = roi_box[2] - roi_box[0]
                roi_h = roi_box[3] - roi_box[1]

                w_scale = self.roi_pool_w / roi_w * self.stride
                h_scale = self.roi_pool_h / roi_h * self.stride

                boxes = target[self.boxes_key].clone()
                
                if self.allow_box_outside:
                    boxes[:,0] = boxes[:,0] - roi_box[0] 
                    boxes[:,1] = boxes[:,1] - roi_box[1]
                    boxes[:,2] = boxes[:,2] - roi_box[2]
                    boxes[:,3] = boxes[:,3] - roi_box[3]

                    intersection = (torch.minimum(boxes[:,2], torch.full_like(boxes[:,2],roi_w))- torch.maximum(torch.full_like(boxes[:,0],0), boxes[:,0]))\
                        *(torch.minimum(boxes[:,3], torch.full_like(boxes[:,3], roi_h))-torch.maximum(boxes[:,1],torch.full_like(boxes[:,1],0)))
                    keep = intersection > 1

                    boxes[:,0] *= w_scale
                    boxes[:,1] *= h_scale
                    boxes[:,2] *= w_scale
                    boxes[:,3] *= h_scale
                    
                else:
                    boxes[:,0] = torch.clamp((boxes[:,0] - roi_box[0]),0, roi_w) * w_scale
                    boxes[:,1] = torch.clamp((boxes[:,1] - roi_box[1]),0,roi_h) * h_scale
                    boxes[:,2] = torch.clamp((boxes[:,2] - roi_box[0]),0,roi_w) * w_scale
                    boxes[:,3] = torch.clamp((boxes[:,3] - roi_box[1]),0,roi_h) * h_scale

                    keep = (boxes[:,2]>boxes[:,0]) & (boxes[:,3]>boxes[:,1])

                # Warning: This is NOT a deep copy
                new_target = target.copy()
                new_target[self.boxes_key] = boxes[keep]
                for k in self.keep_key:
                    new_target[k] = target[k][keep]

                if self.mask_key is not None:
                    # TODO fix the rest of code
                    masks = target[self.mask_key]

                targets_new.append(new_target)
        return targets_new


    def convert_back(self, roi_boxes_batch, results):
        roi_per_im = [len(roi) for roi in roi_boxes_batch]
        roi_boxes_all = torch.cat(roi_boxes_batch, dim=0)

        # convert boxes
        if 'boxes' in results:
            boxes_batch = results['boxes']
            new_batch = []
            for boxes, roi_box in zip(boxes_batch, roi_boxes_all):
                roi_w = roi_box[2] - roi_box[0]
                roi_h = roi_box[3] - roi_box[1]

                w_scale =  roi_w / self.roi_pool_w /self.stride
                h_scale =  roi_h / self.roi_pool_h /self.stride

                boxes[:,0] = boxes[:,0]*w_scale + roi_box[0]
                boxes[:,1] = boxes[:,1]*h_scale + roi_box[1]
                boxes[:,2] = boxes[:,2]*w_scale + roi_box[0]
                boxes[:,3] = boxes[:,3]*h_scale + roi_box[1]

                #boxes = boxes/self.stride
                new_batch.append(boxes)
            results['boxes'] = new_batch
        results = self.merge_result(results, roi_per_im)
        return results
        
    def merge_result(self, results, roi_per_im):
        new_result = {k:[] for k in results.keys()}
        i = 0
        for roi_num in roi_per_im:
            for key in results.keys():
                result_single = results[key][i:i+roi_num]
                new_result[key].append(torch.cat(result_single, dim=0))
            i += roi_num
        return new_result
        

        
                

        