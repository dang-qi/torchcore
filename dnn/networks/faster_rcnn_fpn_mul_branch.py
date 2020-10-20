import torch
from collections import OrderedDict
from .pooling import RoiAliagnFPN
from .general_detector import GeneralDetector
from torchvision.ops import roi_align, nms

class FasterRCNNFPNMulBranch(GeneralDetector):
    def __init__(self, backbone, neck=None, heads=None, cfg=None, training=True):
        super(GeneralDetector, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn = heads['rpn']
        self.roi_head = heads['bbox']
        self.training = training
        self.cfg = cfg
        

    def forward(self, inputs, targets=None):
        features = self.backbone(inputs['data'])
        print('feature keys:', features.keys())
        print('feature[0] shape', features[0].shape)
        if self.neck is not None:
            features = self.neck(features)
        strides = self.get_strides(inputs, features)
        inputs_human, inputs_garment = self.split_dict(inputs, inputs['dataset_label'], dataset_num=2)
        targets_human, targets_garment = self.split_list(targets, inputs['dataset_label'], dataset_num=2)
        features_human, features_garment = self.split_dict(features, inputs['dataset_label'], dataset_num=2)

        if self.training:
            proposals, losses_rpn = self.rpn(inputs_human, features_human, targets_human)
        else:
            proposals, scores = self.rpn(inputs, features, targets)
        keep_dataset_index = self.get_keep_index(inputs['dataset_label'])

        for proposal in proposals:
            print('proposal shape', proposal.shape)
        return proposals

        #if self.training:
        #    losses_roi = self.roi_head(proposals, features, strides, targets=targets)
        #    losses = {**losses_rpn, **losses_roi}
        #    return losses
        #else:
        #    results = self.roi_head(proposals, features, strides, targets=targets)
        #    results = self.post_process(results, inputs)
        #    return results


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




    def get_strides(self, inputs, features):
        if isinstance(features, dict):
            features = list(features.values())
        grid_sizes = tuple([feature_map.shape[-2:] for feature_map in features])
        image_size = inputs['data'].shape[-2:]
        strides = tuple((image_size[0] / g[0] + image_size[1] / g[1])/2 for g in grid_sizes)
        return strides

