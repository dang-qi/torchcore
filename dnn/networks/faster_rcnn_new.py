import torch
from collections import OrderedDict
from .pooling import RoiAliagnFPN
from .general_detector import GeneralDetector
from torchvision.ops import roi_align, nms

class FasterRCNNNew(GeneralDetector):
    def __init__(self, backbone, neck=None, heads=None, cfg=None, training=True):
        super(GeneralDetector, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn = heads['rpn']
        self.roi_head = heads['bbox']
        self.training = training
        

    def forward(self, inputs, targets=None):
        features = self.backbone(inputs['data'])
        #print('feature keys:', features.keys())
        if self.neck is not None:
            features = self.neck(features)
        strides = self.get_strides(inputs, features)

        if self.training:
            proposals, losses_rpn = self.rpn(inputs, features, targets)
        else:
            proposals, scores = self.rpn(inputs, features, targets)

        #for proposal in proposals:
        #    print('proposal shape', proposal.shape)

        if self.training:
            losses_roi = self.roi_head(proposals, features, strides, targets=targets)
            losses = {**losses_rpn, **losses_roi}
            return losses
        else:
            results = self.roi_head(proposals, features, strides, targets=targets)
            results = self.post_process(results, inputs)
            return results

    def post_process(self, results, inputs):
        for i, (boxes, scores, labels) in enumerate(zip(results['boxes'], results['scores'], results['labels'])):
            # perform nms
            keep = nms(boxes, scores, iou_threshold=0.5)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

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
        image_size = inputs['data'].shape[-2:]
        strides = tuple((image_size[0] / g[0] + image_size[1] / g[1])/2 for g in grid_sizes)
        return strides

