import torch
from torch import nn

from collections import OrderedDict

class FasterRCNNHeads(nn.Module):
    def __init__(self, cfg, rpn, roi_head, post_process=True):
        super(FasterRCNNHeads, self).__init__()
        self.rpn = rpn
        self.roi_head = roi_head
        self.strides = None
        self._post_process = post_process
        # used for partial network training
        if hasattr(cfg, 'dataset_label'):
            self.dataset_label = cfg.dataset_label
            self.rpn.dataset_label = cfg.dataset_label
            self.roi_head.dataset_label = cfg.dataset_label
        else:
            self.dataset_label = None

    def forward(self, features, inputs, targets=None):
        if self.training:
            proposals, losses_rpn = self.rpn(inputs, features, targets)
        else:
            proposals, scores = self.rpn(inputs, features, targets)

        if self.strides is None:
            strides = self.get_strides(inputs, features)
        else:
            strides = self.strides

        if self.training:
            if self.dataset_label is None:
                losses_roi = self.roi_head(proposals, features, strides, targets=targets)
                losses = {**losses_rpn, **losses_roi}
                return losses
            else:
                losses_roi, human_proposals = self.roi_head(proposals, features, strides, targets=targets, inputs=inputs)
                losses = {**losses_rpn, **losses_roi}
                return losses, human_proposals

        else:
            results = self.roi_head(proposals, features, strides, targets=targets)
            if self._post_process:
                results = self.post_process(results, inputs)
            return results

    def get_strides(self, inputs, features):
        if isinstance(features, dict):
            features = list(features.values())
        grid_sizes = tuple([feature_map.shape[-2:] for feature_map in features])
        image_size = inputs['data'].shape[-2:]
        strides = tuple((image_size[0] / g[0] + image_size[1] / g[1])/2 for g in grid_sizes)
        return strides

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