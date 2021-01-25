import torch
from torch import nn

from collections import OrderedDict

class RoIFasterRCNNHeads(nn.Module):
    def __init__(self, cfg, rpn, roi_head):
        super(RoIFasterRCNNHeads, self).__init__()
        self.rpn = rpn
        self.roi_head = roi_head
        self.cfg = cfg
        # used for partial network training
        if hasattr(cfg, 'dataset_label'):
            self.dataset_label = cfg.dataset_label
            self.rpn.dataset_label = cfg.dataset_label
            self.roi_head.dataset_label = cfg.dataset_label
        else:
            self.dataset_label = None

    def forward(self, roi_boxes, features, stride, inputs, targets=None):
        strides = [stride]
        if self.training:
            proposals, losses_rpn = self.rpn(inputs, features, targets)
        else:
            proposals, scores = self.rpn(inputs, features, targets)

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
            results = self.post_process(results,roi_boxes, stride)
            return results

    def get_strides(self, inputs, features):
        if isinstance(features, dict):
            features = list(features.values())
        grid_sizes = tuple([feature_map.shape[-2:] for feature_map in features])
        image_size = inputs['data'].shape[-2:]
        strides = tuple((image_size[0] / g[0] + image_size[1] / g[1])/2 for g in grid_sizes)
        return strides

    @torch.no_grad()
    def post_process(self, results, human_boxes_batch, stride):
        roi_per_im = [len(proposal) for proposal in human_boxes_batch]
        roi_boxes = torch.cat(human_boxes_batch, dim=0) # ROI_NUM_N x 4
        roi_w = roi_boxes[:,2] - roi_boxes[:,0] # ROI_NUM_N
        roi_h = roi_boxes[:,3] - roi_boxes[:,1]
        w_scale = (roi_w / self.cfg.roi_pool_w / stride) # ROI_NUM_N 
        h_scale = (roi_h / self.cfg.roi_pool_h / stride)
        boxes_batch = results['boxes'] # list[DET_PER_IM x 4,...]
        for i, boxes in enumerate(boxes_batch):
            boxes[:,0] *= w_scale[i]
            boxes[:,2] *= w_scale[i]
            boxes[:,1] *= h_scale[i]
            boxes[:,3] *= h_scale[i]

            boxes[:,0] += roi_boxes[i,0]
            boxes[:,2] += roi_boxes[i,0]
            boxes[:,1] += roi_boxes[i,1]
            boxes[:,3] += roi_boxes[i,1]

        #results['boxes'] = [torch.cat(boxes[im_num:im_num+i], dim=0) for i, im_num in zip(accumulate_list,human_per_im)]
        scores = []
        labels = []
        boxes = []

        ind = 0
        for human_num in roi_per_im:
            boxes.append(torch.cat(results['boxes'][ind:ind+human_num], dim=0))
            labels.append(torch.cat(results['labels'][ind:ind+human_num], dim=0))
            scores.append(torch.cat(results['scores'][ind:ind+human_num], dim=0))
            ind+=human_num
        results['boxes'] = boxes
        results['labels'] = labels
        results['scores'] = scores
        return results