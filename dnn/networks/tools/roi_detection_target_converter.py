import torch
import numpy as np

class RoIDetectionTargetsConverter():
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, human_proposals_batch, targets, stride):
        targets = self.generate_targets(human_proposals_batch, targets, stride)
        return targets

    @torch.no_grad()
    def generate_targets(self, human_boxes_batch, targets, stride):
        new_targets = []
        for human_boxes, target in zip(human_boxes_batch, targets):
            for human_box in human_boxes:
                boxes = target['boxes']
                boxes = self.normalize_boxes(human_box, boxes, stride)
                #boxes = boxes.detach().cpu().numpy()
                keep = self.find_valid_boxes(boxes)
                boxes  = boxes[keep]
                labels = target['labels'][keep]
                new_target = {}
                new_target['boxes'] = boxes
                new_target['labels'] = labels
                new_targets.append(new_target)
        
        return new_targets


    def find_valid_boxes(self, boxes):
        if torch.is_tensor(boxes):
            keep = torch.logical_and(boxes[:,2] > boxes[:,0], boxes[:,3] > boxes[:,1])
        elif isinstance(boxes, (np.ndarray, np.generic) ):
            keep = np.logical_and(boxes[:,2] > boxes[:,0], boxes[:,3] > boxes[:,1])
        return keep

    def normalize_boxes(self, human_boxes, boxes, stride):
        # clone the boxes and not change the original input boxes
        boxes = boxes.clone().detach()

        boxes[:,0] = boxes[:,0]-human_boxes[0]
        boxes[:,1] = boxes[:,1]-human_boxes[1]
        boxes[:,2] = boxes[:,2]-human_boxes[0]
        boxes[:,3] = boxes[:,3]-human_boxes[1]
        human_width = human_boxes[2] - human_boxes[0]
        human_height = human_boxes[3] - human_boxes[1]
        boxes = self.crop_boxes(boxes, (int(human_height), int(human_width)))

        # normalize the boxes using the roi size
        roi_h = self.cfg.roi_pool_h
        roi_w = self.cfg.roi_pool_w

        boxes[:,0] = boxes[:,0] / human_width * roi_w * stride
        boxes[:,1] = boxes[:,1] / human_height * roi_h * stride
        boxes[:,2] = boxes[:,2] / human_width * roi_w * stride
        boxes[:,3] = boxes[:,3] / human_height * roi_h * stride
        return boxes

    def crop_boxes(self, boxes, image_size):
        # boxes: N * 4 tensor, x1, y1, x2, y2 format
        # image_size: height, width
        height, width = image_size
        boxes[...,0] = boxes[...,0].clamp(min=0, max=width)
        boxes[...,1] = boxes[...,1].clamp(min=0, max=height)
        boxes[...,2] = boxes[...,2].clamp(min=0, max=width)
        boxes[...,3] = boxes[...,3].clamp(min=0, max=height)
        return boxes

class RoIRPNInputConverter():
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, stride, batch_size ):
        '''
            return input_size: (h,w), image_sizes: list[(h,w)...] and batch_size: int
        '''
        inputs = {}
        h = stride * self.cfg.roi_pool_h
        w = stride * self.cfg.roi_pool_w

        inputs['input_size'] = (h, w)
        inputs['image_sizes'] = [(h, w)]*batch_size
        inputs['batch_size'] = batch_size
        return inputs