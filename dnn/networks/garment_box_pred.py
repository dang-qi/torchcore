import torch
from torch.nn import Module
from torchcore.dnn.networks.tools import BBoxesCoder

class GarmentBoxPredNet(Module):
    def __init__(self, roi_pooler, head, targets_converter=None, dataset_label=None, feature_name=None, middle_layer=False ) -> None:
        super().__init__()
        if targets_converter is None:
            self.targets_converter = BBoxesCoder()
        else:
            self.targets_converter = targets_converter

        self.head = head
        self.loss = torch.nn.SmoothL1Loss(reduction='sum', beta= 1.0 / 9)
        self.roi_pooler = roi_pooler
        self.dataset_label = dataset_label
        self.middle_layer = middle_layer
        self.feature_name = feature_name

    def forward(self, features, person_proposal, stride, inputs=None, targets=None):
        ## get the features with tensor format 
        #if isinstance(features, dict):
        #    if self.feature_name is not None:
        #        features = features[self.feature_name]
        #    elif len(features)==1:
        #        name = list(features.keys())[0]
        #        features = features[name]
        #    else:
        #        raise ValueError('Please setup feature name')

        if self.dataset_label is not None and self.training:
            ind = inputs['dataset_label'] == self.dataset_label
            if isinstance(features, dict):
                for k,v in features.items():
                    features[k] = v[ind]
            else:
                raise ValueError('not support the feature type')
            targets = [target for target, i in zip(targets,ind) if i ]
        
        if self.training:
            assert len(targets) == len(person_proposal)
            person_proposal = self.filter_proposals(targets, person_proposal)
            person_proposal = self.add_input_box_to_human_proposal(targets, person_proposal)
            #print(person_proposal)
            #print(targets)

        rois = self.roi_pooler(features, person_proposal, stride)
        pred = self.head(rois)

        if self.training:
            target_boxes = [target['target_box'] for target in targets]
            targets_code = self.targets_converter.encode(person_proposal, target_boxes)
            box_num = len(pred)
            targets_code = torch.cat(targets_code, dim=0)
            loss = self.loss(pred, targets_code) / box_num
            if self.middle_layer:
                target_box = self.targets_converter.decode(pred, person_proposal)
                num_per_im = [len(prop) for prop in person_proposal]
                outfit_boxes = torch.split(target_box, num_per_im)
                return {'outfit_box_loss': loss}, outfit_boxes
            else:
                return {'outfit_box_loss': loss}
        else:
            target_box = self.targets_converter.decode(pred, person_proposal)
            num_per_im = [len(prop) for prop in person_proposal]
            target_boxes = torch.split(target_box, num_per_im)
            if self.middle_layer:
                return target_boxes
            else:
                result = {}
                result['target_box'] = target_boxes
                return result

    def compute_loss(self, pred, target_boxes, person_proposal):
            targets_code = self.targets_converter.encode(person_proposal, target_boxes)
            box_num = len(pred)
            targets_code = torch.cat(targets_code, dim=0)
            loss = self.loss(pred, targets_code) / box_num
            return {'outfit_box_loss': loss}

        
        


    def filter_proposals(self, targets, person_proposal, thresh=0.3):
        person_proposal_out = []
        for target, persons in zip(targets, person_proposal):
            input_box = target['input_box']
            ious = self.cal_iou(input_box, persons)
            keep = ious >= thresh
            person_proposal_out.append(persons[keep])
        return person_proposal_out
            

    def cal_iou(self,target_box, boxes):
        x1 = torch.maximum(boxes[:,0], target_box[0])
        y1 = torch.maximum(boxes[:,1], target_box[1])
        x2 = torch.minimum(boxes[:,2], target_box[2])
        y2 = torch.minimum(boxes[:,3], target_box[3])
        w = (x2-x1).clamp_min(0)
        h = (y2-y1).clamp_min(0)
        inter = w*h
        box_area1 = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
        box_area2 = (target_box[2]-target_box[0]) * (target_box[3]-target_box[1])
        union = box_area1 + box_area2 - inter
        return inter / union

    
    def add_input_box_to_human_proposal(self, targets, person_proposal):
        person_proposal_out = []
        for target, person_box in zip(targets, person_proposal):
            boxes = torch.cat((target['input_box'].unsqueeze(0), person_box), dim=0)
            person_proposal_out.append(boxes)
        return person_proposal_out

        