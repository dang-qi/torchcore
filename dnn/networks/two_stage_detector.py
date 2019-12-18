from torch import nn

class TwoStageDetector(nn.Module):
    '''
    Two stage detector

    The typical model is Faster RCNN
    Arguments:
        backbone (nn.Module): backbone network
        rpn (nn.Module): Region proposal network
        roi_extractor (nn.Module): Extract roi
        bbox_heads (nn.Module): Extract bbox from roi_features
        neck (nn.Module): Can extract feature from different stage like FPN
        cfg: configuration for the detector
    '''
    def __init__(self, backbone, neck = None, rpn=None, roi_extractor=None, bbox_heads=None, cfg=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_extractor = roi_extractor
        self.bbox_heads=bbox_heads
        self.neck = neck

    def forward(self, inputs, targets=None):
        if self.training and targets is None:
            raise ValueError('targets should not be None during the training')
        features = self.backbone(inputs['data'])
        if  self.has_neck():
            features = self.neck(features)
        proposals, proposal_losses = self.rpn(inputs, features, targets)
        #rois = self.roi_extractor(proposals)
        bbox, bbox_losses = self.bbox_heads(features, proposals, inputs['image_sizes'], targets)
        
        losses = {}
        losses.update(proposal_losses)
        losses.update(bbox_losses)

        if self.training:
            return losses

        bbox = self.post_process(inputs, bbox)
        return bbox


    def has_neck(self):
        return self.neck is not None
