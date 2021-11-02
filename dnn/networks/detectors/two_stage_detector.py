from torch import nn

from .base_detector import BaseDetector

from ..necks.build import build_neck
from ..backbone.build import build_backbone
from .build import build_detector, DETECTOR_REG
from ..roi_heads.build import build_roi_head


@DETECTOR_REG.register()
class TwoStageDetector(BaseDetector):
    '''
    Two stage detector

    The typical model is Faster RCNN
    Arguments:
        backbone (config dict): backbone network
        rpn (config dict): Region proposal network
        roi_heads (config dict): roi head
        neck (config dict): Can extract feature from different stage like FPN
    '''
    def __init__(self, backbone, neck=None, rpn=None, roi_head=None, training=True ):
        super(TwoStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        self.rpn = build_detector(rpn)
        self.roi_head = build_roi_head(roi_head)
        self.training = training

    def forward(self, inputs, targets=None):
        if self.training and targets is None:
            raise ValueError('targets should not be None during the training')
        features = self.backbone(inputs['data'])
        if  self.has_neck():
            features = self.neck(features)

        if self.training:
            proposals, losses_rpn = self.rpn(inputs, features, targets)
        else:
            proposals, scores = self.rpn(inputs, features, targets)

        if self.training:
            losses_roi = self.roi_head(proposals, features, targets=targets)
            losses = {**losses_rpn, **losses_roi}
            return losses
        else:
            results = self.roi_head(proposals, features, targets=targets)
            results = self.post_process(results, inputs)
            return results

    def set_training_flag(self,flag):
        self.training=flag
