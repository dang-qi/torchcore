from torch import nn

from .base_detector import BaseDetector

from ..necks.build import build_neck
from ..backbone.build import build_backbone
from .build import build_detector, DETECTOR_REG
from ..roi_heads.build import build_roi_head
from ..detection_heads import build_detection_head


@DETECTOR_REG.register()
class OneStageDetector(BaseDetector):
    '''
    Two stage detector

    Arguments:
        backbone (config dict): backbone network
        rpn (config dict): Region proposal network
        roi_heads (config dict): roi head
        neck (config dict): Can extract feature from different stage like FPN
    '''
    def __init__(self, backbone, neck=None, det_head=None, training=True, init_cfg=None ):
        super(OneStageDetector, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        self.det_head = build_detection_head(det_head)
        self.training = training

    def forward(self, inputs, targets=None):
        if self.training and targets is None:
            raise ValueError('targets should not be None during the training')
        features = self.backbone(inputs['data'])
        if  self.has_neck():
            features = self.neck(features)

        if self.training:
            losses = self.det_head(inputs=inputs,features=features, targets=targets)
            return losses
        else:
            results = self.det_head(inputs=inputs, features=features, targets=targets)
            results = self.post_process(results, inputs)
            return results

    def set_training_flag(self,flag):
        self.training=flag
