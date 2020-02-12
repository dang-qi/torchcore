from torch import nn

class OneStageDetector(nn.Module):
    def __init__(self, backbone, pred_heads, losses, neck=None):
        super(OneStageDetector, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.pred_heads = pred_heads
        self.losses = losses

    def forward(self, inputs, targets=None):
        if self.training and targets is None:
            raise ValueError('targets should not be None during the training')

        features = self.backbone(inputs['data'])
        if self.neck is not None:
            features = self.neck(features)
        pred = self.pred_heads(features)

        if self.training:
            losses = self.losses(pred, targets)

        if self.training:
            return losses
        
        output = self.postprocess(pred, inputs)
        return output

    def postprocess(self, pred, inputs):
        raise NotImplementedError
