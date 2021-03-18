from torch import nn

class FoveaNet(nn.Module):
    def __init__(self, backbone, pred_heads, neck=None):
        super(FoveaNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.pred_heads = pred_heads

    def forward(self, inputs, targets=None):
        if self.training and targets is None:
            raise ValueError('targets should not be None during the training')

        features = self.backbone(inputs['data'])
        if self.neck is not None:
            features = self.neck(features)
        pred = self.pred_heads(features, inputs,targets)

        if self.training:
            return pred
        
        output = self.post_process(pred, inputs)
        return output

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
