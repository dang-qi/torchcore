import torch
from torch import nn
from torch.nn import functional as F
from .heads import ClassificationHead

class ClassificationNet(nn.Module):
    def __init__(self, backbone, input_size, class_num):
        super().__init__()
        self.backbone = backbone
        self.insize = input_size
        self.head = ClassificationHead(class_num=class_num, in_channel=backbone.out_channel, input_size=input_size, head_conv_channel=256)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets=None):
        x = self.backbone(inputs['data'])
        x = self.head(x)
        x = x.view(x.size(0), x.size(1))
        if self.training:
            category_loss = self.loss(x, targets['labels'])
            loss = {}
            loss['category'] = category_loss
            return loss
        else:
            out = self.post_process(x, inputs)
            return out

    def post_process(self, x, inputs):
        scores = F.softmax(x, dim=1)
        score, pred = torch.max(scores, dim=1)
        result = {}
        result['score'] = score
        result['pred'] = pred
        return result 
