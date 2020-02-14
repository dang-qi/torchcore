import torch
import sys
sys.path.append('torchcore')

from torchcore.dnn import networks
from torchcore.dnn.networks.center_net import CenterNet 
from torchvision.models import resnet50
from torchcore.data.datasets import COCOPersonCenterDataset

def dataset_test():
    anno_path = '/ssd/data/annotations/coco2017_instances_person.pkl'
    root = '/ssd/data/datasets/COCO'
    dataset = COCOPersonCenterDataset(root=root, anno=anno_path, part='train2017',transforms=transforms)

def network_test():
    a = torch.randn(2, 3, 224, 224)
    inputs = {}
    inputs['data'] = a
    backbone = networks.feature.resnet50()
    in_channel = backbone.out_channel
    neck = networks.neck['upsample_basic'](in_channel)
    #net = resnet50()
    backbone.multi_feature = False
    model = CenterNet(backbone, 13, neck=neck)
    model.eval()
    c={}
    b=model(inputs)
    #print(b.keys())
    for key, item in b.items():
        print('{}:{}'.format(key, item.shape))