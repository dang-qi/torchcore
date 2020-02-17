import torch
import sys
import os
import random
sys.path.append('torchcore')
from PIL import Image
import numpy as np
import torch

from dnn import networks
from dnn.networks.center_net import CenterNet 
from torchvision.models import resnet50
from torchvision.transforms import ToPILImage
from data.datasets import COCOPersonCenterDataset
from data.transforms import Compose, RandomCrop, RandomScale, RandomMirror, ToTensor, Normalize
from tools.visulize_tools import draw_single_box, visulize_heatmaps_with_image
from dnn.networks.losses import FocalLossHeatmap

def get_dataset():
    anno_path = os.path.expanduser('~/Vision/data/annotations/coco2014_instances_person_debug.pkl')
    root = os.path.expanduser('~/Vision/data/datasets/COCO')
    transform_list = []
    random_crop = RandomCrop((512,448)) #(width, height)
    #random_crop = RandomCrop(512)
    random_scale = RandomScale(0.6, 1.4)
    random_mirror = RandomMirror()
    to_tensor= ToTensor()
    normalize = Normalize()
    transform_list.append(random_scale)
    transform_list.append(random_crop)
    transform_list.append(random_mirror)
    #transform_list.append(to_tensor)
    #transform_list.append(normalize)
    transforms = Compose(transform_list)
    dataset = COCOPersonCenterDataset(root=root, anno=anno_path, part='val2014',transforms=transforms)
    return dataset

def get_data_loader():
    anno_path = os.path.expanduser('~/Vision/data/annotations/coco2014_instances_person_debug.pkl')
    root = os.path.expanduser('~/Vision/data/datasets/COCO')
    transform_list = []
    random_crop = RandomCrop((512,448)) #(width, height)
    #random_crop = RandomCrop(512)
    random_scale = RandomScale(0.6, 1.4)
    random_mirror = RandomMirror()
    to_tensor= ToTensor()
    normalize = Normalize()
    transform_list.append(random_scale)
    transform_list.append(random_crop)
    transform_list.append(random_mirror)
    #transform_list.append(to_tensor)
    #transform_list.append(normalize)
    transforms = Compose(transform_list)
    dataset = COCOPersonCenterDataset(root=root, anno=anno_path, part='val2014',transforms=transforms)

    data_loader = torch.utils.data.DataLoader(
      dataset, 
      batch_size=2, 
      shuffle=True,
      num_workers=2,
      pin_memory=True,
      drop_last=True
    )
    return data_loader

def dataset_test():
    anno_path = os.path.expanduser('~/Vision/data/annotations/coco2014_instances_person_debug.pkl')
    root = os.path.expanduser('~/Vision/data/datasets/COCO')
    transform_list = []
    random_crop = RandomCrop((256,224)) #(width, height)
    #random_crop = RandomCrop(512)
    random_scale = RandomScale(0.6, 1.4)
    random_mirror = RandomMirror()
    to_tensor= ToTensor()
    normalize = Normalize()
    transform_list.append(random_scale)
    transform_list.append(random_crop)
    transform_list.append(random_mirror)
    #transform_list.append(to_tensor)
    #transform_list.append(normalize)
    transforms = Compose(transform_list)
    dataset = COCOPersonCenterDataset(root=root, anno=anno_path, part='val2014',transforms=transforms)
    index = random.randint(0, 99)
    #index = 1
    inputs, targets = dataset[index]
    #im = ToPILImage()(inputs['data'])
    im = inputs['data']
    heatmap = targets['heatmap']
    #im.show()
    #heatmap_im = np.amax(heatmap,axis=0)
    #heatmap_im = Image.fromarray((heatmap_im*255).astype(np.uint8)).convert('RGB')
    #heatmap_im = heatmap_im.resize(im.size)
    #boxes = targets['boxes']
    #labels = targets['cat_labels']
    #for box in boxes:
    #    draw_single_box(im, box)
    #heatmap_im.show()
    
    #visulize_heatmaps_with_image(heatmap, im)
    #print(im.size)
    data_loader = torch.utils.data.DataLoader(
      dataset, 
      batch_size=4, 
      shuffle=True,
      num_workers=2,
      pin_memory=True,
      drop_last=True
    )
    for inputs,targets in data_loader:
        print(inputs.keys())
        print(targets.keys())
        for k, v in inputs.items():
            print('{}:{}'.format(k, v.shape))
        for k, v in targets.items():
            print('{}:{}'.format(k, v.shape))
        break
    #im.show()

def network_test():
    a = torch.randn(2, 3, 300, 400)
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

def get_model():
    backbone = networks.feature.resnet50()
    in_channel = backbone.out_channel
    neck = networks.neck['upsample_basic'](in_channel)
    #net = resnet50()
    backbone.multi_feature = False
    loss_parts = ['heatmap', 'offset', 'width_height']
    model = CenterNet(backbone, 1, neck=neck, loss_parts=loss_parts)
    return model


def loss_test(data_loader, model):
    #data_loader = get_data_loader()
    #model = get_model()
    for inputs, targets in data_loader:
        loss = model(inputs, targets)
        print(loss)
        break

def dataset_test1(dataset):
    i=0
    for inputs, targets in dataset:
        im = inputs['data']
        print(im.shape)
        i+=1
        if i==5:
            break

