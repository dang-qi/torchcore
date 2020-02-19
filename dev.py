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
from dnn.networks.losses import FocalLossHeatmap, L1LossWithInd

def post_process_test(dataset, model):
    for inputs, targets in dataset:
        out = model.postprocess(targets, inputs)
        print(out['offset'][:,:,:10])
        print(targets['offset'][:,:,:10])
        break
        

def test_loss():
    fcloss= FocalLossHeatmap()
    pred = torch.rand((3,3))
    pred1 = pred.clone()
    gt = torch.rand((3,3))
    gt[0,1] = 1
    gt1 = gt.clone()

    loss1 = fcloss._forword_impl(pred1, gt1)
    loss2 = _neg_loss(pred, gt)
    print(loss1*2)
    print(loss2)

def test_l1_loss(dataset):
    f1loss = L1LossWithInd()
    for inputs, targets in dataset:
        pre = targets['width_height_map']
        #pre = torch.rand_like(pre)
        ind = targets['ind']
        ind_mask = targets['ind_mask']
        gt = targets['width_height']
        loss = f1loss.forward(pre, ind, ind_mask, gt)
        #print(ind_mask.sum())
        print(loss)
        #print(len(gt.nonzero()))
        break

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
    model_path = 'checkpoints_20200217_centernet_24.pkl'
    #model_path = 'checkpoints_20200216_frcnn_human_416_15.pkl'
    backbone = networks.feature.resnet50()
    in_channel = backbone.out_channel
    neck = networks.neck['upsample_basic'](in_channel)
    #net = resnet50()
    backbone.multi_feature = False
    loss_parts = ['heatmap', 'offset', 'width_height']
    model = CenterNet(backbone, 1, neck=neck, parts=loss_parts)
    #device = torch.device('cpu')
    #state_dict_ = torch.load(model_path, map_location=device)['model_state_dict']
    #state_dict = {}
    #for k in state_dict_:
    #    if k.startswith('module') and not k.startswith('module_list'):
    #        state_dict[k[7:]] = state_dict_[k]
    #    else:
    #        state_dict[k] = state_dict_[k]
    #model.load_state_dict(state_dict, strict=True )
    return model


def loss_test(data_loader, model):
    #data_loader = get_data_loader()
    #model = get_model()
    for inputs, targets in data_loader:
        loss = model(inputs, targets)
        print(loss)
        break

def inference_test(dataset, model):
    model.eval()
    i=0
    for inputs, targets in dataset:
        pred = model(inputs, targets)
        ori_image = inputs['cropped_im'][0].numpy()
        print(ori_image.shape)
        im = Image.fromarray(np.uint8(ori_image))
        pred['heatmap'] = torch.clamp(pred['heatmap'].sigmoid_(), min=1e-4, max=1-1e-4)
        heatmap = pred['heatmap'][0][0].detach().numpy()
        heatmap_im = Image.fromarray((heatmap*255).astype(np.uint8)).convert('RGB')
        heatmap_im = heatmap_im.resize(im.size)
        #heatmap = pred['heatmap'][0].detach().numpy()
        #print(heatmap)
        #visulize_heatmaps_with_image(heatmap, im)
        heatmap_im.show()
        im.show()
        #print(ori_image[0].size())
        #for k, v in pred.items():
        #    print('{}:{}'.format(k, v.shape))
        break
        i+=1
        if i==5:
            break


def dataset_test1(dataset):
    i=0
    for inputs, targets in dataset:
        im = inputs['data']
        print(im.shape)
        i+=1
        if i==5:
            break


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss