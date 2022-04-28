from PIL import Image
import collections
import torch
import math
import random
import numpy as np
import pickle
import cv2
from . import transform_func as F

from .build import TRANSFORM_REG, build_transform

Iterable = collections.abc.Iterable
Sequence = collections.abc.Sequence

__all__ = ['Compose', 'Resize','ResizeAndPadding', 'ResizeMinMax',
            'ResizeMinMaxTV', 'ToTensor', 'ImageToTensor','Normalize', 'GroupPadding',
            'GeneralRCNNTransform', 'GeneralRCNNTransformTV', 'RandomMirror', 'RandomCrop', 'RandomScale', 'RandomAbsoluteScale', 'PadNumpyArray', 'AddSurrandingBox','AddPersonBox', 'GroupPaddingWithBBox', 'HSVColorJittering','Mosaic','RandomAffine','MixUp']

#def find_inside_bboxes(boxes, h, w):
#        return (boxes[:,0]<w) & (boxes[:,2]>0) &\
#            (boxes[:,1]<h) & (boxes[:,3]>0)
def find_inside_bboxes(bboxes, img_h, img_w):
    """Find bboxes as long as a part of bboxes is inside the image.

    Args:
        bboxes (Tensor): Shape (N, 4).
        img_h (int): Image height.
        img_w (int): Image width.

    Returns:
        Tensor: Index of the remaining bboxes.
    """
    inside_inds = (bboxes[:, 0] < img_w) & (bboxes[:, 2] > 0) \
        & (bboxes[:, 1] < img_h) & (bboxes[:, 3] > 0)
    return inside_inds

@TRANSFORM_REG.register(force=True)
class Compose(object):
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for trans in transforms:
            if isinstance(trans, dict):
                self.transforms.append(build_transform(trans))
            elif callable(trans):
                self.transforms.append(trans)
            else:
                raise TypeError('transform must be callable or dict')

    def __call__(self, inputs, targets=None):
        for transform in self.transforms:
            inputs, targets = transform(inputs, targets)
        return inputs, targets

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

@TRANSFORM_REG.register()
class HSVColorJittering():
    def __init__(self,h_range=5, s_range=30, v_range=30):
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, inputs, targets=None):
        inputs['data'] = F.hsv_color_jittering(inputs['data'], self.h_range, self.s_range, self.v_range)

        return inputs, targets

@TRANSFORM_REG.register()
class Resize(object):
    def __init__(self, size, interplotation=Image.BILINEAR, smaller_edge=None):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size)==2)
        self.size = size
        self.interplotation = interplotation
        self.smaller_edge = smaller_edge

    def __call__(self, inputs, targets=None):
        #if 'data' in inputs:
        inputs['data'], self.scale = F.resize(inputs['data'], self.size, self.interplotation, smaller_edge=self.smaller_edge)

        if targets is None:
            return inputs, targets

        if 'boxes' in targets:
            targets['boxes'] = F.resize_boxes(targets['boxes'], self.scale)

        return inputs, targets
        # TODO add mask, keypoints and other things

@TRANSFORM_REG.register()
class ResizeMax(object):
    '''Resize the image to fit into max_size,
       max_size can be int or (h, w)
    '''
    def __init__(self, max_size, interplotation=Image.BILINEAR):
        assert isinstance(max_size, int) or (isinstance(max_size, Iterable) and len(max_size)==2)
        self.max_size = max_size
        self.interplotation = interplotation

    def __call__(self, inputs, targets=None):
        #if 'data' in inputs:
        inputs['data'], self.scale = F.resize_max(inputs['data'], self.max_size, self.interplotation)

        if targets is None:
            return inputs, targets

        if 'boxes' in targets:
            targets['boxes'] = F.resize_boxes(targets['boxes'], self.scale)

        return inputs, targets
        # TODO add mask, keypoints and other things

@TRANSFORM_REG.register()
class ResizeAndPadding(object):
    def __init__(self, size, interplotation=Image.BILINEAR):
        assert isinstance(size, int)
        self.size = size
        self.interplotation = interplotation

    def __call__(self, inputs, targets=None):
        inputs['data'], self.scale, self.padding = F.resize_and_pad(inputs['data'], self.size, self.interplotation)
        if targets is None:
            return inputs, targets

        if 'boxes' in targets:
            targets['boxes'] = F.resize_and_pad_boxes(targets['boxes'], self.scale, self.padding)
        return inputs, targets

@TRANSFORM_REG.register()
class ResizeMinMax(object):
    '''Resize and Pandding function for Rcnn'''
    def __init__(self, min_size, max_size, interplotation=Image.BILINEAR):
        assert isinstance(min_size, int)
        assert isinstance(max_size, int)
        assert max_size >= min_size
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, inputs, targets=None):
        inputs['data'], self.scale = F.resize_min_max(inputs['data'], self.min_size, self.max_size)
        inputs['scale'] = self.scale
        
        if targets is not None:
            if 'boxes' in targets:
                targets['boxes'] = F.resize_boxes(targets['boxes'], self.scale)
        return inputs, targets

# Torchvision similar version Min Max Resize
@TRANSFORM_REG.register()
class ResizeMinMaxTV(object):
    '''Resize and Pandding function for Rcnn'''
    def __init__(self, min_size, max_size, interplotation=Image.BILINEAR):
        assert isinstance(min_size, int)
        assert isinstance(max_size, int)
        assert max_size >= min_size
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, inputs, targets=None):
        inputs['data'], scale = F.resize_tensor_min_max(inputs['data'], self.min_size, self.max_size)
        if 'scale' not in inputs:
            inputs['scale'] = scale
        else:
            if isinstance(scale, tuple) and isinstance(inputs['scale'], tuple):
                inputs['scale'] = (scale[0]*inputs['scale'][0], scale[1]*inputs['scale'][1])
            elif isinstance(scale, tuple) and isinstance(inputs['scale'], float):
                inputs['scale'] = (scale[0]*inputs['scale'], scale[1]*inputs['scale'])
            elif isinstance(scale, float) and isinstance(inputs['scale'], tuple):
                inputs['scale'] = (scale*inputs['scale'][0], scale*inputs['scale'][1])
            else:
                inputs['scale'] = scale * inputs['scale']
        #inputs['scale'] = self.scale
        
        if targets is not None:
            if 'boxes' in targets:
                targets['boxes'] = F.resize_boxes(targets['boxes'], scale)
        return inputs, targets

@TRANSFORM_REG.register()
class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, inputs, targets=None):
        inputs['data'] = F.to_tensor(inputs['data'])

        if targets is not None:
            for k,v in targets.items():
                if isinstance(v, np.ndarray):
                    targets[k] = torch.from_numpy(targets[k])
            #if 'boxes' in targets:
            #    targets['boxes'] = torch.from_numpy(targets['boxes'])
            #if 'labels' in targets:
            #    targets['labels'] = torch.from_numpy(targets['labels'])
            #if 'cat_labels' in targets:
            #    targets['labels'] = torch.from_numpy(targets['cat_labels'])

        return inputs, targets

@TRANSFORM_REG.register()
class Normalize(object):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean=[0.485, 0.456, 0.406]
        if std is None:
            std=[0.229, 0.224, 0.225]
        self.mean = mean
        self.std = std
    
    def __call__(self, inputs, targets=None):
        inputs['data'] = F.normalize(inputs['data'], self.mean, self.std)

        return inputs, targets

@TRANSFORM_REG.register()
class NormalizeImage(object):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean=[123.675, 116.28, 103.53]
        if std is None:
            std=[58.395, 57.12, 57.375]
        self.mean = mean
        self.std = std
    
    def __call__(self, inputs, targets=None):
        inputs['data'] = F.normalize_image(inputs['data'], self.mean, self.std)

        return inputs, targets

@TRANSFORM_REG.register()
class BatchStack(object):
    def __init__(self):
        pass

    def __call__(self, inputs, targets=None):
        '''here inputs is a list of inputs and targets are lists of targets'''
        pass

@TRANSFORM_REG.register()
class Padding(object):
    ''' Pad the image to the given size,
        The width and height of the original image shouldn't
        be larger than the width and height
        mode: ['constant', 'edge', 'reflect', 'symmetric']
    '''
    def __init__(self, size=None, padding=None, pad_value=0, mode='constant', use_pillow_img=True):
        self.size = size  # (h, w)
        self.padding = padding
        self.pad_value=pad_value
        self.mode = mode
        self.use_pillow_img=use_pillow_img

    def __call__(self, inputs, targets=None):
        inputs['data']= F.pad(inputs['data'],
                              shape=self.size,
                              padding=self.padding,
                              pad_val=self.pad_value,
                              padding_mode='constant',
                              use_pillow_img=self.use_pillow_img)
        return inputs, targets

@TRANSFORM_REG.register()
class GroupPadding(object):
    ''' Padding for group of images tensors
    '''
    def __init__(self, max_width, max_height, size_devidable=32, pad_value=0):
        self.width = int(math.ceil(float(max_width) / size_devidable)*size_devidable)
        self.height = int(math.ceil(float(max_height) / size_devidable)*size_devidable)
        self.pad_value=pad_value

    def __call__(self, images):
        '''
        Parameters:
            images(list[tensors]): input images for group padding
        '''
        images  = F.group_padding(images, self.width, self.height,self.pad_value)
        return images

@TRANSFORM_REG.register(force=True)
class GeneralRCNNTransformMMdet(object):
    '''adapt to mm detection version'''
    def __init__(self, min_size, max_size, image_mean=None, image_std=None, resized=False):
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.resize_min_max = ResizeMinMax(min_size, max_size)
        self.normalize = NormalizeImage(mean=image_mean, std=image_std)
        self.to_tensor = ToTensor()
        self.resized = resized
        #self.device = device
        #self.transforms = Compose(self.resize_min_max, self.to_tensor)

    def __call__(self, inputs, targets=None):
        '''
        Arguments:
            inputs(list[dict{'data':PIL.Image,...}])
            targets(list[dict{'boxes':x1y1x2y2, 'cat_labels':}])
        '''
        images = []
        image_path = []
        dataset_label = []
        #scales = np.zeros(len(inputs))
        scales = []
        if targets is None:
            targets=[None]*len(inputs)

        for i, (ainput, target) in enumerate(zip(inputs, targets)):
            # this is operated in
            if not self.resized:
                ainput, target = self.resize_min_max(ainput, target)

            # normalize after resize, which might be slower 
            # set the tensor to device before normalize
            #ainput['data'].to(self.device)
            #scales[i] = ainput['scale']
            scales.append(ainput['scale'])
            ainput, target = self.normalize(ainput, target)
            ainput, target = self.to_tensor(ainput, target)
            images.append(ainput['data'])

            if 'path' in ainput:
                image_path.append(ainput['path'])
            if 'dataset_label' in ainput:
                dataset_label.append(ainput['dataset_label'])

        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        _, height, width = max_size

        image_sizes = [img.shape[-2:] for img in images]
        self.group_padding = GroupPadding(width, height, size_devidable=32)
        im_tensor = self.group_padding(images)

        inputs = {}
        inputs['data'] = im_tensor
        inputs['scale'] = scales
        inputs['image_sizes'] = image_sizes
        if len(image_path) > 0:
            inputs['path'] = image_path
        if len(dataset_label) > 0:
            inputs['dataset_label'] = torch.tensor(dataset_label)
        return inputs, targets

@TRANSFORM_REG.register()
class GeneralRCNNTransform(object):
    def __init__(self, min_size, max_size, image_mean=None, image_std=None, resized=False):
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.resize_min_max = ResizeMinMax(min_size, max_size)
        self.normalize = Normalize(mean=image_mean, std=image_std)
        self.to_tensor = ToTensor()
        self.resized = resized
        #self.device = device
        #self.transforms = Compose(self.resize_min_max, self.to_tensor)

    def __call__(self, inputs, targets=None):
        '''
        Arguments:
            inputs(list[dict{'data':PIL.Image,...}])
            targets(list[dict{'boxes':x1y1x2y2, 'cat_labels':}])
        '''
        images = []
        image_path = []
        dataset_label = []
        #scales = np.zeros(len(inputs))
        scales = []
        if targets is None:
            targets=[None]*len(inputs)

        for i, (ainput, target) in enumerate(zip(inputs, targets)):
            # this is operated in
            if not self.resized:
                ainput, target = self.resize_min_max(ainput, target)

            # normalize after resize, which might be slower 
            ainput, target = self.to_tensor(ainput, target)
            # set the tensor to device before normalize
            #ainput['data'].to(self.device)
            #scales[i] = ainput['scale']
            scales.append(ainput['scale'])
            ainput, target = self.normalize(ainput, target)
            images.append(ainput['data'])

            if 'path' in ainput:
                image_path.append(ainput['path'])
            if 'dataset_label' in ainput:
                dataset_label.append(ainput['dataset_label'])

        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        _, height, width = max_size

        image_sizes = [img.shape[-2:] for img in images]
        self.group_padding = GroupPadding(width, height, size_devidable=32)
        im_tensor = self.group_padding(images)

        inputs = {}
        inputs['data'] = im_tensor
        inputs['scale'] = scales
        inputs['image_sizes'] = image_sizes
        if len(image_path) > 0:
            inputs['path'] = image_path
        if len(dataset_label) > 0:
            inputs['dataset_label'] = torch.tensor(dataset_label)
        return inputs, targets

# a version similar to torchvision
@TRANSFORM_REG.register()
class GeneralRCNNTransformTV(object): 
    def __init__(self, min_size, max_size, image_mean=None, image_std=None, resized=False):
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.resized = resized
        self.resize_min_max = ResizeMinMaxTV(min_size, max_size)
        self.normalize = Normalize(mean=image_mean, std=image_std)
        self.to_tensor = ToTensor()
        #self.device = device
        #self.transforms = Compose(self.resize_min_max, self.to_tensor)

    def __call__(self, inputs, targets=None):
        '''
        Arguments:
            inputs(list[dict{'data':PIL.Image,...}])
            targets(list[dict{'boxes':x1y1x2y2, 'cat_labels':}])
        '''
        images = []
        image_path = []
        dataset_label = []
        #scales = np.zeros(len(inputs))
        scales = []
        if targets is None:
            targets=[None]*len(inputs)

        for i, (ainput, target) in enumerate(zip(inputs, targets)):
            # this is operated in
            #ainput, target = self.resize_min_max(ainput, target)

            # normalize after resize, which might be slower 
            ainput, target = self.to_tensor(ainput, target)
            ainput, target = self.normalize(ainput, target)
            if not self.resized:
                ainput, target = self.resize_min_max(ainput, target)
            # set the tensor to device before normalize
            #ainput['data'].to(self.device)
            scales.append(ainput['scale'])
            images.append(ainput['data'])

            if 'path' in ainput:
                image_path.append(ainput['path'])
            if 'dataset_label' in ainput:
                dataset_label.append(ainput['dataset_label'])

        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        _, height, width = max_size

        image_sizes = [img.shape[-2:] for img in images]
        self.group_padding = GroupPadding(width, height, size_devidable=32)
        im_tensor = self.group_padding(images)

        inputs = {}
        inputs['data'] = im_tensor
        inputs['scale'] = scales
        inputs['image_sizes'] = image_sizes
        if len(image_path) > 0:
            inputs['path'] = image_path
        if len(dataset_label) > 0:
            inputs['dataset_label'] = torch.tensor(dataset_label)
        return inputs, targets


@TRANSFORM_REG.register()
class RandomMirror(object):
    def __init__(self, probability=0.5, inputs_box_keys=[], targets_box_keys=['boxes'], mask_key=None):
        self.probability = probability
        self.inputs_box_keys = inputs_box_keys
        self.targets_box_keys = targets_box_keys
        self.mask_key = mask_key

    def __call__(self, inputs, targets):
        inputs['mirrored'] = False
        if random.random() < self.probability:
            inputs['mirrored'] = True
            inputs['data'] = F.mirror(inputs['data'])
            
            im_width = inputs['data'].width
            for input_key in self.inputs_box_keys:
                inputs[input_key] = F.mirror_boxes(inputs[input_key], im_width)

            for target_key in self.targets_box_keys:
                targets[target_key] = F.mirror_boxes(targets[target_key], im_width)
            if self.mask_key is not None:
                targets[self.mask_key] = F.mirror_masks(targets[self.mask_key])
        
        return inputs, targets

@TRANSFORM_REG.register()
class RandomCrop(object):
    '''
       Random crop to the image
       The padding will be added if the image is too samll
       The target_box_main_key is 'boxes' in detection task
       other box_keys can be the additional boxes used in the process
       targets_other_key can be labels, etc
       After cropping, the invalid boxes and labels are deleted
       padding_value: the value that pad if the image is bigger after cropping
    '''
    def __init__(self, size, box_inside, target_box_main_key='boxes', inputs_box_keys=[], targets_box_keys=[], targets_other_key=['labels'], inputs_box_inside=False, targets_box_inside=False, mask_key=None, padding_value=0):
        if isinstance(size, Iterable):
            self.size = size
        else:
            self.size = [size, size]
        assert len(self.size) == 2
        self.box_inside = box_inside
        self.inputs_box_keys = inputs_box_keys
        self.targets_box_keys = targets_box_keys
        self.targets_other_key = targets_other_key
        self.targets_box_main_key = target_box_main_key
        self.inputs_box_inside = inputs_box_inside
        self.targets_box_inside = targets_box_inside
        if box_inside:
            assert target_box_main_key is not None
        if target_box_main_key is not None:
            assert isinstance(target_box_main_key, str)
        self.mask_key = mask_key
        self.padding_value = padding_value


    def __call__(self, inputs, targets):
        image = inputs['data']
        while True:
            inputs['data'], position = F.random_crop(image, self.size, self.padding_value)
            inputs['crop_position'] = position

            #if 'boxes' in targets:
            if self.targets_box_main_key is not None:
                boxes = targets[self.targets_box_main_key].copy()
                boxes = F.random_crop_boxes(boxes, position)
                # only keep the valid boxes
                keep = np.logical_and(boxes[:,3]>boxes[:,1], boxes[:,2]>boxes[:,0])
                boxes_keep = boxes[keep]
                if not self.box_inside or keep.any():
                    valid = True
                    inputs_temp = {}
                    for k in self.inputs_box_keys:
                        inputs_temp[k] = F.random_crop_boxes(inputs[k].copy(), position)
                        if self.inputs_box_inside:
                            boxes_temp = inputs_temp[k]
                            keep_temp = np.logical_and(boxes_temp[...,3]>boxes_temp[...,1], boxes_temp[...,2]>boxes_temp[...,0])
                            if not keep_temp.any():
                                valid = False
                                break
                            if not box_overlap(boxes_temp, boxes_keep):
                                valid = False
                                break

                        inputs_temp[k] = inputs_temp[k][keep_temp]
                    if not valid:
                        continue

                    targets_temp = {}
                    for k in self.targets_box_keys:
                        targets_temp[k] = F.random_crop_boxes(targets[k].copy(), position)
                        if self.targets_box_inside:
                            boxes_temp = targets_temp[k]
                            keep_temp = np.logical_and(boxes_temp[...,3]>boxes_temp[...,1], boxes_temp[...,2]>boxes_temp[...,0])
                            if not keep_temp.any():
                                valid = False
                                break
                            if not box_overlap(boxes_temp, boxes_keep):
                                valid = False
                                break
                        targets_temp[k] = targets_temp[k][keep_temp]
                    if not valid:
                        continue
                else:
                    continue
                if self.mask_key is not None:
                    targets[self.mask_key] = targets[self.mask_key][keep]
                    targets[self.mask_key] = F.crop_masks(targets[self.mask_key], position)
                # put it in the last line in case override the original one before other things are settle
                targets[self.targets_box_main_key] = boxes_keep
                for k in self.targets_other_key:
                    targets[k] = targets[k][keep]

                for k in self.inputs_box_keys:
                    inputs[k] =inputs_temp[k]
                for k in self.targets_box_keys:
                    targets[k] =targets_temp[k]

                break
            else:
                break

        return inputs, targets

def box_overlap(box, boxes):
    x1 = np.maximum(box[...,0], boxes[...,0])
    y1 = np.maximum(box[...,1], boxes[...,1])
    x2 = np.minimum(box[...,2], boxes[...,2])
    y2 = np.minimum(box[...,3], boxes[...,3])
    if ((x2>x1) & (y2>y1)).any():
        return True
    else:
        return False


@TRANSFORM_REG.register()
class RandomScale(object):
    def __init__(self, low, high, inputs_box_keys=[], targets_box_keys=['boxes'], mask_key=None):
        assert low > 0
        assert high > 0
        assert low<=high
        self.low =low
        self.high = high
        self.inputs_box_keys = inputs_box_keys
        self.targets_box_keys = targets_box_keys
        self.mask_key = mask_key

    def __call__(self, inputs, targets):
        scale = random.uniform(self.low, self.high)
        image = inputs['data']
        inputs['data'] = F.scale(image, scale)
        # in case scale already used in other operation
        if 'scale' not in inputs:
            inputs['scale'] = scale
        else:
            inputs['scale'] = scale * inputs['scale']

        for input_key in self.inputs_box_keys:
            inputs[input_key] = F.scale_box(inputs[input_key], scale)

        for target_key in self.targets_box_keys:
            targets[target_key] = F.scale_box[targets[target_key], scale]
        if self.mask_key is not None:
            targets[self.mask_key] = F.scale_masks(targets[self.mask_key], scale)
        
        return inputs, targets

@TRANSFORM_REG.register()
class RandomAbsoluteScale(object):
    def __init__(self, low, high, inputs_box_keys=[], targets_box_keys=['boxes'], mask_key=None):
        assert low > 0
        assert high > 0
        assert low<=high
        self.low = int(low)
        self.high = int(high)
        self.inputs_box_keys = inputs_box_keys
        self.targets_box_keys = targets_box_keys
        self.mask_key = mask_key

    def __call__(self, inputs, targets):
        longgest_side = random.randint(self.low, self.high)
        width, height = inputs['data'].width, inputs['data'].height
        max_side = max(width, height)
        scale = longgest_side / max_side
        image = inputs['data']
        inputs['data'] = F.scale(image, scale)
        # in case scale already used in other operation
        if 'scale' not in inputs:
            inputs['scale'] = scale
        else:
            inputs['scale'] = scale * inputs['scale']

        for input_key in self.inputs_box_keys:
            inputs[input_key] = F.scale_box(inputs[input_key], scale)

        for target_key in self.targets_box_keys:
            targets[target_key] = F.scale_box(targets[target_key], scale)
        if self.mask_key is not None:
            targets[self.mask_key] = F.scale_masks(targets[self.mask_key], scale)
        
        return inputs, targets

@TRANSFORM_REG.register()
class PadNumpyArray():
    # pad zero to array
    def __init__(self, input_len_dict=None, input_val_dict=None, target_len_dict=None, target_val_dict=None):
        if input_len_dict is not None:
            assert input_len_dict.keys() == input_val_dict.keys()
        if target_len_dict is not None:
            assert target_len_dict.keys() == target_val_dict.keys()
        self.input_len_dict = input_len_dict
        self.input_val_dict = input_val_dict
        self.target_len_dict = target_len_dict
        self.target_val_dict = target_val_dict

    def __call__(self, inputs, targets):
        if self.input_len_dict is not None:
            inputs = self.pad_variable(inputs, self.input_len_dict, self.input_val_dict)
        if self.target_len_dict is not None:
            targets = self.pad_variable(targets, self.target_len_dict, self.target_val_dict)
        return inputs, targets
    
    def pad_variable(self, targets, len_dict, val_dict):
        for k in len_dict:
            length = len_dict[k]
            value = val_dict[k]
            shape = (length,)+targets[k].shape[1:]
            dtype = targets[k].dtype
            pad_array = np.full(shape, value, dtype=dtype)
            pad_array[:len(targets[k])] = targets[k]
            targets[k] = pad_array
        return targets

@TRANSFORM_REG.register()
class AddSurrandingBox(object):
    def __init__(self, box_name='target_box') -> None:
        self.box_name = box_name

    def __call__(self, inputs, targets):
        if targets is None:
            return inputs, targets
        targets[self.box_name] = F.surrounding_box(targets['boxes'])
        return inputs, targets

@TRANSFORM_REG.register()
class AddPersonBox(object):
    def __init__(self, anno_path, name='input_box', out_name='input_box', targets_boxes_name='boxes', add_extra_dim=False, extend_to_target_boxes=None, extra_padding=None, random_scale_and_crop=None, extend_ratio=None):
        '''
        name: the key of person box in annotation file
        out_name: the key in output targets
        targets_boxes_name: garments boxes key in annotation file
        add_extra_dim: if we need to add extra dimension for person box since its size can be (4,), instead of (n,4)
        extra_padding: do we need to add extra padding for person box so it can be devided by the number
        random_scale_and_crop: List or tuple(min_scale, max_scale) randomly extend or shrink the person box for training, random scale ONLY SUPPORT ONE PERSON BOX
        extend_ratio: extend the person box by the ratio from center, the 
        code does nothing when extend_ratio=1, the output box will have the
        side_length=side_length*extend_ratio
        '''
        self.name = name
        self.out_name = out_name
        self.targets_boxes_name = targets_boxes_name
        self.add_extra_dim = add_extra_dim
        self.extend_to_target_boxes = extend_to_target_boxes
        self.extra_padding = extra_padding
        self.random_scale = random_scale_and_crop
        self.extend_ratio = extend_ratio
        assert sum([x is not None for x in [self.extend_to_target_boxes, self.random_scale, self.extend_ratio]]) <= 1
        with open(anno_path, 'rb') as f:
            self.anno = pickle.load(f)
    
    def __call__(self, inputs, targets):
        im_id = targets['image_id']
        input_box = self.anno[im_id][self.name]
        if self.add_extra_dim:
            roi_box = np.expand_dims(np.array(input_box), axis=0)
        else:
            roi_box = np.array(input_box)

        if self.extend_to_target_boxes:
            boxes = targets[self.targets_boxes_name]
            x1 = min(boxes[:,0])
            y1 = min(boxes[:,1])
            x2 = max(boxes[:,2])
            y2 = max(boxes[:,3])

            roi_box[...,0] = min(x1, roi_box[...,0])
            roi_box[...,1] = min(y1, roi_box[...,1])
            roi_box[...,2] = max(x2, roi_box[...,2])
            roi_box[...,3] = max(y2, roi_box[...,3])

        if self.extend_ratio is not None:
            im_width, im_height = inputs['data'].size
            roi_box = F.extend_boxes(roi_box, self.extend_ratio, im_width, im_height)

        if self.random_scale is not None:
            assert self.random_scale[1] >= self.random_scale[0]
            # only support one person box
            assert roi_box.shape == (1,4) or roi_box.shape == (4,)
            while True:
                scale = np.random.uniform(low=self.random_scale[0],high=self.random_scale[1], size=4)

                x1 = roi_box[...,0].copy()
                y1 = roi_box[...,1].copy()
                x2 = roi_box[...,2].copy()
                y2 = roi_box[...,3].copy()

                h_half = (y2 - y1) / 2
                w_half = (x2 - x1) / 2

                x1 -= w_half*(scale[0]-1)
                y1 -= h_half*(scale[1]-1)
                x2 += w_half*(scale[2]-1)
                y2 += h_half*(scale[3]-1)

                #  garment boxes should be inside the human box and smaller than the image size
                boxes = targets[self.targets_boxes_name].copy()
                boxes[...,0] = np.maximum(boxes[...,0], x1)
                boxes[...,1] = np.maximum(boxes[...,1], y1)
                boxes[...,2] = np.minimum(boxes[...,2], x2)
                boxes[...,3] = np.minimum(boxes[...,3], y2)

                if ((boxes[...,2]>boxes[...,0])&(boxes[...,3]>boxes[...,1])).any():
                    break

            im_width, im_height = inputs['data'].size
            x1 = x1.clip(0, im_width)
            y1 = y1.clip(0, im_height)
            x2 = x2.clip(0, im_width)
            y2 = y2.clip(0, im_height)

            roi_box[...,0] = x1
            roi_box[...,1] = y1
            roi_box[...,2] = x2
            roi_box[...,3] = y2

        
        if self.extra_padding is not None:
            assert isinstance(self.extra_padding, int)
            roi_box[...,0] = (roi_box[...,0] // self.extra_padding) * self.extra_padding
            roi_box[...,1] = (roi_box[...,1] // self.extra_padding) * self.extra_padding
            roi_box[...,2] = (roi_box[...,2] // self.extra_padding + 1) * self.extra_padding
            roi_box[...,3] = (roi_box[...,3] // self.extra_padding + 1) * self.extra_padding
            
        targets[self.out_name] = roi_box
        return inputs, targets

@TRANSFORM_REG.register()
class ImageToTensor(object):
    def __init__(self):
        pass

    def __call__(self, inputs, targets=None):
        inputs['data'] = F.image_to_tensor(inputs['data'])

        if targets is not None:
            for k,v in targets.items():
                if isinstance(v, np.ndarray):
                    targets[k] = torch.from_numpy(targets[k])
        return inputs, targets
    

@TRANSFORM_REG.register()
class GroupPaddingWithBBox(object):
    def __init__(self, image_mean=None, image_std=None, normalize=True, normalize_to_one=True, pad_value=0):
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalize = Normalize(mean=image_mean, std=image_std)
        if normalize_to_one:
            self.to_tensor = ToTensor()
        else:
            self.to_tensor = ImageToTensor()
        self.normalize_tensor = normalize
        self.normalize_image_to_one = normalize_to_one
        self.pad_value=pad_value
        #self.device = device
        #self.transforms = Compose(self.resize_min_max, self.to_tensor)

    def __call__(self, inputs, targets=None):
        '''
        Arguments:
            inputs(list[dict{'data':PIL.Image,...}])
            targets(list[dict{'boxes':x1y1x2y2, 'cat_labels':}])
        '''
        images = []
        image_path = []
        dataset_label = []
        scales = np.ones(len(inputs))
        if targets is None:
            targets=[None]*len(inputs)

        for i, (ainput, target) in enumerate(zip(inputs, targets)):
            # normalize after resize, which might be slower 
            ainput, target = self.to_tensor(ainput, target)
            # set the tensor to device before normalize
            #ainput['data'].to(self.device)
            if 'scale' in ainput:
                scales[i] = ainput['scale']
            if self.normalize_tensor:
                ainput, target = self.normalize(ainput, target)
            images.append(ainput['data'])

            if 'path' in ainput:
                image_path.append(ainput['path'])
            if 'dataset_label' in ainput:
                dataset_label.append(ainput['dataset_label'])

        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        _, height, width = max_size

        image_sizes = [img.shape[-2:] for img in images]
        self.group_padding = GroupPadding(width, height, size_devidable=32,pad_value=self.pad_value)
        im_tensor = self.group_padding(images)

        inputs = {}
        inputs['data'] = im_tensor
        inputs['scale'] = scales
        inputs['image_sizes'] = image_sizes
        if len(image_path) > 0:
            inputs['path'] = image_path
        if len(dataset_label) > 0:
            inputs['dataset_label'] = torch.tensor(dataset_label)
        return inputs, targets

@TRANSFORM_REG.register(force=True)
class Mosaic():
    '''This class apply Mosaic transfrom on images
        First, select one image as the top-left image,
        Second, random select there other images from the dataset 
        Third, resize each image to the img_size and add it to mosaic_image
    '''
    def __init__(self,
                 img_size=(640, 640), # (h,w)
                 center_ratio_range=(0.5, 1.5),
                 min_bbox_size=0,
                 bbox_clip_border=True,
                 skip_filter=True,
                 pad_val=114,
                 prob=1.0,
                 return_pillow_img=True,
                 ):
        self.img_size = img_size
        self.out_img_size = tuple(int(s*2) for s in img_size)
        self.center_ratio_range = center_ratio_range
        self.resize_max = ResizeMax(img_size,)

        #(h_low, w_low, h_high, w_high)
        self.y_low = int(center_ratio_range[0]*img_size[0]),
        self.x_low = int(center_ratio_range[0]*img_size[1]),
        self.y_high = int(center_ratio_range[1]*img_size[0]),
        self.x_high = int(center_ratio_range[1]*img_size[1])

        self.min_bbox_size = min_bbox_size
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter=skip_filter
        self.pad_val = pad_val
        self.prob = prob
        self.locs = ['top_left','top_right','bottom_left','bottom_right']
        self.return_pillow_img = return_pillow_img

    def __call__(self, inputs, targets=None):
        assert 'extra_images' in inputs
        if isinstance(inputs['data'], Image.Image):
            im = np.array(inputs['data'])
        else:
            im = inputs['data']

        mosaic_im = np.full((*self.out_img_size, 3), self.pad_val, dtype=im.dtype)

        # get the center point
        c_y = np.random.randint(self.y_low, self.y_high)
        c_x = np.random.randint(self.x_low, self.x_high)
        
        mosaic_boxes = []
        mosaic_labels = []
        for i, loc in enumerate(self.locs):
            if loc == 'top_left':
                inputs_temp, targets_temp = inputs, targets
            else:
                inputs_temp, targets_temp = inputs['extra_images'][i-1]
            inputs_temp, targets_temp = self.resize_max(inputs_temp, targets_temp)
            im = inputs_temp['data']
            im_w,im_h = im.size
            im_cord, crop_cord = self._cal_shift_and_crop((im_w,im_h), loc, c_x, c_y)
            x1, y1, x2, y2 = im_cord
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_cord
            #print('loc',loc)
            #print('im shape',im_w, im_h)
            #print('cx, cy', c_x, c_y)
            #print('im cord:',im_cord)
            #print('crop cord', crop_cord)

            #print(im.size)
            mosaic_im[y1:y2,x1:x2] = np.array(im)[crop_y1:crop_y2, crop_x1:crop_x2]

            gt_boxes = targets_temp['boxes']
            gt_labels = targets_temp['labels']

            if gt_boxes.shape[0]>0:
                pad_w = x1 - crop_x1
                pad_h = y1 - crop_y1
                gt_boxes[:, 0::2] += pad_w
                gt_boxes[:, 1::2] += pad_h

            mosaic_boxes.append(gt_boxes)
            mosaic_labels.append(gt_labels)
        
        if len(mosaic_labels) >0:
            mosaic_boxes = np.concatenate(mosaic_boxes, axis=0)
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)

            if self.bbox_clip_border:
                # clip the boxes and remove the boxes outside the image
                mosaic_boxes[:,0::2] = np.clip(mosaic_boxes[:,0::2], 0, self.out_img_size[1])
                mosaic_boxes[:,1::2] = np.clip(mosaic_boxes[:,1::2], 0, self.out_img_size[0])

            if not self.skip_filter:
                mosaic_boxes, mosaic_labels = self._get_valid_size_box(mosaic_boxes, mosaic_labels)

            # if not clip box, the boxes in the boarder can be removed
            keep = self.inside_boxes(mosaic_boxes)
            mosaic_boxes = mosaic_boxes[keep]
            mosaic_labels = mosaic_labels[keep]

        if self.return_pillow_img:
            mosaic_im = Image.fromarray(mosaic_im)
        inputs['data'] = mosaic_im
        targets['boxes'] = mosaic_boxes
        targets['labels'] = mosaic_labels

        return inputs, targets
        

    def inside_boxes(self, boxes):
        return (boxes[:,0]<self.out_img_size[1]) & (boxes[:,2]>0) &\
            (boxes[:,1]<self.out_img_size[0]) & (boxes[:,3]>0)

    def _get_valid_size_box(self, boxes, labels):
        w = boxes[:,2] - boxes[:,0]
        h = boxes[:,3] - boxes[:,1]

        valid = (w>self.min_bbox_size) & (h>self.min_bbox_size)
        return boxes[valid], labels[valid]


    def _cal_shift_and_crop(self, im_size_wh, loc, center_x, center_y):
        '''im_size: (h, w)'''
        if loc == 'top_left':
            x1 = max(0, center_x - im_size_wh[0])
            y1 = max(0, center_y - im_size_wh[1])
            x2 = center_x
            y2 = center_y

            crop_x1 = im_size_wh[0]-(x2-x1)
            crop_y1 = im_size_wh[1]-(y2-y1)
            crop_x2 = im_size_wh[0]
            crop_y2 = im_size_wh[1]
        elif loc == 'top_right':
            x1 = center_x
            y1 = max(0, center_y - im_size_wh[1])
            x2 = min(self.out_img_size[1], center_x + im_size_wh[0])
            y2 = center_y

            crop_x1 = 0
            crop_y1 = im_size_wh[1]-(y2-y1)
            crop_x2 = x2-x1
            crop_y2 = im_size_wh[1]
        elif loc == 'bottom_left':
            x1 = max(0, center_x - im_size_wh[0])
            y1 = center_y
            x2 = center_x
            y2 = min(self.out_img_size[0], center_y + im_size_wh[1])

            crop_x1 = im_size_wh[0]-(x2-x1)
            crop_y1 = 0
            crop_x2 = im_size_wh[0]
            crop_y2 = y2-y1
        elif loc == 'bottom_right':
            x1 = center_x
            y1 = center_y
            x2 = min(self.out_img_size[1], center_x + im_size_wh[0])
            y2 = min(self.out_img_size[0], center_y + im_size_wh[1])

            crop_x1 = 0
            crop_y1 = 0
            crop_x2 = x2-x1
            crop_y2 = y2-y1

        return (int(x1),int(y1),int(x2),int(y2)), (int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2))


    def get_index(self,dataset):
        return np.random.randint(0, len(dataset), 3)

@TRANSFORM_REG.register()
class MixUp:
    '''default transforms should be:
        mix_up_transforms = [dict(type='ResizeMax',
                          max_size=(min_size,min_size), #(h,w)
                          ),
                    dict(type='RandomMirror',
                        probability=0.5, 
                        targets_box_keys=['boxes'], 
                        mask_key=None),
                    dict(type='RandomAbsoluteScale',
                        low=max_size/2,
                        high=max_size*2,
                        targets_box_keys=['boxes'], 
                        mask_key=None),
                    dict(type='RandomCrop',
                        size=max_size,
                        box_inside=True, 
                        mask_key=None)
                    ]
    '''
    def __init__(self, 
                 transforms=None,
                 min_bbox_size=5,
                 min_area_ratio=0.2,
                 max_aspect_ratio=20, 
                 backend='pillow'):
        if transforms is None:
            self._transforms= []
        else:
            self._transforms = [build_transform(t) for t in transforms]

        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.backend = backend

    def __call__(self, inputs, targets):
        assert 'extra_images' in inputs
        assert len(inputs['extra_images']) == 1 
        if isinstance(inputs['data'], Image.Image):
            im = np.array(inputs['data'])
        else:
            im = inputs['data']
        inputs_temp, targets_temp = inputs['extra_images'][0]
        for t in self._transforms:
            inputs_temp, targets_temp = t(inputs_temp, targets_temp)
        assert inputs['data'].size == inputs_temp['data'].size, \
        "the correct transform should be selected to make the output size same with original image" 

        # mix up
        if self.backend == 'opencv':
            im = inputs['data']*0.5 + inputs_temp['data']*0.5
        else:
            im = Image.blend(inputs['data'], inputs_temp['data'], alpha=0.5)

        mix_up_boxes = np.concatenate((targets['boxes'], targets_temp['boxes']), axis=0)
        mix_up_labels = np.concatenate((targets['labels'], targets_temp['labels']), axis=0)

        #TODO remove outside boxes, not sure if it is necessary

        #TODO The difference without padding is that the defualt value is 0 instead of 114

        inputs['data'] = im
        targets['boxes'] = mix_up_boxes
        targets['labels'] = mix_up_labels
        return inputs, targets
        



    def _filter_box_candidates(self, bbox1, bbox2):
        """Compute candidate boxes which include following 5 things:

        bbox1 before augment, bbox2 after augment, min_bbox_size (pixels),
        min_area_ratio, max_aspect_ratio.
        """

        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return ((w2 > self.min_bbox_size)
                & (h2 > self.min_bbox_size)
                & (w2 * h2 / (w1 * h1 + 1e-16) > self.min_area_ratio)
                & (ar < self.max_aspect_ratio))

    def get_index(self,dataset):
        return np.random.randint(0, len(dataset), 1)

#copy and revise from mmdetection
@TRANSFORM_REG.register()
class RandomAffine:
    """Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Default: 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Default: 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Default: (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Default: 2.
        border (tuple[int]): Distance from height and width sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Default: (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Default: (114, 114, 114).
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 2.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` and `min_area_ratio` and `max_aspect_ratio`
            is invalid. Default to True.
    """

    def __init__(self,
                 max_rotate_degree=10.0,
                 max_translate_ratio=0.1,
                 scaling_ratio_range=(0.5, 1.5),
                 max_shear_degree=2.0,
                 border=(0, 0),
                 border_val=(114, 114, 114),
                 min_bbox_size=2,
                 min_area_ratio=0.2,
                 max_aspect_ratio=20,
                 bbox_clip_border=True,
                 skip_filter=True,
                 return_pillow_image=True,
                 bbox_keys=['boxes']):
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter
        self.return_pillow_image=return_pillow_image
        self.bbox_keys = bbox_keys

    def __call__(self, inputs, targets):
        img = np.array(inputs['data'])
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        # Center
        center_matrix = np.eye(3, dtype=np.float32)
        center_matrix[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        center_matrix[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(0.5-self.max_translate_ratio,
                                 0.5+self.max_translate_ratio) * width
        trans_y = random.uniform(0.5-self.max_translate_ratio,
                                 0.5+self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix
            @ center_matrix)

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        if self.return_pillow_image:
            img = Image.fromarray(img)
        inputs['data'] = img
        #results['img_shape'] = img.shape

        for key in self.bbox_keys:
            bboxes = targets[key]
            num_bboxes = len(bboxes)
            if num_bboxes:
                # homogeneous coordinates
                xs = bboxes[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
                ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
                ones = np.ones_like(xs)
                points = np.vstack([xs, ys, ones])

                warp_points = warp_matrix @ points
                warp_points = warp_points[:2] / warp_points[2]
                xs = warp_points[0].reshape(num_bboxes, 4)
                ys = warp_points[1].reshape(num_bboxes, 4)

                warp_bboxes = np.vstack(
                    (xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

                if self.bbox_clip_border:
                    warp_bboxes[:, [0, 2]] = \
                        warp_bboxes[:, [0, 2]].clip(0, width)
                    warp_bboxes[:, [1, 3]] = \
                        warp_bboxes[:, [1, 3]].clip(0, height)

                # remove outside bbox
                valid_index = find_inside_bboxes(warp_bboxes, height, width)
                if not self.skip_filter:
                    # filter bboxes
                    filter_index = self.filter_gt_bboxes(
                        bboxes * scaling_ratio, warp_bboxes)
                    valid_index = valid_index & filter_index

                targets[key] = warp_bboxes[valid_index]
                if key in ['boxes']:
                    if 'labels' in targets:
                        targets['labels'] = targets['labels'][
                            valid_index]

                if 'masks' in targets:
                    raise NotImplementedError(
                        'RandomAffine only supports bbox.')
        return inputs, targets

    def filter_gt_bboxes(self, origin_bboxes, wrapped_bboxes):
        origin_w = origin_bboxes[:, 2] - origin_bboxes[:, 0]
        origin_h = origin_bboxes[:, 3] - origin_bboxes[:, 1]
        wrapped_w = wrapped_bboxes[:, 2] - wrapped_bboxes[:, 0]
        wrapped_h = wrapped_bboxes[:, 3] - wrapped_bboxes[:, 1]
        aspect_ratio = np.maximum(wrapped_w / (wrapped_h + 1e-16),
                                  wrapped_h / (wrapped_w + 1e-16))

        wh_valid_idx = (wrapped_w > self.min_bbox_size) & \
                       (wrapped_h > self.min_bbox_size)
        area_valid_idx = wrapped_w * wrapped_h / (origin_w * origin_h +
                                                  1e-16) > self.min_area_ratio
        aspect_ratio_valid_idx = aspect_ratio < self.max_aspect_ratio
        return wh_valid_idx & area_valid_idx & aspect_ratio_valid_idx

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_rotate_degree={self.max_rotate_degree}, '
        repr_str += f'max_translate_ratio={self.max_translate_ratio}, '
        repr_str += f'scaling_ratio={self.scaling_ratio_range}, '
        repr_str += f'max_shear_degree={self.max_shear_degree}, '
        repr_str += f'border={self.border}, '
        repr_str += f'border_val={self.border_val}, '
        repr_str += f'min_bbox_size={self.min_bbox_size}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_aspect_ratio={self.max_aspect_ratio}, '
        repr_str += f'skip_filter={self.skip_filter})'
        return repr_str

    @staticmethod
    def _get_rotation_matrix(rotate_degrees):
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_share_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees, y_shear_degrees):
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                                dtype=np.float32)
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x, y):
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                      dtype=np.float32)
        return translation_matrix