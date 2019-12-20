from PIL import Image
import collections
import torch
import math
import random
import numpy as np
try:
    import accimage
except ImportError:
    accimage = None
from . import transform_func as F

Iterable = collections.abc.Iterable
Sequence = collections.abc.Sequence

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

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

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, inputs, targets=None):
        inputs['data'] = F.to_tensor(inputs['data'])

        if targets is not None:
            if 'boxes' in targets:
                targets['boxes'] = torch.from_numpy(targets['boxes'])
            if 'labels' in targets:
                targets['labels'] = torch.from_numpy(targets['labels'])
            if 'cat_labels' in targets:
                targets['labels'] = torch.from_numpy(targets['cat_labels'])

        return inputs, targets

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

class BatchStack(object):
    def __init__(self):
        pass

    def __call__(self, inputs, targets=None):
        '''here inputs is a list of inputs and targets are lists of targets'''
        pass

#class Padding(object):
#    ''' Pad the image to the given size,
#        The width and height of the original image shouldn't
#        be larger than the width and height
#    '''
#    def __init__(self, width, height, mode='right_down'):
#        self.width = width
#        self.height = height
#        self.mode = mode
#
#    def __call__(inputs, targets=None):
#        inputs, padding = F.pad

class GroupPadding(object):
    ''' Padding for group of images tensors
    '''
    def __init__(self, max_width, max_height, size_devidable=32):
        self.width = int(math.ceil(float(max_width) / size_devidable)*size_devidable)
        self.height = int(math.ceil(float(max_height) / size_devidable)*size_devidable)

    def __call__(self, images):
        '''
        Parameters:
            images(list[tensors]): input images for group padding
        '''
        images  = F.group_padding(images, self.width, self.height)
        return images

class GeneralRCNNTransform(object):
    def __init__(self, min_size, max_size, device, image_mean=None, image_std=None):
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.resize_min_max = ResizeMinMax(min_size, max_size)
        self.normalize = Normalize(mean=image_mean, std=image_std)
        self.to_tensor = ToTensor()
        self.device = device
        #self.transforms = Compose(self.resize_min_max, self.to_tensor)

    def __call__(self, inputs, targets=None):
        '''
        Arguments:
            inputs(list[dict{'data':PIL.Image,...}])
            targets(list[dict{'boxes':x1y1x2y2, 'cat_labels':}])
        '''
        images = []
        scales = np.zeros(len(inputs))
        if targets is None:
            targets=[None]*len(inputs)

        for i, (ainput, target) in enumerate(zip(inputs, targets)):
            # this is operated in
            ainput, target = self.resize_min_max(ainput, target)

            # normalize after resize, which might be slower 
            ainput, target = self.to_tensor(ainput, target)
            # set the tensor to device before normalize
            #ainput['data'].to(self.device)
            scales[i] = ainput['scale']
            ainput, target = self.normalize(ainput, target)
            images.append(ainput['data'])
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        _, height, width = max_size

        image_sizes = [img.shape[-2:] for img in images]
        self.group_padding = GroupPadding(width, height, size_devidable=32)
        im_tensor = self.group_padding(images)

        inputs = {}
        inputs['data'] = im_tensor
        inputs['scale'] = scales
        inputs['image_sizes'] = image_sizes
        return inputs, targets

class RandomMirror(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, inputs, targets):
        inputs['mirrored'] = False
        if random.random() < self.probability:
            inputs['mirrored'] = True
            inputs['data'] = F.mirror(inputs['data'])
            
            if 'boxes' in targets:
                im_width = inputs['data'].width
                targets['boxes'] = F.mirror_boxes(targets['boxes'], im_width)
        
        return inputs, targets
