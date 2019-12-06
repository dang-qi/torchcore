from PIL import Image
import collections
import torch
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

class Resize_and_Padding(object):
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

class Resize_and_Padding_Min_Max(object):
    '''Resize and Pandding function for Rcnn'''
    def __init__(self, min_size, max_size, interplotation=Image.BILINEAR):
        assert isinstance(min_size, int)
        assert isinstance(max_size, int)
        assert max_size >= min_size
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, inputs, targets=None):
        inputs['data'], self.scale = F.resize_min_max(inputs['data'], self.min_size, self.max_size)
        
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
        return inputs, targets

class Normalize(object):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean=[0.485, 0.456, 0.406],
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
