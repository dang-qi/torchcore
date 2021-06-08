from PIL import Image
import collections
import torch
import math
import random
import numpy as np
import pickle
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

# Torchvision similar version Min Max Resize
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
            inputs['scale'] = scale * inputs['scale']
        #inputs['scale'] = self.scale
        
        if targets is not None:
            if 'boxes' in targets:
                targets['boxes'] = F.resize_boxes(targets['boxes'], scale)
        return inputs, targets

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
    def __init__(self, min_size, max_size, image_mean=None, image_std=None):
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.resize_min_max = ResizeMinMax(min_size, max_size)
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
        scales = np.zeros(len(inputs))
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
            scales[i] = ainput['scale']
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

class RandomCrop(object):
    '''
       Random crop to the image
       The padding will be added if the image is too samll
       The target_box_main_key is 'boxes' in detection task
       other box_keys can be the additional boxes used in the process
       targets_other_key can be labels, etc
       After cropping, the invalid boxes and labels are deleted
    '''
    def __init__(self, size, box_inside, target_box_main_key='boxes', inputs_box_keys=[], targets_box_keys=[], targets_other_key=['labels'], inputs_box_inside=False, targets_box_inside=False, mask_key=None):
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


    def __call__(self, inputs, targets):
        image = inputs['data']
        while True:
            inputs['data'], position = F.random_crop(image, self.size)
            inputs['crop_position'] = position

            #if 'boxes' in targets:
            if self.targets_box_main_key is not None:
                boxes = targets[self.targets_box_main_key].copy()
                boxes = F.random_crop_boxes(boxes, position)
                # only keep the valid boxes
                keep = np.logical_and(boxes[:,3]>boxes[:,1], boxes[:,2]>boxes[:,0])
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
                        targets_temp[k] = targets_temp[k][keep_temp]
                    if not valid:
                        continue
                    if self.mask_key is not None:
                        targets[self.mask_key] = targets[self.mask_key][keep]
                        targets[self.mask_key] = F.crop_masks(targets[self.mask_key], position)
                    # put it in the last line in case override the original one before other things are settle
                    targets[self.targets_box_main_key] = boxes[keep]
                    for k in self.targets_other_key:
                        targets[k] = targets[k][keep]

                    for k in self.inputs_box_keys:
                        inputs[k] =inputs_temp[k]
                    for k in self.targets_box_keys:
                        targets[k] =targets_temp[k]

                    break
                else:
                    continue
            else:
                break

        return inputs, targets

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

class AddSurrandingBox(object):
    def __init__(self, box_name='target_box') -> None:
        self.box_name = box_name

    def __call__(self, inputs, targets):
        if targets is None:
            return inputs, targets
        targets[self.box_name] = F.surrounding_box(targets['boxes'])
        return inputs, targets

class AddPersonBox(object):
    def __init__(self, anno_path, name='input_box', out_name='input_box', targets_boxes_name='boxes', add_extra_dim=False, extend_to_target_boxes=False, extra_padding=None, random_scale_and_crop=None):
        '''
        name: the key of person box in annotation file
        out_name: the key in output targets
        targets_boxes_name: garments boxes key in annotation file
        add_extra_dim: if we need to add extra dimension for person box since its size can be (4,), instead of (n,4)
        extra_padding: do we need to add extra padding for person box so it can be devided by the number
        random_scale_and_crop: List or tuple(min_scale, max_scale) randomly extend or shrink the person box for training, random scale ONLY SUPPORT ONE PERSON BOX
        '''
        self.name = name
        self.out_name = out_name
        self.targets_boxes_name = targets_boxes_name
        self.add_extra_dim = add_extra_dim
        self.extend_to_target_boxes = extend_to_target_boxes
        self.extra_padding = extra_padding
        self.random_scale = random_scale_and_crop
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

        if self.random_scale is not None:
            assert self.random_scale[1] > self.random_scale[0]
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

                x1 += w_half*(scale[0]-1)
                y1 += h_half*(scale[1]-1)
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
    

class GroupPaddingWithBBox(object):
    def __init__(self, image_mean=None, image_std=None):
        self.image_mean = image_mean
        self.image_std = image_std
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
        scales = np.zeros(len(inputs))
        if targets is None:
            targets=[None]*len(inputs)

        for i, (ainput, target) in enumerate(zip(inputs, targets)):
            # normalize after resize, which might be slower 
            ainput, target = self.to_tensor(ainput, target)
            # set the tensor to device before normalize
            #ainput['data'].to(self.device)
            scales[i] = ainput['scale']
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