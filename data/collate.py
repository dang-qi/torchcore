from .transforms import ResizeAndPadding, ToTensor, Compose
from .transforms import GeneralRCNNTransform, GeneralRCNNTransformTV
import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np

class rcnn_collate(object):
    ''' collate function for rcnn, inputs are (inputs, targets)'''
    
    def __init__(self, max_size, min_size, img_mean=None, img_std=None):
        if max_size < min_size:
            raise ValueError('max size must equal or larger than min_size')
        self.max_size = max_size
        self.min_size = min_size

    def __call__(self):
        pass
        

class collate_fn(object):
    def __init__(self, batch_transforms=None):
        self.batch_transforms = batch_transforms

    def __call__(self, batch):
        inputs, targets = list(zip(*batch))
        if self.batch_transforms is not None:
            inputs, targets = self.batch_transforms(inputs, targets)
        # stack the images in batch


class yolo_collate(object):
    ''' collate function for yolo, inputs are (inputs, targets)'''
    def __init__(self, size, img_mean=None, img_std=None):
        #self.transforms = Compose(Resize_and_Padding(size), ToTensor())
        self.transforms = Compose([ResizeAndPadding(size), ToTensor()])

    def __call__(self, batch):
        imgs = []
        boxes = []
        cat_labels = []
        all_inputs = []
        all_targets = []
        for inputs, targets in batch:
            inputs, targets = self.transforms(inputs, targets)
            all_inputs.append(inputs)
            all_targets.append(targets)
            imgs.append(inputs['data'])
            boxes.append(targets['boxes'])
            cat_labels.append(targets['cat_labels'])

        inputs['data'] = torch.stack(imgs, dim=0)
        targets['boxes'] = boxes
        #targets['cat_labels'] = torch.cat(cat_labels, dim=-1)
        targets['cat_labels'] = cat_labels

        return inputs, targets


class test_collate(object):
    ''' collate function for test, inputs are (inputs, targets)'''
    def __init__(self, size, img_mean=None, img_std=None):
        #self.transforms = Compose(Resize_and_Padding(size), ToTensor())
        # test all the transform here
        self.transforms = Compose([ResizeAndPadding(size)])

    def __call__(self, batch):
        imgs = []
        boxes = []
        cat_labels = []
        all_inputs = []
        all_targets = []
        for inputs, targets in batch:
            inputs, targets = self.transforms(inputs, targets)
            all_inputs.append(inputs)
            all_targets.append(targets)

        return all_inputs, all_targets

class mix_dataset_collate(object):
    def __init__(self, dataset_num, inputs_split_keys, targets_split_keys, ind_key ):
        # parameters for centernet:
        # ind_key = 'dataset_label'
        # input_keys = ['data']
        # target_keys = ['image_id', 'heatmap', 'offset', 'width_height', 'ind', 'ind_mask', 'mask']
        self._dataset_num = dataset_num
        self._ind_key = ind_key
        self._inputs_split_keys = inputs_split_keys
        self._targets_split_keys = targets_split_keys

    def __call__(self, batch):
        batch_size = len(batch)
        inputs_dict = {key:[[] for i in range(self._dataset_num)] for key in self._inputs_split_keys }
        targets_dict = {key:[[] for i in range(self._dataset_num)] for key in self._targets_split_keys }

        for inputs, targets in batch:
            ind = inputs[self._ind_key]
            self.collect_data_once(inputs, inputs_dict, ind, self._inputs_split_keys)
            self.collect_data_once(targets, targets_dict, ind, self._targets_split_keys)

        self.summary_data(inputs_dict, batch_size, self._inputs_split_keys)
        self.summary_data(targets_dict, batch_size, self._targets_split_keys)

        #inputs_dict[self._ind_key] = torch.tensor(inds)
        return inputs_dict, targets_dict

    def collect_data_once(self, data, data_dict, ind, split_key):
        for key, val in data.items():
            if key in split_key:
                data_dict[key][ind].append(val)
                for i in range(self._dataset_num):
                    if ind != i:
                        data_dict[key][i].append(None)
            else:
                if key not in data_dict:
                    data_dict[key] = []
                data_dict[key].append(val)
                
    def summary_data(self, data_dict, batch_size, split_key):
        for key, val_list in data_dict.items():
            if key not in split_key:
                data_dict[key] = default_collate(val_list)
                #inputs_dict[key] = [default_collate(item) for item in val_list]
            else:
                for i, val in enumerate(val_list):
                    val = self.convert_None_to_zero(val)
                    data_dict[key][i] = default_collate(val)

    def convert_None_to_zero(self, data_list):
        sample = None
        for data in data_list:
            if data is not None:
                sample = np.zeros_like(data)
                break

        if sample is None:
            return data_list

        for i, data in enumerate(data_list):
            if data is None:
                data_list[i] = np.copy(sample)

        return data_list

class CollateFnRCNN(object):
    '''apply general rcnn transform to batchs'''
    def __init__(self, min_size, max_size, image_mean=None, image_std=None):
        if isinstance(min_size, (list, tuple)):
            #self.transforms = [GeneralRCNNTransform(min_size_i, max_size, image_mean=image_mean, image_std=image_std) for min_size_i in min_size]
            self.transforms = [GeneralRCNNTransformTV(min_size_i, max_size, image_mean=image_mean, image_std=image_std) for min_size_i in min_size]
            self.transform_num = len(min_size)
            self.multi_scale = True
        else:
            #self.transforms = GeneralRCNNTransform(min_size, max_size,  
            #                                   image_mean=image_mean, image_std=image_std)
            self.transforms = GeneralRCNNTransformTV(min_size, max_size, image_mean=image_mean, image_std=image_std)
            self.multi_scale = False

    def __call__(self, batch):
        inputs, targets = [list(s) for s in zip(*batch)]
        ori_image = None
        if 'ori_image' in inputs[0]:
            ori_image = [input['ori_image'] for input in inputs]
        if self.multi_scale:
            i = np.random.randint(self.transform_num)
            transform = self.transforms[i]
        else:
            transform = self.transforms
        inputs, targets = transform(inputs, targets)
        if ori_image is not None:
            inputs['ori_image'] = ori_image
        return inputs, targets

def collate_fn_torchvision(batch):
    return tuple(zip(*batch))