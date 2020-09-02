from .transforms import ResizeAndPadding, ToTensor, Compose
import torch
from torch.utils.data._utils.collate import default_collate

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
    def __init__(self, dataset_num, inputs_keys, targets_keys, ind_key, image_key='data'):
        # parameters for centernet:
        # ind_key = 'dataset_label'
        # input_keys = ['data']
        # target_keys = ['image_id', 'heatmap', 'offset', 'width_height', 'ind', 'ind_mask', 'mask']
        self._dataset_num = dataset_num
        self._inputs_keys = inputs_keys
        self._target_keys = targets_keys
        self._ind_key = ind_key
        self._image_key = image_key

    def __call__(self, batch):
        #batch_size = len(batch)
        inputs_dict = {key:[[] for i in range(self._dataset_num)] for key in self._inputs_keys }
        image_key = self._image_key
        inputs_dict[image_key] = []
        targets_dict = {key:[[] for i in range(self._dataset_num)] for key in self._target_keys }
        inds = []

        for inputs, targets in batch:
            ind = inputs[self._ind_key]
            inds.append(ind)
            for key, val in inputs.items():
                if key == self._ind_key:
                    continue
                elif key == image_key:
                    inputs_dict[key].append(val)
                else:
                    if key not in inputs_dict:
                        inputs_dict[key] = [[] for i in range(self._dataset_num)]
                    inputs_dict[key][ind].append(val)

            for key, val in targets.items():
                if key != self._ind_key:
                    if key not in targets_dict:
                        targets_dict[key] = [[] for i in range(self._dataset_num)]
                    targets_dict[key][ind].append(val)

        for key, val_list in inputs_dict.items():
            if key == image_key:
                inputs_dict[key] = default_collate(val_list)
                continue
            for i, val in enumerate(val_list):
                inputs_dict[key][i] = default_collate(val)

        for key, val_list in targets_dict.items():
            for i, val in enumerate(val_list):
                targets_dict[key][i] = default_collate(val)

        inputs_dict[self._ind_key] = inds
        return inputs_dict, targets_dict