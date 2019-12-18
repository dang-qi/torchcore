from .transforms import ResizeAndPadding, ToTensor, Compose
import torch

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