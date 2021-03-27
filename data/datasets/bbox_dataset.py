import pickle
import torch
import os
import numpy as np
from PIL import Image
from torchcore.data.transform_func import resize_min_max, resize_boxes
from torchcore.data.transforms import Normalize, ToTensor

import re
import collections
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

class BBoxDataset():
    def __init__(self, anno, modanet_anno, part, root, train_ratio=0.9, transforms=None, add_bbox=False, use_modanet_split=False) -> None:
        self._part = part
        self._root = root
        self._transforms = transforms
        self._train_ratio= train_ratio
        self._add_box = add_bbox
        self._use_modanet_split = use_modanet_split
        
        with open(modanet_anno, 'rb') as f:
            annos = pickle.load(f)
            if use_modanet_split:
                anno_part = annos[part]
            #if part == 'train':
            annos = annos['train'] + annos['val']
            #else:
            #    annos = annos['val']

        with open(anno, 'rb') as f:
            self._images = pickle.load(f)['train']

        self.add_info(annos)
        self.convert_anno_to_list()
        self.convert_to_xyxy()
        self.gen_target_boxes()
        if add_bbox:
            self.set_box_max_length()

        if not self._use_modanet_split:
            self.select_subset(part)
        else:
            self.select_modanet_subset(anno_part)
        self._folder = 'train'

        #if part=='test':
        #    self._folder = 'val'
        #else:
        #    self._folder = 'train'

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]

        # Load image
        img_path = os.path.join(self._root, self._folder, image['file_name'] )
        image_id=image['id']
        img = Image.open(img_path).convert('RGB')
        ori_image = img

        input_box = np.array(image['input_box'], dtype=np.float32)
        target_box = np.array(image['target_box'], dtype=np.float32)

        inputs = {}
        inputs['data'] = img

        targets = {}
        #inputs['input_box'] = input_box
        targets['input_box'] = input_box
        targets['target_box'] = target_box
        targets["image_id"] = image_id
        if self._add_box:
            boxes = []
            labels = []
            for obj in image['objects']:
                boxes.append(obj['bbox'])
                labels.append(obj['category_id'])
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            #boxes = np.zeros((self._max_length, 4))
            #labels = np.full(self._max_length, -1)
            #for i, obj in enumerate(image['objects']):
            #    #bbox_len = len(obj['bbox'])
            #    boxes[i,:] = obj['bbox']
            #    labels[i] = obj['category_id']
            targets['boxes'] = boxes
            targets['labels'] = labels

        if self._transforms is not None:
            inputs, targets = self._transforms(inputs, targets)

        return inputs, targets

    def select_modanet_subset(self, anno_part):
        valid_ids = set([anno['id'] for anno in anno_part])
        self._images = [im  for im in self._images if im['id'] in valid_ids ]

    def select_subset(self, part, seed=2021):
        length = len(self._images)
        train_len = int(length*self._train_ratio)
        np.random.seed(seed)
        #inds = np.random.choice(length, sample_len, replace=False)
        inds = np.random.permutation(length)
        if part == 'train':
            inds = inds[:train_len]
        else:
            inds = inds[train_len:]
        self._images = [self._images[i] for i in inds]



    def convert_anno_to_list(self):
        images = []
        for k,v in self._images.items():
            images.append(v)
        self._images = images
        return images

    def gen_target_boxes(self):
        for image in self._images:
            boxes = []
            for obj in image['objects']:
                boxes.append(obj['bbox'])
            boxes = np.array(boxes)
            x1 = np.min(boxes[:,0])
            y1 = np.min(boxes[:,1])
            x2 = np.max(boxes[:,2])
            y2 = np.max(boxes[:,3])
            image['target_box'] = np.array([x1,y1,x2,y2])

    def add_info(self, modanet_anno):
        annos = map_to_id(modanet_anno)
        for k,v in self._images.items():
            v['file_name'] = annos[k]['file_name']
            v['objects'] = annos[k]['objects']
            v['id'] = k

    def convert_to_xyxy(self):
        for image in self._images:
            for obj in image['objects']:
                obj['bbox'][2]+=obj['bbox'][0]
                obj['bbox'][3]+=obj['bbox'][1]

    def set_box_max_length(self):
        max_length = 0
        for image in self._images:
            for obj in image['objects']:
                max_length = max(max_length, len(obj['bbox']))
        self._max_length = max_length


def map_to_id(annos):
    out = {}
    for anno in annos:
        out[anno['id']]=anno
    return out

class BBoxTransform():
    def __init__(self, max_side) -> None:
        self.max_side = max_side
        self.normalize = Normalize()
        self.to_tensor = ToTensor()

    def __call__(self, inputs, targets):
        inputs['data'], scale = resize_min_max(inputs['data'], min_size=self.max_side, max_size=self.max_side)

        inputs['scale'] = scale
        inputs['input_box'] = inputs['input_box'] * scale
        targets['target_box'] = targets['target_box'] * scale
        if 'bbox' in targets:
            targets['bbox'] = targets['bbox'] * scale

        inputs, targets = self.to_tensor(inputs, targets)
        inputs, targets = self.normalize(inputs, targets)

        return inputs, targets

class BBoxCollateFn():
    def __init__(self, max_size, image_mean=None, im_std=None) -> None:
        if isinstance(max_size, (list, tuple)):
            self.transforms = [BBoxTransform(max_side) for max_side in max_size]
            self.transform_num = len(max_size)
            self.multi_scale = True
        else:
            self.transforms = BBoxTransform(max_size)
            self.multi_scale = False

    def __call__(self, batch):
        if self.multi_scale:
            i = np.random.randint(self.transform_num)
            transform = self.transforms[i]
        else:
            transform = self.transforms

        for inputs, targets in batch:
            inputs, targets = transform(inputs, targets)
        out = default_collate(batch)
        return out

# from torch
# https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
default_collate_err_msg_format = (
"default_collate: batch must contain tensors, numpy arrays, numbers, "
"dicts or lists; found {}")

# from torch
# https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

         