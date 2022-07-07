from attr import attributes
from cv2 import add
import numpy as np
import pickle
import torch
from .dataset_new import Dataset
from PIL import Image
import os
from ..dataset_util import get_binary_mask
from torchvision.transforms.functional import to_tensor
import copy
from .build import DATASET_REG
from .coco import COCODataset

@DATASET_REG.register(force=True)
class SVHNDataset(COCODataset):
    '''SVHN dataset'''
    def __init__(self, root, anno, part, transforms=None, debug=False, xyxy=True, torchvision_format=False, add_mask=False, first_n_subset=None, subcategory=None, map_id_to_continuous=True, backend='pillow', RGB=True):
        super().__init__(root, anno, part, transforms, debug, xyxy, torchvision_format, add_mask, first_n_subset, subcategory, map_id_to_continuous, backend, RGB)
        self.zero_start_label_index = False

    #def __getitem__(self, idx):
    #    inputs, targets = super().__getitem__(idx)
    #    image = self._images[idx]
    #    if self.add_attibutes:
    #        attributes = []
    #        for obj in image['objects']:
    #            attributes.append(obj['attribute_ids'])
    #    if self.add_attibutes:
    #        targets['attributes'] = attributes
    #    return inputs, targets

    @property
    def category_id_name_dict(self):
        id_name_dict = {i+1: i+1 for i in range(10)}
        id_name_dict[10] = 0
        return id_name_dict

