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

@DATASET_REG.register()
class FashionPediaDataset(COCODataset):
    '''FashionPedia dataset'''
    pass
#    def __init__( self, root, anno, part, transforms=None, xyxy=True, debug=False, torchvision_format=False, add_mask=False, sub_category=None, map_id_to_continuous=False ):
#        super().__init__( root, anno=anno, part=part, transforms=transforms )
#        self._part = part
#        folder_dict = {'train':'train', 'val':'test', 'test':'test'}
#        self._folder = folder_dict[part]
#
#        self.torchvision_format = torchvision_format
#        self.add_mask = add_mask
#
#        ## load annotations
#        #with open(anno, 'rb') as f:
#        #    self._images = pickle.load(f)[part] 
#        self.xyxy = xyxy
#        if xyxy:
#            self.convert_to_xyxy()
#        if sub_category is not None:
#            self.set_category_subset(sub_category, ignore_other_category=True)
#
#        if map_id_to_continuous:
#            self.map_category_id_to_continous()
#        self._set_aspect_ratio_flag()
#        self.debug = debug
#
#    def __len__(self):
#        return len(self._images)
#
#    def __getitem__(self, idx):
#        image = self._images[idx]
#
#        # Load image
#        img_path = os.path.join(self._root, self._folder, image['file_name'] )
#        image_id=image['id']
#        img = Image.open(img_path).convert('RGB')
#        ori_image = img
#
#        # Load targets
#        boxes = []
#        labels = []
#        for obj in image['objects']:
#            boxes.append(obj['bbox'])
#            labels.append(obj['category_id'])
#        boxes = np.array(boxes, dtype=np.float32)
#        labels = np.array(labels, dtype=np.int64)
#
#
#        if self.add_mask:
#            height = image['height']
#            width = image['width']
#            masks = [get_binary_mask(obj['segmentation'], height, width, use_compressed_rle=True) for obj in image['objects']]
#            masks = np.array(masks, dtype=np.uint8)
#
#        inputs = {}
#        inputs['data'] = img
#        if self.debug:
#            inputs['ori_image'] = ori_image
#
#        targets = {}
#        targets["boxes"] = boxes
#        targets["cat_labels"] = labels 
#        targets["labels"] = labels
#        if self.add_mask:
#            targets["masks"] = masks
#        targets["image_id"] = image_id
#        #target["area"] = area
#        #target["iscrowd"] = iscrowd
#        # The transform funcs are based on Image liberary
#        if self._transforms is not None:
#            inputs, targets = self._transforms(inputs, targets)
#
#        if self.torchvision_format:
#            boxes = torch.from_numpy(targets['boxes'])
#            labels = torch.from_numpy(targets['labels'])
#            img = to_tensor(inputs['data'])
#            targets["boxes"] = boxes
#            targets["labels"] = labels 
#            if self.add_mask:
#                targets["masks"] = torch.from_numpy(targets['masks'])
#            return img, targets
#        return inputs, targets

