import numpy as np
import pickle
import torch
import h5py
from .dataset_new import Dataset
from PIL import Image
import os
from ..transforms import ToTensor, Normalize, Compose

class ModanetHDF5Dataset(Dataset):
    '''Modanet dataset'''
    def __init__( self, h5_path, part, transforms=None ):
        self._part = part
        self._h5_path = h5_path
        self.h5 = h5py.File(h5_path, 'r')
        self.count=1

        # load annotations
        self._images = self.h5[part]
        self._keys = list(self._images.keys())
        #with open(anno, 'rb') as f:
        #    self._images = pickle.load(f)[part] 
        if transforms is None:
            self._transforms = Compose([ToTensor(),Normalize()])

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[self._keys[idx]]

        img = np.array(image['data'])
        boxes = np.array(image['bbox']).astype(np.float32)
        crop_box = np.array(image['crop_box'])
        labels = np.array(image['category_id'])
        image_id = np.array(image['image_id'])
        scale = np.array(image['scale'])
        mirrored = image['mirrored'][()]
        h, w, c = img.shape

        #images (list[Tensor]): images to be processed
        #targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        inputs = {}
        inputs['data'] = img
        inputs['scale'] = scale
        inputs['mirrored'] = mirrored
        inputs['image_sizes'] = (h, w)

        targets = {}
        targets["boxes"] = boxes
        targets["cat_labels"] = labels
        #target["masks"] = masks
        targets["image_id"] = image_id
        targets['crop_box'] = crop_box
        #target["area"] = area
        #target["iscrowd"] = iscrowd
        if self._transforms is not None:
            inputs, targets = self._transforms(inputs, targets)

        self.count+=1
        if self.count % 100 == 0:
            self.h5.close()
            self.h5 = h5py.File(self._h5_path, 'r')
            self._images = self.h5[self._part]
            self.count = 1

        return inputs, targets

