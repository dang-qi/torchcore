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

        # load annotations
        self._images = h5py.File(h5_path, 'r')[part]
        self._keys = self._images.keys()
        #with open(anno, 'rb') as f:
        #    self._images = pickle.load(f)[part] 
        if transforms is None:
            self._transforms = Normalize()

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[self._keys[idx]]

        img = np.array(image['data'])
        boxes = np.array(image['bbox'])
        labels = np.array(image['category_id'])
        image_id = np.array(image['image_id'])
        scale = np.array(image['scale'])
        mirrored = image['mirrored'][()]

        #images (list[Tensor]): images to be processed
        #targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        inputs = {}
        inputs['data'] = torch.from_numpy(img)
        inputs['scale'] = scale
        inputs['mirrored'] = mirrored

        targets = {}
        targets["boxes"] = boxes
        targets["cat_labels"] = labels
        #target["masks"] = masks
        targets["image_id"] = image_id
        #target["area"] = area
        #target["iscrowd"] = iscrowd
        if self._transforms is not None:
            inputs, targets = self._transforms(inputs, targets)

        return inputs, targets

