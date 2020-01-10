from .dataset_new import Dataset
import pickle
from PIL import Image
from PIL.ImageDraw import Draw
import os
import numpy as np
import torch
import random

def _convert_box_cord(human_box, box):
    '''convert the origianl box with relative cord of human box,
       the box out of the human box will be cropped
    '''
    x1, y1, x2, y2 = box
    hx1, hy1, hx2, hy2 = human_box
    hw1 = hx2-hx1
    hh1 = hy2-hy1

    x1 = x1 - hx1
    y1 = y1 - hy1
    x2 = x2 - hx1
    y2 = y2 - hy1

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(hw1-1, x2)
    y2 = min(hh1-1, y2)

    if x1 >= x2 or y1>= y2:
        return None
    else:
        return [x1, y1, x2, y2]

def _to_xyxy(box):
    box[2] = box[0] + box[2]
    box[3] = box[1] + box[3]
    return box


class ModanetHumanDataset(Dataset):
    def __init__(self, anno_root, image_root, part, transforms=None, use_revised_box=True, box_extend=0.0):
        super().__init__( image_root, transforms )
        with open(anno_root, 'rb') as f:
            self.images = pickle.load(f)[part]
        self.use_revised_box=use_revised_box
        self.box_extend = box_extend
        self._part = part
        self._remove_image_with_bad_box()
        self._convert_human_box(extend=box_extend)
        # convert all the box cordinates to relative cordinate with human box
        self._convert_all_box_cord()
        # remove the images without valid human
        #if self._part =='train':
        self._remove_empty_image()

    def __getitem__(self, idx):
        # ---------
        #  Image
        # ---------
        image = self.images[idx]
        im_path = os.path.join(self._root, 'train', image['file_name'])

        # Extract cropped image as PyTorch tensor
        box = image['human_box'] if self.use_revised_box else image['human_box_det']
        im = Image.open(im_path).convert('RGB')
        im = im.crop(box)
        
        # ---------
        #  Label
        # ---------

        boxes = np.zeros([], dtype=np.float32)
        categories = np.zeros([], dtype=np.int64)
        if len(image['objects'])>0:
            boxes = np.zeros((len(image['objects']), 4), dtype=np.float32)
            categories = np.zeros((len(image['objects'])), dtype=np.int64)
            for i, a_object in enumerate(image['objects']):
                categories[i] = a_object['category_id'] 
                boxes[i] = a_object['bbox']


        inputs = {}
        inputs['data'] = im

        targets = {}
        targets['boxes'] = boxes
        targets['image_id'] = image['id']
        targets['image_path'] = im_path
        targets['labels'] = categories

        if self._transforms is not None:
            inputs, targets = self._transforms(inputs, targets)

        return inputs, targets

    def __len__(self):
        return len(self.images)

    def _convert_all_box_cord(self):
        ''' convert all the box to xyxy and convert the boxes to cord relative to human box,
            remove the boxes out of the human box
        '''
        for image in self.images:
            for i in range(len(image['objects'])-1, -1, -1):
                obj = image['objects'][i]
                human_box = image['human_box'] if self.use_revised_box else image['human_box_det']
                obj['bbox'] = _convert_box_cord(human_box, _to_xyxy(obj['bbox']))
                if obj['bbox'] is None:
                    del image['objects'][i]

    def _has_bad_human_box(self, image):
            human_box = image['human_box'] if self.use_revised_box else image['human_box_det']
            if human_box == []:
                return True
            x1 = int(max(0, human_box[0]))
            y1 = int(max(0, human_box[1]))
            x2 = int(min(image['width']-1, human_box[2]))
            y2 = int(min(image['height']-1, human_box[3]))
            if x1>=x2 or y1>=y2:
                return True
            return False

    def _remove_image_with_bad_box(self):
        num_before = len(self.images)
        self.images = [image for image in self.images if not self._has_bad_human_box(image)]
        num_after = len(self.images)
        print("{} images have been removed because they have bad human detection!".format(num_before-num_after))

    def _convert_human_box(self, extend=0):
        '''Just make sure all the human box are inside the image'''
        for image in self.images:
            human_box = image['human_box'] if self.use_revised_box else image['human_box_det']
            x1 = int(max(0, human_box[0]*(1-extend)))
            y1 = int(max(0, human_box[1]*(1-extend)))
            x2 = int(min(image['width']-1, human_box[2]*(1+extend)))
            y2 = int(min(image['height']-1, human_box[3]*(1+extend)))

            if self.use_revised_box:
                image['human_box'] = [x1, y1, x2, y2]
            else:
                image['human_box_det'] = [x1, y1, x2, y2]

    def _remove_empty_image(self):
        num_before = len(self.images)
        self.images = [image for image in self.images if len(image['objects'])>0]
        num_after = len(self.images)
        print("{} images have been removed because they have no boxes!".format(num_before-num_after))

    def get_human_boxes(self):
        id_box_map = {}
        for image in self.images:
            im_id = image['id']
            human_box = image['human_box'] if self.use_revised_box else image['human_box_det']
            id_box_map[im_id] = human_box
        return id_box_map

