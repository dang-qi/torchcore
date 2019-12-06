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
        #self.img_size = image_size
        #self.multiscale = multiscale
        #self.min_size = self.img_size - 3 * 32
        #self.max_size = self.img_size + 3 * 32
        #self.augment = augment
        self.box_extend = box_extend
        self._remove_image_with_bad_box()
        self._convert_human_box(extend=box_extend)
        # convert all the box cordinates to relative cordinate with human box
        self._convert_all_box_cord()

    def __getitem__(self, idx):
        # ---------
        #  Image
        # ---------
        image = self.images[idx]
        im_path = os.path.join(self._root, image['file_name'])

        # Extract cropped image as PyTorch tensor
        box = image['human_box'] if self.use_revised_box else image['human_box_det']
        im = Image.open(im_path).convert('RGB')
        im = im.crop(box)
        
        # ---------
        #  Label
        # ---------

        if len(image['objects'])>0:
            boxes = np.zeros((len(image['objects']), 4), dtype=float)
            categories = np.zeros((len(image['objects']), 1), dtype=float)
            for i, a_object in enumerate(image['objects']):
                categories[i] = a_object['category_id'] - 1
                boxes[i] = a_object['bbox']


        inputs = {}
        inputs['data'] = im

        targets = {}
        targets['boxes'] = boxes
        targets['im_id'] = image['id']

        if self._transforms is not None:
            inputs, targets = self._transforms(inputs, targets)
        ## Apply augmentations
        #if self.augment:
        #    if np.random.random() < 0.5:
        #        img, targets['origin'] = horisontal_flip(img, targets['origin'])

        return inputs, targets

    def collate_fn(self, batch):
        inputs, targets = list(zip(*batch))
        if targets[0] is None:
            targets = None
        else:
            # Remove empty placeholder targets
            targets = [target for target in targets if target is not None]
            # Add sample index to targets
            for i, target in enumerate(targets):
                boxes = target['origin']
                boxes[:, 0] = i
            boxes = torch.cat(tuple([target['origin'] for target in targets]), 0)
            pads = [target['pad'] for target in targets]
            scales = [target['scale'] for target in targets]
            im_ids = [target['im_id'] for target in targets]

            targets = {}
            targets['origin'] = boxes
            targets['pad'] = pads
            targets['scale'] = scales
            targets['im_id'] = im_ids
        ## Selects new image size every tenth batch
        #if self.multiscale and self.batch_count % 10 == 0:
        #    self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.images)

    def _convert_all_box_cord(self):
        ''' convert all the box to xyxy and convert the boxes to cord relative to human box,
            remove the boxes out of the human box
        '''
        for image in self.images:
            for obj in image['objects']:
                human_box = image['human_box'] if self.use_revised_box else image['human_box_det']
                obj['bbox'] = _convert_box_cord(human_box, _to_xyxy(obj['bbox']))
                if obj['bbox'] is None:
                    image['objects'].remove(obj)

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

    def get_human_boxes(self):
        id_box_map = {}
        for image in self.images:
            im_id = image['id']
            human_box = image['human_box'] if self.use_revised_box else image['human_box_det']
            id_box_map[im_id] = human_box
        return id_box_map

