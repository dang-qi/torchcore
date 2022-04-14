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

@DATASET_REG.register(force=True)
class FashionPediaDataset(Dataset):
    '''FashionPedia dataset'''
    VALID_CAT_ID_WITH_HAS_ATTRIBUTES=[32, 31, 28, 0, 10, 33, 6, 9, 1, 29, 4, 8, 7, 11, 2, 3, 5, 12,]
    def __init__( self, root, anno, part, transforms=None, xyxy=True, debug=False, torchvision_format=False, add_mask=False, sub_category=None, add_attributes=False ):
        super().__init__( root, anno=anno, part=part, transforms=transforms )
        self._part = part
        folder_dict = {'train':'train', 'val':'test', 'test':'test'}
        self._folder = folder_dict[part]

        self.torchvision_format = torchvision_format
        self.add_mask = add_mask
        self.add_attibutes = add_attributes
        if add_attributes:
            pass
            #self.remove_wrong_attri_by_category()

        ## load annotations
        #with open(anno, 'rb') as f:
        #    self._images = pickle.load(f)[part] 
        self.xyxy = xyxy
        if xyxy:
            self.convert_to_xyxy()
        if sub_category is not None:
            self.set_category_subset(sub_category, ignore_other_category=True)
        self._set_aspect_ratio_flag()
        self.debug = debug

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]

        # Load image
        img_path = os.path.join(self._root, self._folder, image['file_name'] )
        image_id=image['id']
        img = Image.open(img_path).convert('RGB')
        ori_image = img

        # Load targets
        boxes = []
        labels = []

        attributes = []
        for obj in image['objects']:
            boxes.append(obj['bbox'])
            labels.append(obj['category_id'])
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        if self.add_attibutes:
            attributes = []
            for obj in image['objects']:
                attributes.append(obj['attribute_ids'])


        if self.add_mask:
            height = image['height']
            width = image['width']
            masks = [get_binary_mask(obj['segmentation'], height, width, use_compressed_rle=True) for obj in image['objects']]
            masks = np.array(masks, dtype=np.uint8)

        inputs = {}
        inputs['data'] = img
        if self.debug:
            inputs['ori_image'] = ori_image

        targets = {}
        targets["boxes"] = boxes
        targets["cat_labels"] = labels 
        targets["labels"] = labels
        if self.add_attibutes:
            targets['attributes'] = attributes
        if self.add_mask:
            targets["masks"] = masks
        targets["image_id"] = image_id
        #target["area"] = area
        #target["iscrowd"] = iscrowd
        # The transform funcs are based on Image liberary
        if self._transforms is not None:
            inputs, targets = self._transforms(inputs, targets)

        if self.torchvision_format:
            boxes = torch.from_numpy(targets['boxes'])
            labels = torch.from_numpy(targets['labels'])
            img = to_tensor(inputs['data'])
            targets["boxes"] = boxes
            targets["labels"] = labels 
            if self.add_mask:
                targets["masks"] = torch.from_numpy(targets['masks'])
            return img, targets
        return inputs, targets

    def convert_to_xyxy(self):
        for image in self._images:
            for obj in image['objects']:
                obj['bbox'][2]+=obj['bbox'][0]
                obj['bbox'][3]+=obj['bbox'][1]

    def remove_wrong_attri_by_category(self):
        wrong_label= 0
        for image in self._images:
            for obj in image['objects']:
                if obj['category_id'] not in self.VALID_CAT_ID_WITH_HAS_ATTRIBUTES:
                    if len(obj['attribute_ids'])!=0:
                        wrong_label+=1
                    obj['attribute_ids'] = []
                else:
                    if len(obj['attribute_ids']) == 0:
                        print(obj)
        print('wrong attribute lable number is {}'.format(wrong_label))

    def get_wrong_attri_sample(self):
        inds = np.arange(len(self))
        np.random.shuffle(inds)
        for i in inds:
            image = self._images[i]
            for obj in image['objects']:
                if obj['category_id'] not in self.VALID_CAT_ID_WITH_HAS_ATTRIBUTES:
                    if len(obj['attribute_ids'])>1:
                        print(obj)
                        return self[i], obj

    @property
    def category_id_name_dict(self):
        dict = {0: 'shirt, blouse',
                1: 'top, t-shirt, sweatshirt',
                2: 'sweater',
                3: 'cardigan',
                4: 'jacket',
                5: 'vest',
                6: 'pants',
                7: 'shorts',
                8: 'skirt',
                9: 'coat',
                10: 'dress',
                11: 'jumpsuit',
                12: 'cape',
                13: 'glasses',
                14: 'hat',
                15: 'headband, head covering, hair accessory',
                16: 'tie',
                17: 'glove',
                18: 'watch',
                19: 'belt',
                20: 'leg warmer',
                21: 'tights, stockings',
                22: 'sock',
                23: 'shoe',
                24: 'bag, wallet',
                25: 'scarf',
                26: 'umbrella',
                27: 'hood',
                28: 'collar',
                29: 'lapel',
                30: 'epaulette',
                31: 'sleeve',
                32: 'pocket',
                33: 'neckline',
                34: 'buckle',
                35: 'zipper',
                36: 'applique',
                37: 'bead',
                38: 'bow',
                39: 'flower',
                40: 'fringe',
                41: 'ribbon',
                42: 'rivet',
                43: 'ruffle',
                44: 'sequin',
                45: 'tassel'}


