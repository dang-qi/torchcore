import numpy as np
import pickle
import torch
from .dataset_new import Dataset
from PIL import Image
import os
from ..dataset_util import get_binary_mask
from torchvision.transforms.functional import to_tensor
import copy

class FashionPediaDataset(Dataset):
    '''FashionPedia dataset'''
    def __init__( self, root, anno, part, transforms=None, xyxy=True, debug=False, torchvision_format=False, add_mask=False ):
        super().__init__( root, transforms )
        self._part = part
        folder_dict = {'train':'train', 'val':'test', 'test':'test'}
        self._folder = folder_dict[part]

        self.torchvision_format = torchvision_format
        self.add_mask = add_mask

        # load annotations
        with open(anno, 'rb') as f:
            self._images = pickle.load(f)[part] 
        self.xyxy = xyxy
        if xyxy:
            self.convert_to_xyxy()
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
        for obj in image['objects']:
            boxes.append(obj['bbox'])
            labels.append(obj['category_id'])
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)


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

    def generate_id_dict(self):
        print('generating id dict...')
        self.id_dict = dict()
        for i,im in enumerate(self._images):
            im_id = im['id']
            self.id_dict[im_id] = i

    def extract_by_id(self, im_id):
        if not hasattr(self,'id_dict'):
            self.generate_id_dict()
        index = self.id_dict[im_id]
        return self.__getitem__(index)

    def set_category_subset(self, cat_id, ignore_other_category=True):
        '''
        cat_id: int or list
        make the dataset the subset with only some of the category
        '''
        if not hasattr(self, 'category_index_dict'):
            self.generate_cat_dict()
        if isinstance(cat_id, int):
            cat_id = [cat_id]
        the_sets = [set(self.category_index_dict[i]) for i in cat_id]
        im_indexs = set.union(set(), *the_sets)

        if ignore_other_category:
            cat_id = set(cat_id)
            self._images = []
            for i in im_indexs:
                im = copy.deepcopy(self._original_images[i])
                im_objs = im['objects']
                im['objects'] = []
                for obj in im_objs:
                    if obj['category_id'] in cat_id:
                        im['objects'].append(obj)
                self._images.append(im)
        else:
            self._images = [self._original_images[i] for i in im_indexs]

    def generate_cat_dict(self):
        if hasattr(self, 'category_index_dict'):
            print('category_index_dict has been generated')
            return

        self.category_im_dict = {}
        self.category_index_dict = {}
        if not hasattr(self, '_original_images'):
            self._original_images = self._images
        
        for i,im in enumerate(self._original_images):
            im_cat_id = set()
            for obj in im['objects']:
                cat_id = obj['category_id']
                if cat_id in im_cat_id:
                    continue
                else:
                    im_cat_id.add(cat_id)
                if cat_id not in self.category_im_dict:
                    self.category_im_dict[cat_id] = []
                    self.category_index_dict[cat_id] = []
                else:
                    self.category_im_dict[cat_id].append(im)
                    self.category_index_dict[cat_id].append(i)

