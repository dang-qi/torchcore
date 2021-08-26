import numpy as np
import pickle
import torch
from .dataset_new import Dataset
from PIL import Image
import os
from ..dataset_util import get_binary_mask
from torchvision.transforms.functional import to_tensor

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

