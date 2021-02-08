import numpy as np
import pickle
from .dataset_new import Dataset
from PIL import Image
import os

class ModanetDataset(Dataset):
    '''Modanet dataset'''
    def __init__( self, root, anno, part, transforms=None, xyxy=True, debug=False ):
        super().__init__( root, transforms )
        self._part = part

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
        img_path = os.path.join(self._root, 'train', image['file_name'] )
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

        inputs = {}
        inputs['data'] = img
        if self.debug:
            inputs['ori_image'] = ori_image

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

    def convert_to_xyxy(self):
        for image in self._images:
            for obj in image['objects']:
                obj['bbox'][2]+=obj['bbox'][0]
                obj['bbox'][3]+=obj['bbox'][1]
