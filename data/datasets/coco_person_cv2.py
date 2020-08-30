import numpy as np
import pickle
from .dataset_new import Dataset
import cv2
import os

class COCOPersonCV2Dataset(Dataset):
    '''COCO dataset only contarin person class'''
    def __init__( self, root, anno, part, transforms=None, test=False ):
        super().__init__( root, transforms )
        self._part = part
        self.test = test

        # load annotations
        with open(anno, 'rb') as f:
            self._images = pickle.load(f)[part] 

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]

        # Load image
        img_path = os.path.join(self._root, self._part, image['file_name'] )
        image_id=image['id']
        img = cv2.imread(img_path)
        #ori_image = img.copy()

        # Load targets
        boxes = []
        labels = []
        for obj in image['objects']:
            # convert the bbox from xywh to xyxy
            obj['bbox'][2]+=obj['bbox'][0]
            obj['bbox'][3]+=obj['bbox'][1]
            boxes.append(obj['bbox'])
            labels.append(obj['category_id'])
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        #images (list[Tensor]): images to be processed
        #targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        inputs = {}
        inputs['data'] = img
        inputs["image_id"] = image_id
        #inputs['ori_image'] = ori_image

        targets = {}
        targets["boxes"] = boxes
        targets["cat_labels"] = labels 
        #target["masks"] = masks
        targets["image_id"] = image_id
        #target["area"] = area
        #target["iscrowd"] = iscrowd
        if self._transforms is not None:
            if not self.test:
                inputs, targets = self._transforms(inputs, targets)
            else:
                inputs = self._transforms(inputs)

        if not self.test:
            return inputs, targets
        else:
            return inputs

