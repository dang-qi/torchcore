import numpy as np
import pickle
from .dataset_new import Dataset
from PIL import Image
import os
from torchvision.transforms.functional import to_tensor
import torch

class COCOTorchVisionDataset(Dataset):
    '''COCO dataset'''
    def __init__( self, root, anno, part, transforms=None  ):
        super().__init__( root, transforms=transforms )
        self._part = part

        # load annotations
        with open(anno, 'rb') as f:
            self._images = pickle.load(f)[part] 
        self.remove_wrong_labels()
        self.map_category_id_to_continous()
        #if xyxy:
        self.convert_to_xyxy()

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]

        # Load image
        img_path = os.path.join(self._root, self._part, image['file_name'] )
        image_id=image['id']
        img = Image.open(img_path).convert('RGB')
        ##if self.debug:
        ##    ori_image = img.copy()

        # Load targets
        boxes = []
        labels = []
        for obj in image['objects']:
            # WARNING:
            # DO NOT CHANGE objects here
            # IT WILL change the data every time and you will keep getting wrong result
            #if self.xyxy:
            #    # convert the bbox from xywh to xyxy
            #    obj['bbox'][2]+=obj['bbox'][0]
            #    obj['bbox'][3]+=obj['bbox'][1]
            boxes.append(obj['bbox'])
            labels.append(obj['category_id'])
        #boxes = np.array(boxes, dtype=np.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)
        #labels = np.array(labels, dtype=np.int64)

        #images (list[Tensor]): images to be processed
        #targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        inputs = {}
        inputs['data'] = img

        targets = {}
        targets["boxes"] = boxes
        targets["labels"] = labels 
        #target["masks"] = masks
        targets["image_id"] = image_id
        #target["area"] = area
        #target["iscrowd"] = iscrowd
        if self._transforms is not None:
            inputs, targets = self._transforms(inputs, targets)
        img = to_tensor(inputs['data'])

        #if self.debug:
        #    inputs['ori_image'] = inputs['data'].copy()

        return img, targets

    def map_category_id_to_continous(self):
        id_set = set()
        for image in self._images:
            for obj in image['objects']:
                id_set.add(obj['category_id'])
        ids = sorted(list(id_set))
        id_map = {aid:i+1 for i, aid in enumerate(ids)}
        #print(id_map)
        for image in self._images:
            for obj in image['objects']:
                obj['category_id'] = id_map[obj['category_id']]

    def remove_wrong_labels(self):
        i = 0
        wrong_im_num=0
        while i < len(self._images):
            image = self._images[i]
            j = 0
            while j < len(image['objects']):
                obj = image['objects'][j]
                if obj['bbox'][2]<= 0 or obj['bbox'][3]<=0:
                    #print('delete one wrong object: {}'.format(image['objects'][j]['bbox']))
                    del image['objects'][j]
                else:
                    j += 1
            if len(image['objects']) == 0:
                #print('delete image {}'.format(image['id']))
                wrong_im_num += 1
                del self._images[i]
            else:
                i += 1
        if wrong_im_num > 0:
            print('{} images are deleted.'.format(wrong_im_num))
    
    def convert_to_xyxy(self):
        for image in self._images:
            for obj in image['objects']:
                obj['bbox'][2]+=obj['bbox'][0]
                obj['bbox'][3]+=obj['bbox'][1]
