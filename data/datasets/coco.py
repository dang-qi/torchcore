import numpy as np
import torch
import pickle
from .dataset_new import Dataset
import os
from ..dataset_util import get_binary_mask
from torchvision.transforms.functional import to_tensor

from .build import DATASET_REG

@DATASET_REG.register(force=True)
class COCODataset(Dataset):
    '''COCO dataset'''
    def __init__( self, root, anno, part, transforms=None, debug=False, xyxy=True, torchvision_format=False, add_mask=False, first_n_subset=None, subcategory=None, map_id_to_continuous=True, backend='pillow',RGB=True):
        super().__init__( root=root, anno=anno, part=part, transforms=transforms )
        ## load annotations
        #with open(anno, 'rb') as f:
        #    self._images = pickle.load(f)[part] 
        self.remove_wrong_labels()
        self.debug = debug
        self.xyxy = xyxy
        self.torchvision_format = torchvision_format
        self.add_mask = add_mask
        self.backend = backend
        self.RGB = RGB
        assert backend in ['pillow', 'opencv']
        if xyxy:
            self.convert_to_xyxy()
        if subcategory is not None:
            self.set_category_subset(subcategory, ignore_other_category=True)
        if map_id_to_continuous:
            self.map_category_id_to_continous()
        if first_n_subset is not None:
            self.set_first_n_subset(first_n_subset)
        self._set_aspect_ratio_flag()

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]

        # Load image
        img_path = os.path.join(self._root, image['file_name'] )
        image_id=image['id']
        img = self._load_image(img_path, self.backend, self.RGB)
        if self.debug:
            ori_image = img.copy()

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
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)


        if self.add_mask:
            height = image['height']
            width = image['width']
            masks = [get_binary_mask(obj['segmentation'], height, width, use_compressed_rle=True) for obj in image['objects']]
            masks = np.array(masks, dtype=np.uint8)
        #images (list[Tensor]): images to be processed
        #targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        inputs = {}
        inputs['data'] = img

        targets = {}
        targets["boxes"] = boxes
        targets["labels"] = labels 
        targets["image_id"] = image_id
        #target["area"] = area
        #target["iscrowd"] = iscrowd
        if self.add_mask:
            targets["masks"] = masks
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

        if self.debug:
            #inputs['ori_image'] = inputs['data'].copy()
            inputs['ori_image'] = np.array(ori_image)

        return inputs, targets



    def remove_wrong_labels(self):
        '''Remove the invalid object boxes and images has no objects'''
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
    
