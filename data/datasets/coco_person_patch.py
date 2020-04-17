import os
import numpy as np
import pickle
from PIL import Image
from matplotlib import pyplot as plt

from .coco_person import COCOPersonDataset
from ..gaussian_tools import gaussian_radius, draw_umich_gaussian
from ..my_gaussian_tools import ellipse_gaussian_radius, draw_ellipse_gaussian
from ..transforms import ToTensor, Normalize, Compose

class COCOPersonPatchDataset(COCOPersonDataset):
    def __init__( self, root, anno, part, transforms=None):
        super(COCOPersonDataset,self).__init__( root, transforms )
        self._part = part
        self.class_num = 1

        # load annotations
        with open(anno, 'rb') as f:
            image_all = pickle.load(f)
            self._images = image_all[part] 

        # filter out the patches
        self.filter_the_person_patch()
        self.convert_boxes()

    def filter_the_person_patch(self):
        self.ind_map = []
        for i, image in enumerate(self._images):
            for j, obj in enumerate(image['objects']):
                if obj['category_id']==1 and min(obj['bbox'][2], obj['bbox'][3])>64:
                    self.ind_map.append((i, j))


    def convert_boxes(self):
        for image in self._images:
            for obj in image['objects']:
                obj['bbox'][2] += obj['bbox'][0]
                obj['bbox'][3] += obj['bbox'][1]

    def __len__(self):
        return len(self.ind_map)

    def __getitem__(self, idx):
        im_ind, box_ind = self.ind_map[idx]
        image = self._images[im_ind]

        # Load image
        img_path = os.path.join(self._root, self._part, image['file_name'] )
        image_id=image['id']
        img = Image.open(img_path).convert('RGB')
        #ori_image = img.copy()

        # Load labels
        human_box = image['objects'][box_ind]['bbox'].copy()
        ## convert bbox from xywh to xyxy
        #human_box[2]+=human_box[0]
        #human_box[3]+=human_box[1]

        #plt.figure()
        #plt.imshow(img)
        img = img.crop(human_box)

        inputs = {}
        inputs['data'] = img

        targets = {}
        #targets["human_box"] = human_box
        targets["labels"] = np.array(0) 
        targets["image_id"] = image_id
        #target["area"] = area
        #target["iscrowd"] = iscrowd
        if self._transforms is not None:
            inputs, targets = self._transforms(inputs, targets)

        return inputs, targets