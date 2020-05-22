import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2

from torch.utils.data import Dataset

class PatchDataset():
    def __init__( self, images, boxes, transforms=None, single_mode=False):
        if single_mode:
            self._images = [images]
            self._boxes = [boxes]
        else:
            self._images = images
            self._boxes = boxes
        self._transforms = transforms
        self.gen_ind_map()

    def gen_ind_map(self):
        ind_map = {}
        box_num = 0
        for i, boxes in enumerate(self._boxes):
            for j, _ in enumerate(boxes):
                ind_map[box_num] = (i, j)
                box_num += 1
        self.ind_map = ind_map

    def __len__(self):
        return len(self.ind_map)

    def __getitem__(self, idx):
        im_ind, box_ind = self.ind_map[idx]
        image = self._images[im_ind]

        # Load labels
        human_box = self._boxes[im_ind][box_ind]
        ## convert bbox from xywh to xyxy
        #human_box[2]+=human_box[0]
        #human_box[3]+=human_box[1]

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) )
        img = image.crop(human_box)

        inputs = {}
        inputs['data'] = img

        targets = None
        ##targets["human_box"] = human_box
        #targets["labels"] = np.array(0) 
        #targets["image_id"] = image_id
        #target["area"] = area
        #target["iscrowd"] = iscrowd
        if self._transforms is not None:
            inputs, targets = self._transforms(inputs, targets)
        targets = []

        return inputs, targets