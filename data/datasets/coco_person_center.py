import os
import numpy as np
from PIL import Image
from .coco_person import COCOPersonDataset


class COCOPersonCenterDataset(COCOPersonDataset):
    def __getitem__(self, idx):
        image = self._images[idx]

        # Load image
        img_path = os.path.join(self._root, self._part, image['file_name'] )
        image_id=image['id']
        img = Image.open(img_path).convert('RGB')
        ori_image = img.copy()

        # Load labels
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
        center_x = (boxes[:,0]+boxes[:,2]) / 2
        center_y = (boxes[:,1]+boxes[:,3]) / 2


        inputs = {}
        inputs['data'] = img

        targets = {}
        targets["boxes"] = boxes
        targets["cat_labels"] = labels 
        #target["masks"] = masks
        targets["image_id"] = image_id
        #target["area"] = area
        #target["iscrowd"] = iscrowd
        if self._transforms is not None:
            inputs, targets = self._transforms(inputs, targets)

        # Generate heatmaps for 

        return inputs, targets
