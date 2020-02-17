import os
import numpy as np
import pickle
from PIL import Image

from .coco_person import COCOPersonDataset
from ..gaussian_tools import gaussian_radius, draw_umich_gaussian
from ..transforms import ToTensor, Normalize, Compose

class COCOPersonCenterDataset(COCOPersonDataset):
    def __init__( self, root, anno, part, transforms=None, max_obj=128, down_stride=4):
        super(COCOPersonDataset,self).__init__( root, transforms )
        self._part = part
        self._max_obj = max_obj # used for ground truth batching 
        self.class_num = 1
        self.down_stride = down_stride

        # load annotations
        with open(anno, 'rb') as f:
            self._images = pickle.load(f)[part] 

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

        # Generate heatmaps, offset, width_hight map, etc.
        class_num = self.class_num
        width, height = inputs['data'].size
        width_out = width // self.down_stride
        height_out = height // self.down_stride

        # filter out the valid boxes
        valid_box_ind = np.where(np.logical_and(boxes[:,3]>boxes[:,1],boxes[:,2]>boxes[:,0]))
        valid_boxes = targets['boxes'][valid_box_ind]
        valid_labels = labels[valid_box_ind]
        targets['cat_labels'] = valid_labels
        del targets['cat_labels']
        del targets['boxes']

        center_x = (valid_boxes[:,0] + valid_boxes[:,2])/2 
        center_y = (valid_boxes[:,1] + valid_boxes[:,3])/2 
        center_x_out = center_x / self.down_stride
        center_y_out = center_y / self.down_stride
        boxes_w = valid_boxes[:,2] - valid_boxes[:,0]
        boxes_h = valid_boxes[:,3] - valid_boxes[:,1]

        heatmaps = generate_gaussian_heatmap(class_num, width_out, height_out, center_x_out, center_y_out, boxes_w, boxes_h, labels )

        offset = generate_offset(center_x_out, center_y_out, self._max_obj)
        width_height = generate_width_height(valid_boxes, self._max_obj)
        mask = np.zeros(self._max_obj, dtype=np.uint8)
        mask[:len(valid_boxes)] = 1

        ind = np.zeros(self._max_obj, dtype=int)

        transforms_post = Compose([ToTensor(), Normalize()])
        inputs, _ = transforms_post(inputs )

        targets['heatmap'] = heatmaps
        targets['offset'] = offset
        targets['width_height'] = width_height
        targets['mask'] = mask

        return inputs, targets

def generate_gaussian_heatmap(class_num, width, height, center_x, center_y, boxes_w, boxes_h, labels ):
    heatmaps = np.zeros((class_num, height, width), dtype=np.float32)
    if len(center_x)== 0:
        return heatmaps
    center_x = center_x.astype(int)
    center_y = center_y.astype(int)
    for x, y, label, w, h in zip(center_x, center_y, labels, boxes_w, boxes_h):
        radius = get_gaussian_radius((h,w))
        radius = max(0, int(radius))
        #print('x:{}  y:{}  radius:{}'.format(x,y,radius))
        draw_gaussian_heatmap(heatmaps[label-1], (x,y), radius )

    return heatmaps

def generate_offset(center_x, center_y, max_length):
    real_len = len(center_x)
    assert real_len <= max_length
    offset = np.zeros((max_length, 2),dtype=np.float32)
    if real_len == 0:
        return offset
    offset[:real_len,0] = center_x - center_x.astype(int)
    offset[:real_len,1] = center_y - center_y.astype(int)
    return offset

def generate_width_height(boxes, max_length):
    real_len = len(boxes)
    assert real_len <= max_length
    width_height = np.zeros((max_length, 2), dtype=np.float32)
    if real_len == 0:
        return width_height
    width_height[:real_len,0] = boxes[:,2]-boxes[:,0]
    width_height[:real_len,1] = boxes[:,3]-boxes[:,1]
    return width_height

def draw_gaussian_heatmap(heatmap, center, radius):
    draw_umich_gaussian(heatmap, center, radius)

def get_gaussian_radius(size):
    radius = gaussian_radius(size)
    return radius
