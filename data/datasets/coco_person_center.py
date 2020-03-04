import os
import numpy as np
import pickle
from PIL import Image

from .coco_person import COCOPersonDataset
from ..gaussian_tools import gaussian_radius, draw_umich_gaussian
from ..my_gaussian_tools import ellipse_gaussian_radius, draw_ellipse_gaussian
from ..transforms import ToTensor, Normalize, Compose

class COCOPersonCenterDataset(COCOPersonDataset):
    def __init__( self, root, anno, part, training=True, transforms=None, max_obj=128, down_stride=4):
        super(COCOPersonDataset,self).__init__( root, transforms )
        self._part = part
        self._max_obj = max_obj # used for ground truth batching 
        self.class_num = 1
        self.down_stride = down_stride
        self.training = training

        # load annotations
        with open(anno, 'rb') as f:
            self._images = pickle.load(f)[part] 

    def __getitem__(self, idx):
        image = self._images[idx]

        # Load image
        img_path = os.path.join(self._root, self._part, image['file_name'] )
        image_id=image['id']
        img = Image.open(img_path).convert('RGB')
        #ori_image = img.copy()

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

        # for debug
        inputs['cropped_im'] = np.asarray(inputs['data'].copy())


        # Generate heatmaps, offset, width_hight map, etc.
        class_num = self.class_num
        width, height = inputs['data'].size
        width_out = width // self.down_stride
        height_out = height // self.down_stride

        transforms_post = Compose([ToTensor(), Normalize()])
        inputs, _ = transforms_post(inputs )
        if not self.training:
            return inputs, targets

        # filter out the valid boxes
        boxes_out = targets['boxes']
        valid_box_ind = np.where(np.logical_and(boxes_out[:,3]>boxes_out[:,1],boxes_out[:,2]>boxes_out[:,0]))
        valid_boxes = targets['boxes'][valid_box_ind]
        valid_labels = labels[valid_box_ind]
        targets['cat_labels'] = valid_labels
        del targets['cat_labels']
        del targets['boxes']

        #if (valid_boxes[:,2] / self.down_stride>=width_out).any():
        #    print('x2 wrong', valid_boxes)
        #if (valid_boxes[:,3] / self.down_stride>=height_out).any():
        #    print('y2 wrong', valid_boxes)

        center_x = (valid_boxes[:,0] + valid_boxes[:,2])/2 
        center_y = (valid_boxes[:,1] + valid_boxes[:,3])/2 
        center_x_out = center_x / self.down_stride
        center_y_out = center_y / self.down_stride
        boxes_w = valid_boxes[:,2] - valid_boxes[:,0]
        boxes_h = valid_boxes[:,3] - valid_boxes[:,1]

        #heatmaps = generate_gaussian_heatmap(class_num, width_out, height_out, center_x_out, center_y_out, boxes_w, boxes_h, labels )

        heatmaps = generate_ellipse_gaussian_heatmap(class_num, width_out ,height_out, center_x_out, center_y_out, boxes_w, boxes_h, labels)
    
        offset = generate_offset(center_x_out, center_y_out, self._max_obj)
        width_height = generate_width_height(valid_boxes, self._max_obj)
        #offset_map = generate_offset_map(center_x_out, center_y_out, height_out, width_out)
        #width_height_map = generate_width_height_map(valid_boxes, center_x_out, center_y_out, height_out, width_out)
        mask = generate_mask(center_x_out, center_y_out, width_out, height_out)
        #mask = np.zeros(self._max_obj, dtype=np.uint8)
        #mask[:len(valid_boxes)] = 1

        ind = np.zeros(self._max_obj, dtype=int)
        ind = generate_ind(ind, center_x_out, center_y_out, width_out)
        ind_mask = np.zeros(self._max_obj, dtype=int)
        ind_mask[:len(center_x_out)] = 1

        targets['heatmap'] = heatmaps
        targets['offset'] = offset
        #targets['offset_map'] = offset_map
        targets['width_height'] = width_height
        #targets['width_height_map'] = width_height_map
        targets['ind'] = ind
        targets['ind_mask'] = ind_mask
        #targets['offset'] = offset_map
        #targets['width_height'] = width_height_map
        targets['mask'] = mask

        return inputs, targets

def generate_ind(ind, cx, cy, w ):
    for i, (x, y ) in enumerate(zip(cx.astype(int), cy.astype(int))):
        ind[i] = y*w + x
    return ind


def generate_mask(center_x, center_y, width, height):
    mask = np.zeros((2, height, width), dtype=np.bool)
    for x,y in zip(center_x.astype(int), center_y.astype(int)):
        mask[:,y,x] = True
    return mask

def generate_gaussian_heatmap(class_num, width, height, center_x, center_y, boxes_w, boxes_h, labels ):
    heatmaps = np.zeros((class_num, height, width), dtype=np.float32)
    if len(center_x)== 0:
        return heatmaps
    center_x = center_x.astype(int)
    center_y = center_y.astype(int)
    for x, y, label, w, h in zip(center_x, center_y, labels, boxes_w, boxes_h):
        radius = get_gaussian_radius((h,w))
        print(radius)
        radius = max(0, int(radius))
        #print('x:{}  y:{}  radius:{}'.format(x,y,radius))
        draw_gaussian_heatmap(heatmaps[label-1], (x,y), radius )

    return heatmaps

def generate_ellipse_gaussian_heatmap(class_num, width, height, center_x, center_y, boxes_w, boxes_h, labels ):
    heatmaps = np.zeros((class_num, height, width), dtype=np.float32)
    if len(center_x)== 0:
        return heatmaps
    center_x = center_x.astype(int)
    center_y = center_y.astype(int)
    for x, y, label, w, h in zip(center_x, center_y, labels, boxes_w, boxes_h):
        r_w, r_h = ellipse_gaussian_radius(w, h, IoU=0.7)
        #set the maximum radius while keep the ratio
        r_max = max(r_w, r_h)
        if r_max > 5:
            scale = r_max / 5
            r_w = r_w / scale
            r_h = r_h / scale
        sigma_x = r_w/3
        sigma_y = r_h/3
        r_w = max(0, int(r_w))
        r_h = max(0, int(r_h))
        #print('rw: {}, rh: {}'.format(r_w, r_h))
        #print('x:{}  y:{}  radius:{}'.format(x,y,radius))
        draw_ellipse_gaussian(heatmaps[label-1], x, y, r_w, r_h, sigma_x, sigma_y)
        #draw_gaussian_heatmap(heatmaps[label-1], (x,y), radius )

    return heatmaps
def generate_offset(center_x, center_y, max_length):
    real_len = len(center_x)
    assert real_len <= max_length
    #offset = np.zeros((max_length, 2),dtype=np.float32)
    offset = np.zeros((2, max_length),dtype=np.float32)
    if real_len == 0:
        return offset
    offset[0, :real_len] = center_x - center_x.astype(int)
    offset[1, :real_len] = center_y - center_y.astype(int)
    return offset

def generate_offset_map(center_x, center_y, height, width):
    offset_map = np.zeros((2, height, width), dtype=np.float32)
    center_x_int = center_x.astype(int)
    center_y_int = center_y.astype(int)
    offset_x = center_x - center_x_int
    offset_y = center_y - center_y_int
    if (center_x_int >= width).any():
        print('center_x:{}, width:{}'.format(center_x_int, width))
    if (center_y_int >= height).any():
        print('center_y:{}, height:{}'.format(center_y_int, height))
    for x, y, off_x, off_y in zip(center_x_int, center_y_int, offset_x, offset_y):
        offset_map[:, y, x] = (off_x, off_y)
    return offset_map
    
def generate_width_height_map(boxes, center_x, center_y, height, width):
    width_height_map = np.zeros((2, height, width), dtype=np.float32)
    center_x = center_x.astype(int)
    center_y = center_y.astype(int)
    width = boxes[:,2]-boxes[:,0]
    height = boxes[:,3]-boxes[:,1]
    for x, y, w, h in zip(center_x, center_y, width, height):
        width_height_map[:, y, x] = (w, h)
    return width_height_map

def generate_width_height(boxes, max_length):
    real_len = len(boxes)
    assert real_len <= max_length
    width_height = np.zeros((2, max_length), dtype=np.float32)
    if real_len == 0:
        return width_height
    width_height[0, :real_len] = boxes[:,2]-boxes[:,0]
    width_height[1, :real_len] = boxes[:,3]-boxes[:,1]
    return width_height

def draw_gaussian_heatmap(heatmap, center, radius):
    draw_umich_gaussian(heatmap, center, radius)

def get_gaussian_radius(size):
    radius = gaussian_radius(size)
    return radius
