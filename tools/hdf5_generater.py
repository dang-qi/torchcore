import h5py
import os
import pickle
import numpy as np

from PIL import Image

from .visulize_tools import draw_plain_boxes
def gen_human_id_map(human_detections):
    human_map = {}
    for detection in human_detections:
        human_map[detection['image_id']] = detection['human_box']
    return human_map

def get_biggest_box(boxes):
    area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    max_ind = np.argmax(area)
    biggest_box = boxes[max_ind][:4]
    return biggest_box

def get_object_inside_box(objects, box):
    '''objects"[{'bbox':bbox(x,y,w,h), 'category_id':cat_id}]'''
    boxes = []
    category = []
    width = box[2] - box[0]
    height = box[3] - box[1]
    for obj in objects:
        bbox = obj['bbox']
        x, y, w, h = bbox
        bbox[0] = max(0, bbox[0]-box[0])
        bbox[1] = max(0, bbox[1]-box[1])
        bbox[2] = min(bbox[0]+w, width)
        bbox[3] = min(bbox[1]+h, height)
        if bbox[0]<bbox[2] and bbox[3]>bbox[1]:
            boxes.append(bbox)
            category.append(obj['category_id'])
    boxes = np.array(boxes)
    category = np.array(boxes)
    return boxes, category

def expand_box(box, expand_rate, im_w, im_h):
    assert expand_rate > -1
    b_w = box[2] - box[0]
    b_h = box[3] - box[1]
    
    exp_w = expand_rate / 2 * b_w
    exp_h = expand_rate / 2 * b_h 

    box[0] -= exp_w
    box[2] += exp_w
    box[1] -= exp_h
    box[3] += exp_h
    
    box[0] = max(0, box[0])
    box[1] = max(0, box[1])
    box[2] = min(box[2], im_w)
    box[3] = min(box[3], im_h)
    
    return box
    
def pad_box_by_ratio(box, ratio):
    '''ratio: height/width
       the returned box can have minus cords, 
       but it is designed for crop
    '''
    box_ratio = (box[3]-box[1])/(box[2]-box[0])
    if box_ratio == ratio:
        return box
    if box_ratio > ratio: # pad width
        out_width = int((box[3]-box[1])/ratio)
        pad_all = out_width - (box[2]-box[0])
        pad_left = int(pad_all/2)
        pad_right = pad_all - pad_left
        box[0] = box[0] - pad_left
        box[2] = box[2] + pad_right
    else:
        out_h = int((box[2]-box[0])*ratio)
        pad_all = out_h - (box[3]-box[1])
        pad_up = int((out_h-(box[3]-box[1]))/2)
        pad_down = out_h - pad_up
        box[1] = box[0] - pad_up
        box[3] = box[2] + pad_down
    return box

def generate_hdf5_patch(hdf5_path, part, human_detections, imageset, im_root, expand_rate=0):
    '''human detections:[{'image_id':int, 'human_box':ndarray}]
       human_boxes:n*6 (x1, y1, x2, y2, score, label)
    '''
    invalid_num = 0
    bad_ratio = 0
    ratio_box = 2
    out_size = (128, 256)
    assert out_size[0] * ratio_box == out_size[1]
    human_map = gen_human_id_map(human_detections)
    h5 = h5py.File(hdf5_path, mode='w')
    h5_group_part = h5.create_group(part)
    for image in imageset:
        im_id = image['id']
        img_path = os.path.join(im_root, 'train', image['file_name'] )
        im = Image.open(img_path).convert('RGB')
        im_w, im_h = im.size
        objects = image['objects']
        human_boxes = human_map[im_id]
        biggest_box = get_biggest_box(human_boxes)
        ratio = (biggest_box[3]-biggest_box[1]) / (biggest_box[2]-biggest_box[0])
        if ratio < 2:
            bad_ratio += 1
            continue

        if expand_rate > 0:
            biggest_box = expand_box(biggest_box, expand_rate, im_w, im_h)

        # make the box correct ratio to crop
        biggest_box = pad_box_by_ratio(biggest_box, ratio_box)

        boxes, labels = get_object_inside_box(objects, biggest_box)
        if len(labels) == 0: # no valid
            invalid_num += 1
            continue

        # resize the boxes
        scale = out_size[0] / (biggest_box[2]-biggest_box[0])
        boxes = boxes * scale
        
        cropped_im = im.crop(biggest_box)
        # resize 
        resized_im = cropped_im.resize(out_size)
        data = np.array(resized_im)

        group_name = str(im_id)
        h5_group = h5_group_part.create_group(group_name)
        h5_group['data'] = data
        h5_group['bbox'] = boxes
        h5_group['category_id'] = labels
        h5_group['image_id'] = im_id
    h5.close()
        
def test():
    human_det_path = 'modanet_human_val.pkl'
    h5_out_path = 'test.hdf5'
    anno_path = '/ssd/data/annotations/modanet2018_instances_revised.pkl'
    root = '/ssd/data/datasets/modanet/Images'
    anno_path = os.path.expanduser('~/data/annotations/modanet2018_instances_revised.pkl')
    root = os.path.expanduser('~/data/datasets/modanet/Images')
    part = 'val'

    with open(anno_path, 'rb') as f:
        imageset = pickle.load(f)[part]
    with open(human_det_path, 'rb') as f:
        human_detections = pickle.load(f)
    generate_hdf5_patch(h5_out_path, part, human_detections, imageset, root, expand_rate=0.2)


if __name__ == '__main__':
    human_det_path = '../../modanet_human_val.pkl'
    anno_path = '/ssd/data/annotations/modanet2018_instances_revised.pkl'
    root = '/ssd/data/datasets/modanet/Images'
    #anno_path = os.path.expanduser('~/data/annotations/modanet2018_instances_revised.pkl')
    #root = os.path.expanduser('~/data/datasets/modanet/Images')
    part = 'val'
    h5_out_path = '/ssd/data/datasets/modanet/modanet_{}_hdf5.hdf5'.format(part)

    with open(anno_path, 'rb') as f:
        imageset = pickle.load(f)[part]
    with open(human_det_path, 'rb') as f:
        human_detections = pickle.load(f)
    generate_hdf5_patch(h5_out_path, part, human_detections, imageset, root, expand_rate=0.2)
