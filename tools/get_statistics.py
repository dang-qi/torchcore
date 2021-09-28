import sys
sys.path.append('..')
sys.path.append('.')
from .hdf5_generater import *

def xywh_to_xyxy(box):
    box_new = box.copy()
    box_new[2] += box_new[0]
    box_new[3] += box_new[1]
    return box_new

def cal_IoR(human_box, garment_box):
    '''IoR: Intersection/garment_box_area'''
    x_min = max(human_box[0], garment_box[0])
    y_min = max(human_box[1], garment_box[1])
    x_max = min(human_box[2], garment_box[2])
    y_max = min(human_box[3], garment_box[3])
    if x_max<=x_min or y_max<=y_min:
        return 0
    intersection = (x_max-x_min)*(y_max-y_min)
    garment_area = (garment_box[2]-garment_box[0])*(garment_box[3]-garment_box[1])
    IoR = intersection / garment_area
    assert IoR>0 and IoR<=1
    return IoR

def intersection_status(human_box, objects, ratio=0.7):
    '''objects"[{'bbox':bbox(x,y,w,h), 'category_id':cat_id}]'''
    good_box = 0
    bad_box = 0
    for obj in objects:
        bbox = obj['bbox']
        bbox = xywh_to_xyxy(bbox)
        IoR = cal_IoR(human_box, bbox)
        if IoR >= ratio:
            good_box+=1
        else:
            bad_box+=1

    return good_box, bad_box

def intersection_status_by_category(human_box, objects, category_num, ratio=0.7):
    '''objects"[{'bbox':bbox(x,y,w,h), 'category_id':cat_id}]'''
    good_box = np.zeros(category_num)
    bad_box = np.zeros(category_num)
    for obj in objects:
        bbox = obj['bbox']
        category_id = obj['category_id']
        bbox = xywh_to_xyxy(bbox)
        IoR = cal_IoR(human_box, bbox)
        if IoR >= ratio:
            good_box[category_id-1]+=1
        else:
            bad_box[category_id-1]+=1

    return good_box, bad_box

def get_statistics(part, human_detections, imageset, expand_rate, im_root=None, gen_id_map=True, intersection_ratio_thresh=0.7):
    no_detection = 0
    has_human_detection = 0
    invalid_num = 0
    bad_ratio = 0
    good_ratio = 0
    lost_garments_all = 0
    lost_garments_roi = 0
    garments = 0
    garments_with_human = 0
    garments_with_good_human = 0
    g_in = 0 # garments in box (ratio>0.7)
    g_out = 0
    g_in_ext = 0 # garments in extended box (ratio>0.7)
    g_out_ext = 0

    # only count the one with good human detecion(human box ratio > 2)
    g_in_good = 0 # garments in box (ratio>0.7)
    g_out_good = 0
    g_in_ext_good = 0 # garments in extended box (ratio>0.7)
    g_out_ext_good = 0
    ratio_box = 2
    out_size = (128, 256)
    assert out_size[0] * ratio_box == out_size[1]
    if gen_id_map:
        human_map = gen_human_id_map(human_detections)
    else:
        human_map = human_detections
    for image in imageset:
        im_id = image['id']
        im_w, im_h = image['width'], image['height']

        objects = image['objects']
        garments += len(objects)
        human_boxes = human_map[im_id]
        if len(human_boxes)==0:
            no_detection += 1
            lost_garments_all += len(objects)
            continue
        else:
            has_human_detection += 1
        garments_with_human += len(objects)
        biggest_box = get_biggest_box(human_boxes)
        ratio = (biggest_box[3]-biggest_box[1]) / (biggest_box[2]-biggest_box[0])
        if ratio < 2:
            bad_ratio += 1
            good_human = False
            #continue
        else:
            good_ratio += 1
            good_human = True
            garments_with_good_human += len(objects)

        #img_path = os.path.join(im_root, 'train', image['file_name'] )
        #im = Image.open(img_path).convert('RGB')

        # before box extend 
        good_box, bad_box = intersection_status(biggest_box, objects)
        g_in += good_box
        g_out += bad_box
        if good_human:
            g_in_good += good_box
            g_out_good += bad_box

        if expand_rate > 0:
            biggest_box = expand_box(biggest_box, expand_rate, im_w, im_h)

        # after box extension
        good_box, bad_box = intersection_status(biggest_box, objects)
        g_in_ext += good_box
        g_out_ext += bad_box
        if good_human:
            g_in_ext_good += good_box
            g_out_ext_good += bad_box

        # make the box correct ratio to crop
        #biggest_box = pad_box_by_ratio(biggest_box, ratio_box)

    image_number = len(imageset)
    print('total image number is {}'.format(image_number))

    print('--%Person Detection--')
    print('Person detection rate is {}'.format(has_human_detection/image_number))
    print()

    print('---%Garments in all image---')
    print('Without extention good garments ratio is {}'.format( g_in/garments))
    print('With extention good garments ratio is {}'.format( g_in_ext/garments))
    print('With extention bad garments ratio is {}'.format( g_out_ext/garments))
    print()

    print('---%Garments in images with person---')
    print('Without extention good garments ratio is {}'.format(g_in/garments_with_human))
    print('With extention good garments ratio is {}'.format(g_in_ext/garments_with_human))
    print()

    print('---%Person detection with h>2*w ---')
    print('Good person detection rate is {}'.format(good_ratio/ image_number))
    print()

    print('---% Garments for h>2*w---')
    print('Without extention good garments in all garments ratio is {}'.format(g_in_good/garments))
    print('With extention good garments in all garments ratio is {}'.format(g_in_ext_good/garments))
    print('Without extention good garments in garments in good ratio image ratio is {}'.format(g_in_good/garments_with_good_human))
    print('With extention good garments in garments in good ratio image ratio is {}'.format(g_in_ext_good/garments_with_good_human))

def get_garments_in_human_statistics(human_detections, imageset, expand_rate, category_num, gen_id_map=False, intersection_ratio_thresh=0.7, print_result=True):
    '''
    category id in the imageset should start from 1
    '''
    no_detection = 0
    has_human_detection = 0
    invalid_num = 0
    bad_ratio = 0
    good_ratio = 0
    lost_garments_all = 0
    lost_garments_roi = 0
    garments = np.zeros(category_num)
    garments_with_human = 0
    garments_with_good_human = 0
    g_in = np.zeros(category_num) # garments in box (ratio>0.7)
    g_out = np.zeros(category_num)
    g_in_ext = np.zeros(category_num) # garments in extended box (ratio>0.7)
    g_out_ext = np.zeros(category_num)

    # only count the one with good human detecion(human box ratio > 2)
    g_in_good = np.zeros(category_num) # garments in box (ratio>0.7)
    g_out_good = np.zeros(category_num)
    g_in_ext_good = np.zeros(category_num) # garments in extended box (ratio>0.7)
    g_out_ext_good = np.zeros(category_num)
    #ratio_box = 2
    #out_size = (128, 256)
    #assert out_size[0] * ratio_box == out_size[1]
    if gen_id_map:
        human_map = gen_human_id_map(human_detections)
    else:
        human_map = human_detections
    for image in imageset:
        im_id = image['id']
        im_w, im_h = image['width'], image['height']

        objects = image['objects']
        for obj in objects:
            garments[obj['category_id']-1] += 1
        if im_id not in human_map:
            print('image {} has no human detection'.format(im_id))
            continue
        human_boxes = human_map[im_id]
        if len(human_boxes)==0:
            no_detection += 1
            lost_garments_all += len(objects)
            continue
        else:
            has_human_detection += 1
        garments_with_human += len(objects)
        #biggest_box = get_biggest_box(human_boxes)
        biggest_box = human_boxes['input_box']
        ratio = (biggest_box[3]-biggest_box[1]) / (biggest_box[2]-biggest_box[0])
        if ratio < 2:
            bad_ratio += 1
            good_human = False
            #continue
        else:
            good_ratio += 1
            good_human = True
            garments_with_good_human += len(objects)

        #img_path = os.path.join(im_root, 'train', image['file_name'] )
        #im = Image.open(img_path).convert('RGB')

        # before box extend 
        good_box, bad_box = intersection_status_by_category(biggest_box, objects, category_num, intersection_ratio_thresh)
        g_in += good_box
        g_out += bad_box
        if good_human:
            g_in_good += good_box
            g_out_good += bad_box

        if expand_rate > 0:
            biggest_box = expand_box(biggest_box, expand_rate, im_w, im_h)

        # after box extension
        good_box, bad_box = intersection_status_by_category(biggest_box, objects, category_num, intersection_ratio_thresh)
        g_in_ext += good_box
        g_out_ext += bad_box
        if good_human:
            g_in_ext_good += good_box
            g_out_ext_good += bad_box

        # make the box correct ratio to crop
        #biggest_box = pad_box_by_ratio(biggest_box, ratio_box)

    image_number = len(imageset)
    if print_result:
        print('total image number is {}'.format(image_number))

        print('--%Person Detection--')
        print('Person detection rate is {}'.format(has_human_detection/image_number))
        print()

        print('---%Garments in all image---')
        print('Without extention good garments ratio is {}'.format( g_in/garments))
        print('With extention good garments ratio is {}'.format( g_in_ext/garments))
        print('With extention bad garments ratio is {}'.format( g_out_ext/garments))
        print()

        print('---%Garments in images with person---')
        print('Without extention good garments ratio is {}'.format(g_in/garments_with_human))
        print('With extention good garments ratio is {}'.format(g_in_ext/garments_with_human))
        print()

        print('---%Person detection with h>2*w ---')
        print('Good person detection rate is {}'.format(good_ratio/ image_number))
        print()

        print('---% Garments for h>2*w---')
        print('Without extention good garments in all garments ratio is {}'.format(g_in_good/garments))
        print('With extention good garments in all garments ratio is {}'.format(g_in_ext_good/garments))
        print('Without extention good garments in garments in good ratio image ratio is {}'.format(g_in_good/garments_with_good_human))
        print('With extention good garments in garments in good ratio image ratio is {}'.format(g_in_ext_good/garments_with_good_human))
    return g_in, g_in_ext, garments

def image_test(human_detections, imageset, im_root):
    human_map = gen_human_id_map(human_detections)
    for image in imageset:
        #ind = np.random.randint(1000)
        #image = imageset[ind]
        im_id = image['id']
        img_path = os.path.join(im_root, 'train', image['file_name'] )
        im = Image.open(img_path).convert('RGB')
        human_boxes = human_map[im_id]
        print(human_boxes)
        draw_plain_boxes(im, human_boxes)
        im.show()
        break

def get_dataset_category_statics(path, part, names=None):
    with open(path, 'rb') as f:
        anno = pickle.load(f)[part]
        labels = []
        boxes = []
        for image in anno:
            for obj in image['objects']:
                labels.append(obj['category_id'])
                boxes.append(obj['bbox'])

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        max_label = labels.max()
        min_label = labels.min()

        if names is None:
            names = [str(i) for i in range(min_label, max_label+1)]

        assert len(names) >= max_label - min_label + 1

        label_num = np.zeros(max_label-min_label+1)
        for i,j in enumerate(range(min_label, max_label+1)):
            num = np.sum(labels==j)
            label_num[i] = num
        item_num = len(labels)

        result = {name:num for name,num in zip(names, label_num)}
        return result

def get_dataset_area_statics_by_category(path, part, area_bin, longer_side_len, names=None):
    '''
    area_bin should be a iterable [] or numpy array that contains the thresh 
    set longer_side_len as None if you don't want to resize the image
    '''
    with open(path, 'rb') as f:
        anno = pickle.load(f)[part]
        labels = []
        boxes = []
        for image in anno:
            im_width = image['width']
            im_height = image['height']
            if longer_side_len is not None:
                scale = max(im_width, im_height) / 1024
            else:
                scale = 1
            for obj in image['objects']:
                labels.append(obj['category_id'])
                boxes.append(obj['bbox'])

        boxes = np.array(boxes, dtype=np.float32) / scale
        labels = np.array(labels, dtype=np.int64)
        box_areas = (boxes[:,3]) * (boxes[:,2])

        max_label = labels.max()
        min_label = labels.min()

        if area_bin is not None:
            bin_len = len(area_bin)+1

        if names is None:
            names = [str(i) for i in range(min_label, max_label+1)]

        assert len(names) >= max_label - min_label + 1

        result_all = []
        for i,label in enumerate(range(min_label, max_label+1)):
            box_areas_with_label = box_areas[labels==label]
            if area_bin is None:
                cat_result = box_areas_with_label
            else:
                cat_result = np.zeros(bin_len)
                for k in range(bin_len):
                    if k==0:
                        num = (box_areas_with_label<area_bin[0]).sum()
                    elif k== bin_len-1:
                        num = (box_areas_with_label>= area_bin[k-1]).sum()
                    else:
                        num = np.bitwise_and(box_areas_with_label>=area_bin[k-1], box_areas_with_label<area_bin[k]).sum()
                    cat_result[k] = num
            result_all.append(cat_result)

        result = {name:num for name,num in zip(names, result_all)}
        return result

def get_dataset_aspect_ratio_statics_by_category(path, part, aspect_ratio_bin, longer_side_len, names=None):
    '''
    area_bin should be a iterable [] or numpy array that contains the thresh 
    set longer_side_len as None if you don't want to resize the image
    '''
    with open(path, 'rb') as f:
        anno = pickle.load(f)[part]
        labels = []
        boxes = []
        for image in anno:
            im_width = image['width']
            im_height = image['height']
            if longer_side_len is not None:
                scale = max(im_width, im_height) / 1024
            else:
                scale = 1
            for obj in image['objects']:
                labels.append(obj['category_id'])
                boxes.append(obj['bbox'])

        boxes = np.array(boxes, dtype=np.float32) / scale
        labels = np.array(labels, dtype=np.int64)
        box_aspect_ratio = (boxes[:,3]) / (boxes[:,2])

        max_label = labels.max()
        min_label = labels.min()

        if aspect_ratio_bin is not None:
            bin_len = len(aspect_ratio_bin)+1

        if names is None:
            names = [str(i) for i in range(min_label, max_label+1)]

        assert len(names) >= max_label - min_label + 1

        result_all = []
        for i,label in enumerate(range(min_label, max_label+1)):
            box_aspect_ratio_with_label = box_aspect_ratio[labels==label]
            if aspect_ratio_bin is None:
                cat_result = box_aspect_ratio_with_label
            else:
                cat_result = np.zeros(bin_len)
                for k in range(bin_len):
                    if k==0:
                        num = (box_aspect_ratio_with_label<aspect_ratio_bin[0]).sum()
                    elif k== bin_len-1:
                        num = (box_aspect_ratio_with_label>= aspect_ratio_bin[k-1]).sum()
                    else:
                        num = np.bitwise_and(box_aspect_ratio_with_label>=aspect_ratio_bin[k-1], box_aspect_ratio_with_label<aspect_ratio_bin[k]).sum()
                    cat_result[k] = num
            result_all.append(cat_result)

        result = {name:num for name,num in zip(names, result_all)}
        return result
        

def get_human_size_statistics(human_detections, imageset, gen_id_map=False):
    '''
    category id in the imageset should start from 1
    '''
    human_ratios = []
    no_detection = 0
    has_human_detection = 0
    if gen_id_map:
        human_map = gen_human_id_map(human_detections)
    else:
        human_map = human_detections
    for image in imageset:
        im_id = image['id']
        im_w, im_h = image['width'], image['height']

        if im_id not in human_map:
            print('image {} has no human detection'.format(im_id))
            continue

        human_boxes = human_map[im_id]
        if len(human_boxes)==0:
            no_detection += 1
            continue
        else:
            has_human_detection += 1
        biggest_box = human_boxes['input_box']
        max_ratio = max((biggest_box[3]-biggest_box[1])/im_h, (biggest_box[2]-biggest_box[0])/im_w)

        human_ratios.append(max_ratio)

    return human_ratios


            




if __name__ == '__main__':
    anno_path = '/ssd/data/annotations/modanet2018_instances_revised.pkl'
    root = '/ssd/data/datasets/modanet/Images'
    anno_path = os.path.expanduser('~/data/annotations/modanet2018_instances_revised.pkl')
    root = os.path.expanduser('~/data/datasets/modanet/Images')
    part = 'val'
    human_det_path = 'modanet_human_{}.pkl'.format(part)
    human_det_path = 'yolo_human_info_modanet_{}.pkl'.format(part)
    human_det_path = 'centernet_human_{}.pkl'.format(part)

    with open(anno_path, 'rb') as f:
        imageset = pickle.load(f)[part]
    with open(human_det_path, 'rb') as f:
        human_detections = pickle.load(f)
    #image_test(human_detections, imageset, root)
    get_statistics(part, human_detections, imageset, expand_rate=0.2, im_root=root)