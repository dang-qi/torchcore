from posixpath import splitext
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL.ImageDraw import Draw
from PIL import ImageFont, Image
from numpy.core.fromnumeric import argmax
from .color_gen import random_colors
import os
import io
import cv2

FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def cv_put_text_with_box(im, text, position, font=cv2.FONT_HERSHEY_COMPLEX, font_scale=1, font_color=(0,0,0), box_color=(255,0,0), thickness=1):
    retval, baseLine = cv2.getTextSize(text, font, font_scale, thickness)
    pos = (position[0], position[1]+retval[1])
    right_down = (position[0]+retval[0], position[1]+retval[1])
    cv2.rectangle(im, position, right_down, box_color, thickness=-1)
    cv2.putText(im, text, pos, font, font_scale, font_color, thickness=thickness )

def cv_put_text(im, text, position, font=cv2.FONT_HERSHEY_COMPLEX, font_scale=1, font_color=(0,0,0), thickness=1, return_im=False):
    retval, baseLine = cv2.getTextSize(text, font, font_scale, thickness)
    pos = (position[0], position[1]+retval[1])
    #cv2.rectangle(im, position, retval, box_color, thickness=-1)
    cv2.putText(im, text, pos, font, font_scale, font_color, thickness=thickness )
    if return_im:
        return im

def put_text_with_box(im, text, position, size=26, box_color=(0,0,0), text_color=(255,255,255), box_opacity=0.5, text_opacity=0.5, text_margin=10):
    font = get_font(size=size)
    # get text size
    nlines = text.count('\n') + 1
    split_text = text.split('\n')
    max_text_ind = argmax([len(t) for t in split_text])
    text_size = font.getsize(split_text[max_text_ind])
    text_size = (text_size[0], text_size[1]*nlines)

    margin = text_margin

    # set button size + 10px margins
    button_size = (text_size[0]+2*margin, text_size[1]+2*margin)

    # create correct color with opacity
    if isinstance(box_color,tuple) and len(box_color) == 3:
        box_color = box_color + (int(255*box_opacity),)
    if isinstance(text_color,tuple) and len(text_color) == 3:
        text_color = text_color + (int(255*text_opacity),)
    #button_img = Image.new('RGBA', button_size, box_color)

    # put text on button with 10px margins
    button_draw = Draw(im, 'RGBA')
    box = (position[0], position[1], position[0]+button_size[0], position[1]+button_size[1])
    button_draw.rectangle(box, fill=box_color)
    button_draw.text((position[0]+margin, position[1]+margin), text, font=font, fill=text_color)

    # put button on source image in position (0, 0)
    #im.paste(button_img, position)

def parse_loss_log(log_path, losses_names):
    epoch = 0
    pre_iter = 0
    iters = []
    losses = {}
    pre_total_iter = 0
    for name in losses_names:
        losses[name]=[]
    
    with open(log_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if line[0]=='epoch':
                epoch=int(line[1])
                pre_total_iter = pre_total_iter + pre_iter
            else:
                iterations = int(line[0]) + pre_total_iter
                iters.append(iterations)
                assert len(losses_names) == len(line)-1
                for i in range(1, len(line)):
                    losses[losses_names[i-1]].append(float(line[i]))
                pre_iter = int(line[0])
    return losses, iters

def draw_loss(log_path, out_path, losses_names=['loss']):
    f = plt.figure()
    losses, iters = parse_loss_log(log_path, losses_names)
    plt.title('Figure for loss vs iterations')
    for name, loss in losses.items():
        plt.plot(iters, loss, label=name)
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.savefig(out_path)

def parse_bench_log(log_path, type_num=1):
    with open(log_path) as f:
        line = f.readline()
        line = line.split()
        accuracy = np.array([float(x) for x in line]).reshape(-1, type_num)
        accuracies = []
        for i in range(type_num):
            accuracies.append(accuracy[:, i])
        epochs = list(range(1, len(accuracy)+1))
    return accuracies, epochs

def draw_accuracy(log_path, out_path, interval=1):
    f = plt.figure()
    accuracy, epochs = parse_bench_log(log_path)
    plt.title('Figure for accuracy vs epochs')
    plt.plot(epochs, accuracy, color='#B16FDE')
    plt.scatter(epochs, accuracy, color='#551A7C')
    for x,y in zip(epochs, accuracy):
        if x%interval==0:
            plt.annotate(str(y),xy=(x,y))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig(out_path)

def draw_accuracy_all(log_paths, names, out_path, dataset_name, interval=1, type_num=1, type_names=['']):
    assert len(names) == len(log_paths)
    assert type_num == len(type_names)
    for i in range(type_num):
        f = plt.figure(i+1)
        plt.title('{}--Figure for {} accuracy vs epochs'.format(dataset_name, type_names[i]))
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        for j, log_path in enumerate(log_paths):
            accuracies, epochs = parse_bench_log(log_path, type_num=type_num)
            accuracy = accuracies[i]
            plt.plot(epochs, accuracy, label=names[j])
            plt.scatter(epochs, accuracy)
            if interval > 0:
                for x,y in zip(epochs, accuracy):
                    if x%interval==0:
                        plt.annotate(str(y),xy=(x,y))
        plt.legend()

        plt.savefig(out_path.format(type_names[i]), dpi=300)

def draw_pair_class_accuracy(log_path, name, out_path, dataset_name, interval=0, type_num=1, type_names=[], ignore_names=[]):
    assert type_num == len(type_names)
    f=plt.figure()
    plt.title('{}--Figure for {} vs epochs'.format(dataset_name, name))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    for i in range(type_num):
        if type_names[i] in ignore_names:
            continue
        accuracies, epochs = parse_bench_log(log_path, type_num=type_num)
        accuracy = accuracies[i]
        plt.plot(epochs, accuracy, label=type_names[i])
        plt.scatter(epochs, accuracy)
        if interval > 0:
            for x,y in zip(epochs, accuracy):
                if x%interval==0:
                    plt.annotate(str(y),xy=(x,y))
    plt.legend()

    plt.savefig(out_path.format('pair_classification'), dpi=300)

def get_font(size):
    font = ImageFont.truetype(FONT_PATH, size)
    return font

def draw_single_image(image, boxes, scores, class_inds, colors, class_names, font_size=26, box_line_width=3, box_opacity=0.5, text_opacity=0.5, use_invert_text_color=True): 
    # (x1, y1, x2, y2, object_conf, class_score, class_pred)
    font = get_font(font_size)
    invert_colors = [tuple(255-i for i in color) for color in colors]
    if boxes is not None:
        draw = Draw(image)
        for i, (box, class_ind) in enumerate(zip(boxes, class_inds)):
            if scores is not None:
                score = scores[i]
            else:
                score = 1
            if type(box) == list:
                return image
            if type(box) is not np.ndarray:
                box = box.detach().cpu().numpy()
            box_rec = box[:4]
            box_rec = tuple(np.array(box_rec))
            draw.rectangle(box_rec, outline=colors[class_ind], width=box_line_width)
            text = '{:.2f} {}'.format(score, class_names[class_ind])
            #draw.text((box[0], box[1]), text, font=font, fill=colors[class_ind] )
            if use_invert_text_color:
                text_color = invert_colors[class_ind]
            else:
                text_color = (0,0,0)
            put_text_with_box(image, text, (int(box[0]),int(box[1])), size=font_size, 
                                box_color=colors[class_ind],text_color=text_color, 
                                box_opacity=box_opacity, text_opacity=text_opacity)
    return image

def put_text_on_the_leftup_corner(image, text, font_size=26, box_color=(0,0,0)):
    put_text_with_box(image, text, position=(0,0),size=font_size, box_color=box_color)

def draw_boxes_old(images, batch_boxes, batch_scores, batch_class_ind, class_names, font_size=26):
    class_num = len(class_names)
    colors = random_colors(class_num)
    for image, boxes, scores, class_ind in zip(images, batch_boxes, batch_scores, batch_class_ind):
        image = draw_single_image(image, boxes, scores, class_ind, colors, class_names, font_size=font_size)

def draw_boxes(images, batch_boxes, batch_scores, batch_class_ind, class_names, font_size=26):
    class_num = len(class_names)
    colors = random_colors(class_num)
    for i, image in enumerate(images):
        if batch_boxes is not None:
            boxes = batch_boxes[i]
        else:
            boxes = None
        if batch_scores is not None:
            scores = batch_scores[i]
        else:
            scores = None
        if batch_class_ind is not None:
            class_ind = batch_class_ind[i]
        else:
            class_ind = None

        image = draw_single_image(image, boxes, scores, class_ind, colors, class_names, font_size=font_size)

def save_images(images, path, ind_start):
    for i, image in enumerate(images):
        im_path = os.path.join(path, '{:06d}.jpg'.format(ind_start+i))
        image.save(im_path)

def draw_and_save(images, batch_boxes, batch_scores, batch_class_ind, path, ind_start, class_names, font_size=26):
    '''
        images: a list of PIL Image
        batch_boxes: list(np.array(n*4) or None for no detection, np.array(m*4), ...)
        batch_scores: list(np.array(n1),np.array(n2)...)
        batch_class_ind: list(np.array(n1),np.array(n2)...)
        path: out_path to save the images
        ind_start: the saveed image is formated as {:06d}.jpg
        class_names: list(name1, name2)
    '''
    #colors = random_color_fix(class_num)
    draw_boxes(images, batch_boxes, batch_scores, batch_class_ind, class_names, font_size=font_size)
    save_images(images, path, ind_start)

def draw_single_box(image, box):
    draw = Draw(image)
    draw.rectangle(box, outline=(255, 0, 0), width=3)
    return image

def draw_plain_boxes(image,boxes, color=(255, 0, 0)):
    draw = Draw(image)
    for box in boxes:
        box = list(np.array(box, dtype=int))
        draw.rectangle(box, outline=color, width=3)
    return image

def visulize_mask_with_image(mask, image):
    #heatmap = np.amax(heatmap,axis=0)
    mask = Image.fromarray((mask*255).astype(np.uint8)).convert('RGB')
    #mask = mask.resize(image.size)
    blend_im = Image.blend(mask, image, 0.5)
    #blend_im.show()
    return blend_im

def visulize_heatmaps_with_image(heatmap, image):
    heatmap = np.amax(heatmap,axis=0)
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8)).convert('RGB')
    heatmap = heatmap.resize(image.size)
    blend_im = Image.blend(heatmap, image, 0.5)
    blend_im.show()
    return blend_im

def visulize_heatmaps(heatmap):
    heatmap = np.amax(heatmap,axis=0)
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8)).convert('RGB')
    heatmap.show()
    return heatmap

def visulize_colored_heatmaps_with_image(heatmap, image):
    fig = plt.figure(frameon=False)
    heatmap = np.amax(heatmap,axis=0)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(heatmap, aspect='auto')
    plt.set_cmap('jet')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    heatmap = Image.open(buf).convert('RGB')
    #heatmap = Image.fromarray((heatmap*255).astype(np.uint8)).convert('RGB')
    heatmap = heatmap.resize(image.size)
    blend_im = Image.blend(heatmap, image, 0.5)
    #blend_im.show()
    return blend_im

def draw_category_distribution_barh(stat_dict, sorted=False, title=None):
    '''
    stat_dict: dict(category_name:item_number)
    sorted: do you what to sort the category distribution
    '''
    names=[]
    val = []
    for k,v in stat_dict.items():
        names.append(k)
        val.append(v)
    if sorted:
        val = np.array(val)
        order = np.argsort(val)
        names = np.array(names)
        names = names[order]
        val = val[order]

    x = np.arange(len(names))
    #plt.figure(figsize=(8,10))
    fig, ax = plt.subplots(figsize=(8,10))
    ax.barh(x, val,log=True, tick_label=names)
    ax.set_xlabel('Number')
    ax.set_ylabel('Category')
    ax.bar_label(ax.containers[0])
    if title is not None:
        ax.set_title(title)
    plt.show()

def visulize_random_sample(dataset, category_ids, sample_num, category_names, category_num):
    colors = random_colors(category_num)
    dataset.set_category_subset(category_ids, ignore_other_category=True)
    dataset_len = len(dataset)
    if dataset_len < sample_num:
        print('sample in the dataset is not enough ({} vs {}), set sample num to dataset length'.format(dataset_len, sample_num))
        sample_num = dataset_len
    inds = np.random.choice(dataset_len, sample_num, replace=False)
    for i in inds:
        inputs, targets = dataset[i]
        im = inputs['data']
        boxes = targets['boxes']
        labels = targets['labels']-1
        draw_single_image(im, boxes, scores=None, class_inds=labels, colors=colors, class_names=category_names)
        plt.figure(figsize=(8,8))
        plt.imshow(im)

def visulize_coco_result(cocoEval, im_id, im_folder, category_num, cat_ids=[], names=None, thresh=0.5, with_gt=False, tag=None):
    '''cocoEval should be a COCOEval object'''
    def get_anno(anno_id, coco_obj, thresh=None,no_score=False):
        im_ids = [anno_id]
        anno_ids = coco_obj.getAnnIds(imgIds=im_ids, catIds=cat_ids)
        annos = coco_obj.loadAnns(ids=anno_ids)
        if thresh is not None:
            annos = [a for a in annos if a['score']>thresh]
        if len(annos)==0:
            return None, None, None
        boxes = np.stack([anno['bbox'] for anno in annos],axis=0)
        boxes[:,2] += boxes[:,0]
        boxes[:,3] += boxes[:,1]
        if no_score:
            scores = np.ones(len(annos))
        else:
            scores = np.array([anno['score'] for anno in annos])
        class_inds = np.array([anno['category_id'] for anno in annos])
        return boxes, scores, class_inds

    im_ids = [im_id]
    colors = random_colors(category_num)
    img = cocoEval.cocoDt.loadImgs(ids=im_ids)[0]
    img_path = os.path.join(im_folder,img['file_name'])
    if names is None:
        names = cocoEval.get_names()
    image = Image.open(img_path)
    boxes, scores, class_inds = get_anno(im_id, cocoEval.cocoDt, thresh)
    if boxes is None:
        put_text_on_the_leftup_corner(image, 'NO DETECTION', font_size=26, box_color=(0,0,0))
    if tag is not None:
        put_text_on_the_leftup_corner(image, tag, font_size=26, box_color=(0,0,0))

    draw_single_image(image, boxes, scores, class_inds, colors, names, box_opacity=0.25, text_opacity=0.5)
    image_gt = Image.open(img_path)
    if with_gt:
        if tag is not None:
            put_text_on_the_leftup_corner(image_gt, 'ground truth', font_size=26, box_color=(0,0,0))
        boxes, scores, class_inds = get_anno(im_id, cocoEval.cocoGt, thresh=None, no_score=True)
        draw_single_image(image_gt, boxes, scores, class_inds, colors, names, box_line_width=5,)
        return image, image_gt

    return image

def cat_images_with_same_size(images, layout_col, w_margin=10, h_margin=10):
    '''concat the images for easier show
    images is List(PIL Images)
    layout_col: column number of expect image
    '''
    layout_row = math.ceil(len(images)/layout_col)

    im_w = images[0].width
    im_h = images[0].height

    out_w = int(im_w*layout_col + w_margin*(layout_col-1))
    out_h = int(im_h*layout_row+h_margin*(layout_row-1))
    dst = Image.new('RGB', (out_w, out_h))
    im_ind = 0
    for i in range(layout_row):
        for j in range(layout_col):
            x = j*im_w + j*w_margin
            y = i*im_h + i*h_margin
            dst.paste(images[im_ind], (x,y))
            im_ind +=1
            if im_ind == len(images):
                break
    return dst

