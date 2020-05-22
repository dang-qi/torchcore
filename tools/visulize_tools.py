import matplotlib.pyplot as plt
import numpy as np
from PIL.ImageDraw import Draw
from PIL import ImageFont, Image
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

def cv_put_text(im, text, position, font=cv2.FONT_HERSHEY_COMPLEX, font_scale=1, font_color=(0,0,0), thickness=1):
    retval, baseLine = cv2.getTextSize(text, font, font_scale, thickness)
    pos = (position[0], position[1]+retval[1])
    #cv2.rectangle(im, position, retval, box_color, thickness=-1)
    cv2.putText(im, text, pos, font, font_scale, font_color, thickness=thickness )

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

def draw_single_image(image, boxes, scores, class_inds, colors, class_names, font_size=26): 
    # (x1, y1, x2, y2, object_conf, class_score, class_pred)
    font = get_font(font_size)
    if boxes is not None:
        draw = Draw(image)
        for box, score, class_ind in zip(boxes, scores, class_inds):
            if type(box) == list:
                return image
            if type(box) is not np.ndarray:
                box = box.detach().cpu().numpy()
            box_rec = box[:4]
            draw.rectangle(box_rec, outline=colors[class_ind], width=3)
            draw.text((box[0], box[1]), '{:.2f} {}'.format(score, class_names[class_ind]), font=font, fill=colors[class_ind] )
    return image

def draw_boxes(images, batch_boxes, batch_scores, batch_class_ind, class_names, font_size=26):
    class_num = len(class_names)
    colors = random_colors(class_num)
    for image, boxes, scores, class_ind in zip(images, batch_boxes, batch_scores, batch_class_ind):
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

def draw_plain_boxes(image,boxes):
    draw = Draw(image)
    for box in boxes:
        box = list(np.array(box, dtype=int))
        draw.rectangle(box, outline=(255, 0, 0), width=3)
    return image


def visulize_heatmaps_with_image(heatmap, image):
    heatmap = np.amax(heatmap,axis=0)
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8)).convert('RGB')
    heatmap = heatmap.resize(image.size)
    blend_im = Image.blend(heatmap, image, 0.5)
    blend_im.show()

def visulize_heatmaps(heatmap):
    heatmap = np.amax(heatmap,axis=0)
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8)).convert('RGB')
    heatmap.show()

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
    blend_im.show()