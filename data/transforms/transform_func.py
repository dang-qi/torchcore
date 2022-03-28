from PIL import Image, ImageOps
import collections
import torch
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import random
import cv2
import math


Iterable = collections.abc.Iterable
Sequence = collections.abc.Sequence

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy(img):
    return isinstance(img, np.ndarray)

def _is_numpy_image(img):
    return img.ndim in {2, 3}

def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

def resize(img, size, interpolation=Image.BILINEAR, smaller_edge=None):
    r"""Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller / bigger edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        smaller_edge: If size is an int the smaller or bigger edge of the image will be matched by this flag
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
        scale: the resized scale, a int or (scale_w, scale_h)
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if isinstance(size, int) and smaller_edge is None:
        raise TypeError('smaller_edge should be speicified when size is an int')

    w, h = img.size
    if isinstance(size, int):
        if smaller_edge:
            if (w <= h and w == size) or (h <= w and h == size):
                return img, 1.0
            if w < h:
                ow = size
                oh = max(1, int(size * h / w+0.5)) # follow mmdet
                #scale = size / w
                scale = (ow/w, oh/h)
                return img.resize((ow, oh), interpolation), scale
            else:
                oh = size
                ow = max(1,int(size * w / h+0.5)) # follow mmdet
                #scale = size / h
                scale = (ow/w, oh/h)
                return img.resize((ow, oh), interpolation), scale
        else:
            if (w >= h and w == size) or (h >= w and h == size):
                return img, 1.0
            if w > h:
                ow = size
                oh = max(1, int(size * h / w))
                #scale = size / w
                scale = (ow/w, oh/h)
                return img.resize((ow, oh), interpolation), scale
            else:
                oh = size
                ow = max(1, int(size * w / h))
                #scale = size / h
                scale = (ow/w, oh/h)
                return img.resize((ow, oh), interpolation), scale

    else:
        scale_w = size[1] / w
        scale_h = size[0] / h
        return img.resize(size[::-1], interpolation), (scale_w, scale_h)

def resize_max(img, max_size, interpolation=Image.BILINEAR):
    '''Resize the image to the longest side reaching max_size while keep the aspect ratio. 
    max_size: int or (h,w)
    '''
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if isinstance(max_size, int):
        max_size = (max_size, max_size)

    w, h = img.size
    if max(w,h) == max_size:
        return img, (1.0, 1.0)
    scale = min(max_size[1]/w, max_size[0]/h)
    ow = max(1, int(w * scale))
    oh = max(1, int(h * scale))
    scale = (ow/w, oh/h)
    return img.resize((ow, oh), interpolation), scale

def resize_tensor_min_max(image, min_size, max_size):
    h, w = image.shape[-2:]
    min_side=float(min(w,h))
    max_side=float(max(w,h)) 
    if min_size / min_side * max_side > max_size:
        scale = max_size / max_side
    else:
        scale = min_size / min_side
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale, mode='bilinear', recompute_scale_factor=True,
        align_corners=False)[0]
    new_h, new_w = image.shape[-2:]
    scale_w = new_w / w
    scale_h = new_h / h
    scale = (scale_w, scale_h)
        
    return image, scale

def resize_boxes(boxes, scale):
    r"""Resize the input boxes (each box is [x1,y1,x2,y2] format) to the given size.
    Args:
        boxes (PIL Image): boxes to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be scaled in both width and height. 
            If scale is an float, the boxes will be scaled keeping the original
            width and height ratio.
    Returns:
        Boxes: Resized boxes.
    """

    if isinstance(scale, float):
        return boxes * scale
    else:
        scale_w, scale_h =  scale
        boxes[:,0] = boxes[:,0] * scale_w
        boxes[:,2] = boxes[:,2] * scale_w
        boxes[:,1] = boxes[:,1] * scale_h
        boxes[:,3] = boxes[:,3] * scale_h
        return boxes

# from mmcv
def pad(img,
          *,
          shape=None,
          padding=None,
          pad_val=0,
          padding_mode='constant',
          use_pillow_img=True):
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray: The padded image.
    """

    assert (shape is not None) ^ (padding is not None)
    if use_pillow_img:
        img = np.asarray(img)
    if shape is not None:
        padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)

    if use_pillow_img:
        img = Image.fromarray(img)

    return img

def pad_torch(img, padding, fill=0, padding_mode='constant'): # this is from pytorch
    r"""Pad the given PIL Image on all sides with specified padding mode and fill value.
    Args:
        img (PIL Image): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value on the edge of the image
            - reflect: pads with reflection of image (without repeating the last value on the edge)
                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image (repeating the last value on the edge)
                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]
    Returns:
        PIL Image: Padded image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if padding_mode == 'constant':
        if img.mode == 'P':
            palette = img.getpalette()
            image = ImageOps.expand(img, border=padding, fill=fill)
            image.putpalette(palette)
            return image

        return ImageOps.expand(img, border=padding, fill=fill)
    else:
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        if isinstance(padding, Sequence) and len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        if isinstance(padding, Sequence) and len(padding) == 4:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

        if img.mode == 'P':
            palette = img.getpalette()
            img = np.asarray(img)
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)
            img = Image.fromarray(img)
            img.putpalette(palette)
            return img

        img = np.asarray(img)
        # RGB image
        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)

        return Image.fromarray(img)

def pad_boxes(boxes, padding):
    if isinstance(padding, int):
        pad_left = pad_top = padding
    if isinstance(padding, Sequence) and len(padding) == 2:
        pad_left = padding[0]
        pad_top = padding[1]
    if isinstance(padding, Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
    boxes[:,0] = boxes[:,0] + pad_left
    boxes[:,2] = boxes[:,2] + pad_left
    boxes[:,1] = boxes[:,1] + pad_top
    boxes[:,3] = boxes[:,3] + pad_top
    return boxes
    

def resize_and_pad(img, size, interpolation=Image.BILINEAR):
    if not isinstance(size, int):
        raise ValueError('The size can only be an int in the setting')

    img, scale = resize(img, size, interpolation=interpolation, smaller_edge=False)

    w, h = img.size
    pad_left = int((size-w)/2)
    pad_top = int((size-h)/2)
    pad_right = size - w - pad_left
    pad_bottom = size - h - pad_top
    padding = (pad_left, pad_top, pad_right, pad_bottom)
    img = pad(img, padding)
    return img, scale, padding

def resize_min_max(img, min_size, max_size, interpolation=Image.BILINEAR):
    w, h =img.size
    min_side=min(w,h) 
    max_side=max(w,h) 
    if min_size / min_side * max_side > max_size:
        img, scale = resize(img, max_size, smaller_edge=False)
    else:
        img, scale = resize(img, min_size, smaller_edge=True)
    return img, scale


def resize_and_pad_boxes(boxes, scale, padding):
    boxes = resize_boxes(boxes, scale)
    boxes = pad_boxes(boxes, padding)
    return boxes

def to_tensor(pic): # from pytorch
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img

def normalize(tensor, mean, std, inplace=False): # from pytorch
    """Normalize a tensor image with mean and standard deviation.
    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor

def normalize_image(image, mean, std, inplace=False):
    '''normalize a PIL image or numpy array'''
    if not inplace:
        image = image.copy()
    pil_img = False
    if isinstance(image, Image.Image):
        image = np.array(image, dtype=np.float64)
        #image = np.array(image)
        pil_img = True
    elif isinstance(image, np.ndarray):
        image=image.astype(np.float64)
    else:
        raise ValueError('Not support data type {}'.format(type(image)))
    mean = np.array(mean, dtype=np.float64) #.reshape(1,-1)
    std = np.array(std, dtype=np.float64) #.reshape(1,-1)
    #stdinv = 1/std
    #image = (image - mean[:, None, None]) / std[:, None, None]
    image = (image - mean) / std
    #cv2.subtract(image, mean, image)  # inplace
    #cv2.multiply(image, stdinv, image)  # inplace
    #if pil_img:
    #    image = Image.fromarray(image.astype(np.uint8))
    return image

def mirror(image):
    '''morror an image horizontally'''
    return ImageOps.mirror(image)

def mirror_boxes(boxes, im_width):
    boxes[...,0], boxes[...,2] = im_width -boxes[...,2] - 1, im_width -boxes[...,0]-1
    #boxes[:,2] = im_width -boxes[:,2]
    return boxes

def group_padding(images, width, height):
    im_num = len(images)
    the_shape = (im_num, images[0].shape[0], height, width)
    new_ims = images[0].new_zeros(the_shape)
    for img, pad_img in zip(images, new_ims):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    return new_ims

def random_crop(image, size):
    width, height = image.size
    range_w = width - size[0] if width > size[0] else 0
    range_h = height - size[1] if height > size[1] else 0
    x1 = random.randint(0, range_w)
    y1 = random.randint(0, range_h)
    x2 = x1 + size[0]
    y2 = y1 + size[1]

    cropped_im = image.crop((x1, y1, x2, y2))
    return cropped_im, (x1, y1, x2, y2)

def random_crop_boxes(boxes, position):
    boxes[...,0] -= position[0]
    boxes[...,2] -= position[0]
    boxes[...,1] -= position[1]
    boxes[...,3] -= position[1]
    boxes[...,0] = np.clip(boxes[...,0], 0, position[2]-position[0]-1)
    boxes[...,2] = np.clip(boxes[...,2], 0, position[2]-position[0]-1)
    boxes[...,1] = np.clip(boxes[...,1], 0, position[3]-position[1]-1)
    boxes[...,3] = np.clip(boxes[...,3], 0, position[3]-position[1]-1)
    return boxes

def scale(image, scale):
    width, height = image.size
    width = round(width*scale)
    height = round(height*scale)
    return image.resize((width, height))

def scale_box(boxes, scale):
    return boxes*scale

def surrounding_box(boxes):
    x1 = np.min(boxes[:,0])
    y1 = np.min(boxes[:,1])
    x2 = np.max(boxes[:,2])
    y2 = np.max(boxes[:,3])
    surrounding_box = np.array([x1,y1,x2,y2])
    return surrounding_box

def mirror_masks(masks):
    '''
        masks: shape: (channel, height, width)
    '''
    #assert len(masks.shape) == 3 , "shape of mask is {}".format(masks.shape)

    masks = np.flip(masks, axis=2)
    #masks = np.transpose(masks,(1,2,0))
    #masks = cv2.flip(masks, 1)
    ## when mask shape = (1, h, w), the cv2 will make ignore the channel,
    ## so we need to expand the dim back
    #if len(masks.shape) == 2:
    #    masks = np.expand_dims(masks, 0)
    #else:
    #    masks = np.transpose(masks,(2,0,1))
    return masks

def scale_masks(masks, scale):
    '''
        masks: shape: (channel, height, width)
    '''
    masks = torch.from_numpy(masks.copy())
    masks = torch.nn.functional.interpolate(masks[None].float(),scale_factor=scale,mode='bilinear',align_corners=False)[0].byte()
    masks = masks.numpy()
    #masks = np.transpose(masks,(1,2,0))
    #masks = cv2.resize(masks, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    ## when mask shape = (1, h, w), the cv2 will make ignore the channel,
    ## so we need to expand the dim back
    #if len(masks.shape) == 2:
    #    masks = np.expand_dims(masks, 0)
    #else:
    #    masks = np.transpose(masks,(2,0,1))
    return masks

def crop_masks(masks, position):
    '''
        masks: shape: (channel, height, width)
        position: (x1,y1,x2,y2)
    '''
    x1, y1, x2, y2 = position
    width = x2 - x1
    height = y2 - y1
    c, h, w = masks.shape
    #if h-y1 >= height and w-x1 >= width:
    #    return masks[:,y1:y2, x1:x2]
    #else:
    out = np.zeros((c, height, width),dtype=masks.dtype)
    out[:,0:h-y1,0:w-x1] = masks[:,y1:y2, x1:x2]
    return out

def extend_boxes(boxes, scale, im_width, im_height):
    '''
    Extend the boxes from the center to up down left right, the boxes doesn't change when scale=1
    im_width and im_height is for boxes cropping in case when the boxes are extended too much
    '''
    x1 = boxes[...,0].copy()
    y1 = boxes[...,1].copy()
    x2 = boxes[...,2].copy()
    y2 = boxes[...,3].copy()

    h_half = (y2 - y1) / 2
    w_half = (x2 - x1) / 2

    x1 -= w_half*(scale-1)
    y1 -= h_half*(scale-1)
    x2 += w_half*(scale-1)
    y2 += h_half*(scale-1)

    x1 = x1.clip(0, im_width)
    y1 = y1.clip(0, im_height)
    x2 = x2.clip(0, im_width)
    y2 = y2.clip(0, im_height)

    boxes[...,0] = x1
    boxes[...,1] = y1
    boxes[...,2] = x2
    boxes[...,3] = y2

    return boxes

def _hsv_color_jittering(img_hsv, h_range, s_range, v_range ):
    hsv_augs = np.random.uniform(-1, 1, 3) * [h_range, s_range, v_range]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

def hsv_color_jittering(im, h_range, s_range, v_range):
    if _is_pil_image(im):
        mode = im.mode
        im = im.convert('HSV')
        im_array = np.array(im)
        _hsv_color_jittering(im_array, h_range, s_range, v_range)
        im = Image.fromarray(im_array.astype('uint8'),mode='HSV').convert(mode)
        return im
    else:
        raise NotImplementedError('have not implement other image type')

