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
                oh = int(size * h / w)
                scale = size / w
                return img.resize((ow, oh), interpolation), scale
            else:
                oh = size
                ow = int(size * w / h)
                scale = size / h
                return img.resize((ow, oh), interpolation), scale
        else:
            if (w >= h and w == size) or (h >= w and h == size):
                return img, 1.0
            if w > h:
                ow = size
                oh = int(size * h / w)
                scale = size / w
                return img.resize((ow, oh), interpolation), scale
            else:
                oh = size
                ow = int(size * w / h)
                scale = size / h
                return img.resize((ow, oh), interpolation), scale

    else:
        scale_w = size[1] / w
        scale_h = size[0] / h
        return img.resize(size[::-1], interpolation), (scale_w, scale_h)

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

def pad(img, padding, fill=0, padding_mode='constant'): # this is from pytorch
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

def mirror(image):
    '''morror an image horizontally'''
    return ImageOps.mirror(image)

def mirror_boxes(boxes, im_width):
    boxes[:,0], boxes[:,2] = im_width -boxes[:,2], im_width -boxes[:,0]
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
    boxes[:,0] -= position[0]
    boxes[:,2] -= position[0]
    boxes[:,1] -= position[1]
    boxes[:,3] -= position[1]
    boxes[:,0] = np.clip(boxes[:,0], 0, position[2]-position[0])
    boxes[:,2] = np.clip(boxes[:,2], 0, position[2]-position[0])
    boxes[:,1] = np.clip(boxes[:,1], 0, position[3]-position[1])
    boxes[:,3] = np.clip(boxes[:,3], 0, position[3]-position[1])
    return boxes

def scale(image, scale):
    width, height = image.size
    width = int(width*scale)
    height = int(height*scale)
    return image.resize((width, height))

def scale_box(boxes, scale):
    return boxes*scale
