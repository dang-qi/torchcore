from pycocotools import mask

def get_binary_mask(mask_anno, height, width, use_compressed_rle=True):
    if isinstance(mask_anno, list):
        out = mask.decode(mask.merge(mask.frPyObjects(mask_anno, height, width)))
    elif isinstance(mask_anno, dict):
        if not use_compressed_rle:
            mask_anno = mask.frPyObjects(mask_anno, height, width)
        out = mask.decode(mask_anno)
    else:
        raise ValueError('unknow mask annotation')
    return out