def convert_to_mmdet(inputs, targets):
    img_meta = [{'img_shape':(imsize[0],imsize[1],3),'scale_factor':(imscale[0],imscale[1],imscale[0],imscale[1]),'flip':False} for imsize,imscale in zip(inputs['image_sizes'],inputs['scale'])]
    for meta in img_meta:
        meta['pad_shape'] = (inputs['data'].shape[2],inputs['data'].shape[3],inputs['data'].shape[1])
    gt_boxes = [t['boxes'] for t in targets]
    gt_labels = [t['labels']-1 for t in targets]
    return img_meta, gt_boxes, gt_labels

#inputs['scale']
#inputs['image_sizes']
#inputs['data']
#
#{'img_metas': [DataContainer([[{'filename': 'data/coco/val2017/000000532481.jpg', 'ori_filename': '000000532481.jpg', 'ori_shape': (426, 640, 3) (h,w,3), 'img_shape': (800, 1202, 3), 'pad_shape': (800, 1216, 3), 'scale_factor': array([1.878125 , 1.8779342, 1.878125 , 1.8779342], dtype=float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}]])]}