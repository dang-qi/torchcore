import torch
from mmdet.models import build_detector
from mmcv import Config
from mmdet.apis import init_detector, inference_detector

def load_mm_retinanet(checkpoint_file, mm_config, model, return_mm_model=False):
    if checkpoint_file is None:
        mmcfg = Config.fromfile(mm_config)
        mm_model = build_detector(
                mmcfg.model,
                train_cfg=mmcfg.get('train_cfg'),
                test_cfg=mmcfg.get('test_cfg'))
        mm_model.init_weights()
    else:
        mm_model = init_detector(mm_config, checkpoint_file, device=next(model.parameters()).device)
    # backbone
    model.backbone.load_state_dict(mm_model.backbone.state_dict(),strict=True)

    # neck
    with torch.no_grad():
        for i in range(3):
            model.neck.inner_blocks[i].weight.copy_(mm_model.neck.lateral_convs[i].conv.weight)
            model.neck.layer_blocks[i].weight.copy_(mm_model.neck.fpn_convs[i].conv.weight)
            model.neck.inner_blocks[i].bias.copy_(mm_model.neck.lateral_convs[i].conv.bias)
            model.neck.layer_blocks[i].bias.copy_(mm_model.neck.fpn_convs[i].conv.bias)
        model.neck.extra_blocks.p6.weight.copy_(mm_model.neck.fpn_convs[3].conv.weight)
        model.neck.extra_blocks.p7.weight.copy_(mm_model.neck.fpn_convs[4].conv.weight)
        model.neck.extra_blocks.p6.bias.copy_(mm_model.neck.fpn_convs[3].conv.bias)
        model.neck.extra_blocks.p7.bias.copy_(mm_model.neck.fpn_convs[4].conv.bias)

    # head
    with torch.no_grad():
        for i in range(4):
            model.det_head.head.cls_head[2*i].weight.copy_(mm_model.bbox_head.cls_convs[i].conv.weight)
            model.det_head.head.bbox_head[2*i].weight.copy_(mm_model.bbox_head.reg_convs[i].conv.weight)
            model.det_head.head.cls_head[2*i].bias.copy_(mm_model.bbox_head.cls_convs[i].conv.bias)
            model.det_head.head.bbox_head[2*i].bias.copy_(mm_model.bbox_head.reg_convs[i].conv.bias)
        
        model.det_head.head.cls_head[8].weight.copy_(mm_model.bbox_head.retina_cls.weight)
        model.det_head.head.bbox_head[8].weight.copy_(mm_model.bbox_head.retina_reg.weight)
        model.det_head.head.cls_head[8].bias.copy_(mm_model.bbox_head.retina_cls.bias)
        model.det_head.head.bbox_head[8].bias.copy_(mm_model.bbox_head.retina_reg.bias)
    
    if return_mm_model:
        return mm_model
    else:
        del mm_model