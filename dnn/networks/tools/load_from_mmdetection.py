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

def load_mm_fcos(checkpoint_file, mm_config, model, return_mm_model=False, change_backbone=False):
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
    if change_backbone:
        model.backbone = mm_model.backbone
    else:
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
            model.det_head.head.cls_conv[3*i].weight.copy_(mm_model.bbox_head.cls_convs[i].conv.weight)
            model.det_head.head.bbox_conv[3*i].weight.copy_(mm_model.bbox_head.reg_convs[i].conv.weight)
            model.det_head.head.cls_conv[3*i].bias.copy_(mm_model.bbox_head.cls_convs[i].conv.bias)
            model.det_head.head.bbox_conv[3*i].bias.copy_(mm_model.bbox_head.reg_convs[i].conv.bias)

            model.det_head.head.cls_conv[3*i+1].weight.copy_(mm_model.bbox_head.cls_convs[i].gn.weight)
            model.det_head.head.cls_conv[3*i+1].bias.copy_(mm_model.bbox_head.cls_convs[i].gn.bias)
            model.det_head.head.bbox_conv[3*i+1].weight.copy_(mm_model.bbox_head.reg_convs[i].gn.weight)
            model.det_head.head.bbox_conv[3*i+1].bias.copy_(mm_model.bbox_head.reg_convs[i].gn.bias)
        model.det_head.head.cls_logits.weight.copy_(mm_model.bbox_head.conv_cls.weight)
        model.det_head.head.cls_logits.bias.copy_(mm_model.bbox_head.conv_cls.bias)
        model.det_head.head.bbox_pred.weight.copy_(mm_model.bbox_head.conv_reg.weight)
        model.det_head.head.bbox_pred.bias.copy_(mm_model.bbox_head.conv_reg.bias)
        model.det_head.head.centerness_head.weight.copy_(mm_model.bbox_head.conv_centerness.weight)
        model.det_head.head.centerness_head.bias.copy_(mm_model.bbox_head.conv_centerness.bias)
        if model.det_head.enable_scale:
            for i in range(len(model.det_head.scales)):
                model.det_head.scales[i].scale.copy_(mm_model.bbox_head.scales[i].scale)
    
    if return_mm_model:
        return mm_model
    else:
        del mm_model