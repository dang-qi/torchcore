from turtle import forward
from typing import OrderedDict
import torch
from torch import nn
from ..base import ConvBlock
from .build import BACKBONE_REG
from torch.nn.modules.batchnorm import _BatchNorm

class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='leaky_relu', negative_slope=0.1),
                 ):
        super().__init__()
        assert in_channels %2 == 0
        half_in_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, half_in_channels,1,conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvBlock(half_in_channels, in_channels,3,padding=1,conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        return x+out

@BACKBONE_REG.register(force=True)
class DarkNet(nn.Module):
    def __init__(self,
                 depth, 
                 returned_layers = [3,4,5],
                 frozen_stage=-1,
                 norm_eval=True,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 conv_cfg=dict(type='Conv2d'),
                 act_cfg=dict(type='leaky_relu', negative_slope=0.1),
                 init_cfg=dict(type='kaiming')):
        super().__init__()
        arch_by_depth={53:((1,2,8,8,4), ((32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)))}
        depth = int(depth)
        self.block_cfg = dict(conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.conv1 = ConvBlock(3,32,3,padding=1,**self.block_cfg)
        layer_num, filter_num = arch_by_depth[depth]
        self.layer_names = ['conv1']
        for i, (l, f) in enumerate(zip(layer_num, filter_num)):
            name = 'conv_res_block{}'.format(i+1)
            in_c, out_c = f
            self.add_module(name, self.make_conv_res_block(in_c, out_c, l))
            self.layer_names.append(name)
        self.norm_eval = norm_eval
        self.return_layers= returned_layers
        self.froze_stage = frozen_stage

        init_cfg = init_cfg.copy()
        init_type = init_cfg.pop('type')

        if init_type == 'pretrained':
            raise ValueError('no pretrained net available yet')
            #progress = init_cfg.get('progress', True)
            #arch = 'resnet{}'.format(depth)
            #state_dict = load_state_dict_from_url(model_urls[arch],
            #                                  progress=progress)
            #self.load_state_dict(state_dict, strict=False)
            #print('init from pretrained model')
        elif init_type == 'kaiming':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',a=act_cfg.get('negative_slope',0), nonlinearity=act_cfg['type'])
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise KeyError('Unsupported init type {}!'.format(init_type))

    def forward(self,x):
        out = OrderedDict()
        for i, name in enumerate(self.layer_names):
            x= getattr(self, name)(x)
            if i in self.return_layers:
                out[i] = x
        return out

        
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            # the first is the conv, so we plus 1 here
            for i in range(self.frozen_stages+1):
                m = getattr(self, self.layer_names[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(DarkNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()



    def make_conv_res_block(self,in_channels,out_channels,res_repeat):
        blcok = nn.Sequential()
        blcok.add_module('conv',
                        ConvBlock(in_channels,out_channels,3,stride=2, padding=1,**self.block_cfg))
        for i in range(res_repeat):
            blcok.add_module('res{}'.format(i),
                             ResBlock(out_channels, **self.block_cfg))
        return blcok
