'''code is revised from torchvision'''
import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models import ResNet as TorchResNet
from torchvision.ops import RoIAlign
from typing import Type, Any, Callable, Union, List, Optional
from .build import BACKBONE_REG
from ..base.norm import build_norm_layer
from ..base.conv import build_conv_layer
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.models.resnet import model_urls

#model_urls = {
#    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
#    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
#    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
#}

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channel, out_channel, stride=1, dilation=1, groups=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer_cfg=dict(type='BN')):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        _, self.bn1 = build_norm_layer(norm_layer_cfg, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        _, self.bn2 = build_norm_layer(norm_layer_cfg, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer_cfg=dict(type='BN')):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        _, self.bn1 = build_norm_layer(norm_layer_cfg, width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        _, self.bn2 = build_norm_layer(norm_layer_cfg, width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        _, self.bn3 = build_norm_layer(norm_layer_cfg, planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

@BACKBONE_REG.register()
class ResNet(TorchResNet):


    def __init__(self, 
                 depth, 
                 returned_layers = [1,2,3,4],
                 zero_init_residual=False,
                 groups=1, 
                 width_per_group=64, 
                 frozen_stage=-1,
                 norm_eval=True,
                 replace_stride_with_dilation=None,
                 norm_layer_cfg=dict(type='BN', requires_grad=True),
                 conv_layer_cfg=dict(type='Conv2d'),
                 init_cfg=dict(type='pretrained')):
        '''Resnet
           Parameters:
            depth(int or str): the depth of the ResNet

        '''
        super(TorchResNet, self).__init__()
        arch_by_depth = {
            18: (BasicBlock, (2, 2, 2, 2)),
            34: (BasicBlock, (3, 4, 6, 3)),
            50: (Bottleneck, (3, 4, 6, 3)),
            101: (Bottleneck, (3, 4, 23, 3)),
            152: (Bottleneck, (3, 8, 36, 3))
        }
        depth = int(depth)
        block = arch_by_depth[depth][0]
        layers = arch_by_depth[depth][1]
        self._norm_layer_cfg = norm_layer_cfg
        self._returned_layers = returned_layers

        self.frozen_stage = frozen_stage
        self.norm_eval = norm_eval

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.bn1 = norm_layer(self.inplanes)
        _, self.bn1 = build_norm_layer(norm_layer_cfg, self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.out_channel = self.inplanes

        init_cfg = init_cfg.copy()
        init_type = init_cfg.pop('type')

        if init_type == 'pretrained':
            progress = init_cfg.get('progress', True)
            arch = 'resnet{}'.format(depth)
            state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
            self.load_state_dict(state_dict, strict=False)
            print('init from pretrained model')
        elif init_type == 'kaiming':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)
        else:
            raise KeyError('Unsupported init type {}!'.format(init_type))
        
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stage >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stage + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer_cfg = self._norm_layer_cfg
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                build_norm_layer(norm_layer_cfg, planes*block.expansion)[1]
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer_cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer_cfg=norm_layer_cfg))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        layer0_out = self.relu(x)
        layer1_1_out = self.maxpool(layer0_out)

        layer1_out = self.layer1(layer1_1_out)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        outputs = [layer0_out, layer1_out, layer2_out, layer3_out, layer4_out]

        out = OrderedDict()
        for i in self._returned_layers:
            out['{}'.format(i)] = outputs[i]
        return out
    
    def forward(self, x):
        return self._forward_impl(x)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


class ResNetRoI(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        roi_w: int = 128,
        roi_h: int = 128,
        roi_layer: int = 2,
        returned_layers: list = [1,2,3,4],
        multi_feature: bool = True
    ) -> None:
        super(ResNetRoI, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.returned_layers = returned_layers

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.roi_pool = RoIAlign((roi_h,roi_w), spatial_scale=1/(2*2**roi_layer), sampling_ratio=2, aligned=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.roi_layer = roi_layer
        self.multi_feature = multi_feature

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, boxes: Optional[List[Tensor]]=None) -> Tensor:
        # See note [TorchScript super()]
        out = []
        #if boxes is not None:
        #    roi_num_per_batch = [len(roi) for roi in boxes]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        out.append(x)

        x = self.layer1(x)
        if boxes is not None and self.roi_layer == 1:
            x = self.roi_pool(x, boxes)
        out.append(x)
        x = self.layer2(x)
        if boxes is not None and self.roi_layer == 2:
            x = self.roi_pool(x, boxes)
        out.append(x)
        x = self.layer3(x)
        if boxes is not None and self.roi_layer == 3:
            x = self.roi_pool(x, boxes)
        out.append(x)
        x = self.layer4(x)
        if boxes is not None and self.roi_layer == 4:
            x = self.roi_pool(x, boxes)
        out.append(x)
        if self.multi_feature:
            out_feature = OrderedDict()
            for ind, i in enumerate(range(self.roi_layer,5)):
                #out_feature[str(ind)] = out[i]
                if i in self.returned_layers:
                    out_feature[str(i-1)] = out[i]
            #for o in out:
            #    print(o.shape)
            #for k,v in out_feature.items():
            #    print(k,v.shape)
            return out_feature
        else:
            return x

    def forward(self, x: Tensor, boxes: Tensor=None) -> Tensor:
        return self._forward_impl(x, boxes)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model

def _roi_resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetRoI(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model

def roi_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _roi_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)
