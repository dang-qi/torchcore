import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import common
from .util import load_state_dict_from_url, convert_state_dict_for_vgg16

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class Vgg16( nn.Module ):
    def _init_weights( self ):
        for name, m in self.named_modules() :
            if len( name ) > 0 :
                common.init(m)

    def __init__( self, feature=False, pretrained=False ):
        super().__init__()

        self._feature = feature

        self.conv1_1 = nn.Conv2d( 3, 64, kernel_size=3, padding=1 )
        self.conv1_2 = nn.Conv2d( 64, 64, kernel_size=3, padding=1 )

        self.conv2_1 = nn.Conv2d( 64, 128, kernel_size=3, padding=1 )
        self.conv2_2 = nn.Conv2d( 128, 128, kernel_size=3, padding=1 )

        self.conv3_1 = nn.Conv2d( 128, 256, kernel_size=3, padding=1 )
        self.conv3_2 = nn.Conv2d( 256, 256, kernel_size=3, padding=1 )
        self.conv3_3 = nn.Conv2d( 256, 256, kernel_size=3, padding=1 )

        self.conv4_1 = nn.Conv2d( 256, 512, kernel_size=3, padding=1 )
        self.conv4_2 = nn.Conv2d( 512, 512, kernel_size=3, padding=1 )
        self.conv4_3 = nn.Conv2d( 512, 512, kernel_size=3, padding=1 )

        self.conv5_1 = nn.Conv2d( 512, 512, kernel_size=3, padding=1 )
        self.conv5_2 = nn.Conv2d( 512, 512, kernel_size=3, padding=1 )
        self.conv5_3 = nn.Conv2d( 512, 512, kernel_size=3, padding=1 )

        if pretrained:
            print('load pretrained module for vgg16')
            state_dict = load_state_dict_from_url(model_urls['vgg16'],
                                              progress=True)
            state_dict = convert_state_dict_for_vgg16(state_dict)
            self.load_state_dict(state_dict, strict=False)
        else:
            self._init_weights()

    def forward( self, x ):
        x = F.relu( self.conv1_1(x) )
        x = F.relu( self.conv1_2(x) )
        x = F.max_pool2d(x,2)

        x = F.relu( self.conv2_1(x) )
        x = F.relu( self.conv2_2(x) )
        x = F.max_pool2d(x,2)

        x = F.relu( self.conv3_1(x) )
        x = F.relu( self.conv3_2(x) )
        x = F.relu( self.conv3_3(x) )
        x = F.max_pool2d(x,2)

        x = F.relu( self.conv4_1(x) )
        x = F.relu( self.conv4_2(x) )
        x = F.relu( self.conv4_3(x) )
        x = F.max_pool2d(x,2)

        x = F.relu( self.conv5_1(x) )
        x = F.relu( self.conv5_2(x) )
        x = F.relu( self.conv5_3(x) )

        if not self._feature :
            x = F.max_pool2d(x,2)

        return x
