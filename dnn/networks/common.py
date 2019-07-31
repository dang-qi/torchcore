import torch
import torch.nn as nn
import torch.nn.functional as F

def make_layers( cfg, in_channels=3 ):
    layers = []
    for v in cfg :
        if v == 'M' :
            layers += [ nn.MaxPool2d( kernel_size=2, stride=2 ) ]
        else :
            conv2d = nn.Conv2d( in_channels, v, kernel_size=3, padding=1 )
            layers += [ conv2d, nn.ReLU( inplace=True ) ]
            in_channels=v

    return nn.Sequential( *layers )

def init( m ):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Dropout):
        pass
    else:
        print('{} is not initialized'.format(type(m)))
