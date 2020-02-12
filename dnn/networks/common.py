import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    elif isinstance(m, nn.ConvTranspose2d):
        fill_up_weights(m)
    elif isinstance(m, nn.Dropout):
        pass
    else:
        print('{} is not initialized'.format(type(m)))

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 
