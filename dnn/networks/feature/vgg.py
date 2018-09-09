import torch
import torch.nn as nn
import torch.functional as F

from .. import common

def vgg_small():
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 'M', 512, 'M', 512]
    return common.make_layers( cfg )

def vgg16():
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    return common.make_layers( cfg )
