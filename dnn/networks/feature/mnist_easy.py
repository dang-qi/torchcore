import torch.nn as nn
import torch
import torch.nn.functional as F
from dnn.networks import common

class EasyNet(nn.Module):
    def _init_weights( self ):
        for name, m in self.named_modules() :
            if len( name ) > 0 :
                common.init(m)
    def __init__(self):
        super().__init__()

class EasyNetA(EasyNet):
    def __init__(self, feature=True, RGB=False):
        super().__init__()
        if RGB:
            self.conv1_1 = nn.Conv2d(3,32, kernel_size=5, stride=1)
        else:
            self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        #self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(32, 50, kernel_size=5, stride=1)
        #self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #self.conv2_3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self._init_weights()
    
    def forward(self,input):
        x = F.relu(self.conv1_1(input))
        #x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.conv2_1(x))
        #x = F.relu(self.conv2_2(x))
        #x = F.relu(self.conv2_3(x))
        x = F.max_pool2d(x, (2,2))
        return x

class EasyNetB(EasyNet):
    def __init__(self, feature=True,RGB=False):
        super().__init__()
        if RGB:
            self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #self.conv2_3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self._init_weights()
    
    def forward(self,input):
        x = F.relu(self.conv1_1(input))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        #x = F.relu(self.conv2_3(x))
        x = F.max_pool2d(x, (2,2))
        return x

class EasyNetC(EasyNet):
    def __init__(self, feature=True,RGB=False):
        super().__init__()
        if RGB:
            self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self._init_weights()
    
    def forward(self,input):
        x = F.relu(self.conv1_1(input))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = F.max_pool2d(x, (2,2))
        return x

class EasyNetD(EasyNet):
    def __init__(self, feature=True,RGB=False):
        super().__init__()
        if RGB:
            self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self._init_weights()
    
    def forward(self,input):
        x = F.relu(self.conv1_1(input))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = F.max_pool2d(x, (2,2))
        return x