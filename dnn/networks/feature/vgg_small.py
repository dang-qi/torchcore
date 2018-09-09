import torch
import torch.nn as nn
import torch.nn.functional as F

class VggSmall( nn.Module ):
    def __init__( self, feature=False ):
        super( VggSmall, self ).__init__()

        self._feature = feature

        self.conv1_1 = nn.Conv2d( 3, 64, kernel_size=3, padding=1 )
        self.conv1_2 = nn.Conv2d( 64, 64, kernel_size=3, padding=1 )

        self.conv2_1 = nn.Conv2d( 64, 128, kernel_size=3, padding=1 )
        self.conv2_2 = nn.Conv2d( 128, 128, kernel_size=3, padding=1 )

        self.conv3_1 = nn.Conv2d( 128, 256, kernel_size=3, padding=1 )

        self.conv4_1 = nn.Conv2d( 256, 512, kernel_size=3, padding=1 )

        self.conv5_1 = nn.Conv2d( 512, 512, kernel_size=3, padding=1 )

    def forward( self, x ):
        x = F.relu( self.conv1_1(x) )
        x = F.relu( self.conv1_2(x) )
        x = F.max_pool2d(x,2)

        x = F.relu( self.conv2_1(x) )
        x = F.relu( self.conv2_2(x) )
        x = F.max_pool2d(x,2)

        x = F.relu( self.conv3_1(x) )
        x = F.max_pool2d(x,2)

        x = F.relu( self.conv4_1(x) )
        x = F.max_pool2d(x,2)

        x = F.relu( self.conv5_1(x) )

        if not self._feature :
            x = F.max_pool2d(x,2)

        return x
