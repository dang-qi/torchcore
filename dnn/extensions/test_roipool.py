import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn

from PIL import Image
from modules import RoiPool

class TestRoiPool :
    def _load_data( self ):
        pathdir = os.path.dirname( os.path.realpath(__file__))
        image_names = [ 'choco.png', 'snow.png' ]

        self._data = []

        num_rand_boxes = 10

        for name in image_names :
            img_path = os.path.join( pathdir, 'tests/data', name )
            img = Image.open( img_path )

            img_data = {}
            img_data['image'] = img
            img_data['boxes'] = []

            width = img.size[0]
            height = img.size[1]

            for i in range( num_rand_boxes ):
                xs = np.sort( np.random.randint( low=0, high=width, size=2 ) )
                ys = np.sort( np.random.randint( low=0, high=height, size=2 ) )

                bbox = [ xs[0], ys[0], xs[1], ys[1] ]
                img_data['boxes'].append( bbox )

            img_data['boxes'] = np.array( img_data['boxes'] )
            self._data.append( img_data )

    def _pytorch_blobs( self, device ):
        data = []
        boxes = []
        box_indices = []

        toTensor = transforms.ToTensor()

        for idx, img_data in enumerate( self._data ):
            data.append( toTensor( img_data['image'] ) )
            boxes.append( img_data['boxes'] )
            box_indices.append( np.ones( len(img_data['boxes']) )*idx )

        boxes = np.concatenate( boxes ).astype( np.float32 )
        box_indices = np.concatenate( box_indices ).astype( np.int32 )

        blobs = {}
        blobs['data'] = torch.stack( data ).to( device )
        blobs['rois'] = torch.from_numpy( boxes ).to( device )
        blobs['roibatches'] = torch.from_numpy( box_indices ).to( device )

        return blobs

    def _pytorch_out( self, device ):
        blobs = self._pytorch_blobs( device )
        pooling = RoiPool(7,7)
        #linear = nn.Linear(147,2)
        out = pooling( blobs['data'], blobs['rois'], blobs['roibatches'], spatial_scale=1.0 )

        out = out.detach().cpu().numpy()

        #grads_out = torch.from_numpy(np.random.rand( *out.shape ).astype(np.float32))
        #back = pooling.backward( grads_out )

        #print( out.shape )
        return out

    def __init__( self ):
        self._load_data()

    def perform_test( self, device='cpu' ):
        device = torch.device( device )
        out = self._pytorch_out( device )

        return out
        #pt_crops = self._pytorch_out( device=device ).transpose((0,2,3,1))
        #tf_crops = self._tf_out()

        #print( pt_crops.shape )
        #print( tf_crops.shape )

        #print(np.sum(np.abs(pt_crops - tf_crops)))
