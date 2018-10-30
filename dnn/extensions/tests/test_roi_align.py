import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import tensorflow as tf

from modules import RoiAlign

np.random.seed( 1234 )

def convert_rois( rois ):
    order = np.array( [1,0,3,2] )
    rois = rois[:,order]
    return rois

def normalize_rois( rois, grid_size ) :
    height = grid_size[1]
    rois = rois / ( height-1 )
    return rois

def prepare_rois( rois, grid_size ):
    rois = normalize_rois( rois, grid_size )
    return convert_rois( rois )

class TestRoiAlign :
    def _load_data( self ):
        pathdir = os.path.dirname( os.path.realpath(__file__))
        image_names = [ 'choco.png', 'snow.png' ]

        self._data = []

        num_rand_boxes = 10

        for name in image_names :
            img_path = os.path.join( pathdir, 'data', name )
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
        blobs['boxes'] = torch.from_numpy( boxes ).to( device )
        blobs['box_indices'] = torch.from_numpy( box_indices ).to( device )

        return blobs

    def _tf_blobs( self ):
        data = []
        boxes = []
        box_indices = []

        for idx, img_data in enumerate( self._data ):
            data.append( np.array( img_data['image'], dtype=np.float32 ) / 255 )
            boxes.append( img_data['boxes'] )
            box_indices.append( np.ones( len(img_data['boxes']) )*idx )

        boxes = np.concatenate( boxes ).astype( np.float32 )
        box_indices = np.concatenate( box_indices ).astype( np.int32 )

        blobs = {}
        blobs['data'] = np.array( data, dtype=np.float32 )
        blobs['boxes'] = prepare_rois( np.array( boxes, dtype=np.float32 ), [500,500] )
        blobs['box_indices'] = np.array( box_indices, dtype=np.int32 )
        blobs['grad'] = np.random.rand(20,7,7,3).astype(np.float32)
        return blobs

    def _pytorch_out( self, device ):
        blobs = self._pytorch_blobs( device )
        pooling = RoiAlign(7,7,transform_fpcoor=False)
        out = pooling( blobs['data'], blobs['boxes'], blobs['box_indices'] )
        out = out.cpu().numpy()

        grads_out = torch.from_numpy(np.random.rand( *out.shape ).astype(np.float32))
        #back = pooling.backward( grads_out )

        print( out.shape )
        return out

    def _tf_out( self ):
        def feed_dict( inputs, blobs ):
            fd = {}
            for k,t in inputs.items() :
                fd[t] = blobs[k]
            return fd

        blobs = self._tf_blobs()

        inputs = {}
        inputs['data'] = tf.placeholder( tf.float32, shape=[2,500,500,3] )
        inputs['boxes'] = tf.placeholder( tf.float32, shape=[20,4] )
        inputs['box_indices'] = tf.placeholder( tf.int32, shape=[20] )
        inputs['grad'] = tf.placeholder(tf.float32, shape=[20,7,7,3])

        crops = tf.image.crop_and_resize( inputs['data'], inputs['boxes'], inputs['box_indices'], [7,7] )
        grad = tf.gradients( crops, inputs['grad'] )

        with tf.Session() as sess :
            res = sess.run( [ crops, grad ], feed_dict=feed_dict( inputs, blobs ) )

<<<<<<< HEAD
        print( res.shape )
        return res
=======
        res_crops = res[0]
        res_grad = res[1]

        return res_crops
>>>>>>> 746bd536e5420b3d429b215dea3dc4b1879e85d0

    def __init__( self ):
        self._load_data()

    def perform_test( self, device='cpu' ):
        device = torch.device( device )

        pt_crops = self._pytorch_out( device=device ).transpose((0,2,3,1))
        tf_crops = self._tf_out()

        print(np.sum(np.abs(pt_crops - tf_crops)))
