import numpy as np
import copy

from PIL import Image
from PIL import ImageOps

from tools import annot_tools
from tools import bbox_tools

class patch :
    def __init__( self, cfg, patch_info, mirrored=False ):
        self._cfg = cfg
        self._patch_info = patch_info
        self._name = patch_info['name']
        self._path = self._cfg.IMAGES_TMP % ( self._name )
        self._patchshape = patch_info['patch_shape'][:2]
        self._mirrored = mirrored

        self._data = {}
        self._properties = {}

    @property
    def is_mirrored( self ):
        return self._mirrored

    def mirrored( self ):
        mimage = type(self)( self._cfg, self._patch_info, not self._mirrored )
        #mimage.label = self.label
        return mimage

    @property
    def name( self ):
        return self._name

    @property
    def scale( self ):
        return self._scale

    @scale.setter
    def scale( self, s ):
        assert s>0, "Scale should be greater than 0"
        if s > 0 :
            self._scale = s

    @property
    def name( self ):
        return self._name

    @property
    def patch( self ):
        scale = self.scale
        img = Image.open( self._path )

        if self._mirrored :
            img = ImageOps.mirror( img )

        if scale != 1.0 :
            h,w = self.shape
            img = img.resize( [w,h], Image.BILINEAR )

        np_img = np.array( img )
        del img
        if len( np_img.shape ) == 2 :
            np_img = np.repeat(np_img[:, :, np.newaxis], 3, axis=2)

        return np_img

    @property
    def patch_PIL( self ):
        scale = self.scale
        img = Image.open( self._path )

        if self._mirrored :
            img = ImageOps.mirror( img )

        if scale != 1.0 :
            h,w = self.shape
            img = img.resize( [w,h], Image.BILINEAR )

        return img

    @property
    def im_PIL( self ):
        scale = self.scale
        img = Image.open( self._cfg.IMAGES_TMP % (self._imname) )

        if self._mirrored :
            img = ImageOps.mirror( img )

        if scale != 1.0 :
            h,w = self.shape
            img = img.resize( [w,h], Image.BILINEAR )

        return img


    @property
    def shape(self):
        # Output [ height, width ]
        patchshape = np.array( self._patchshape )
        patchshape = np.round( patchshape * self.scale ).astype( int )

        return patchshape
