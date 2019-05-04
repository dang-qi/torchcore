import numpy as np
import io
import copy

from PIL import Image
from PIL import ImageOps

class image_deploy :
    def __init__( self, mirrored=False ):
        self._mirrored = mirrored
        self._imshape = None
        self._pil_image = None
        self.scale = 1.0
        self.label = -1

    def open( self, path ):
        img = Image.open( path )
        if self._mirrored :
            img = ImageOps.mirror( img )
        self._pil_image = img
        self._imshape = np.array(self._pil_image.size[::-1])

    def set( self, img ):
        if self._mirrored :
            img = ImageOps.mirror( img )
        self._pil_image = img
        self._imshape = np.array(self._pil_image.size[::-1])

    def read( self, image_buf ):
        img = Image.open( io.BytesIO(image_buf) )
        if self._mirrored :
            img = ImageOps.mirror( img )
        self._pil_image = img
        self._imshape = np.array(self._pil_image.size[::-1])

    @property
    def im( self ):
        scale = self.scale
        img = copy.deepcopy( self._pil_image )

        if scale != 1.0 :
            w = int(np.floor( img.size[0]*scale ))
            h = int(np.floor( img.size[1]*scale ))
            img = img.resize( [w,h], Image.BILINEAR )

        img = np.array( img )
        if len( img.shape ) == 2 :
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        return img

    @property
    def im_PIL( self ):
        scale = self.scale
        img = copy.deepcopy( self._pil_image )

        if self._mirrored :
            img = ImageOps.mirror( img )

        if scale != 1.0 :
            w = int(np.floor( img.size[0]*scale ))
            h = int(np.floor( img.size[1]*scale ))
            img = img.resize( [w,h], Image.BILINEAR )

        return img


    @property
    def shape( self ):
        imshape = np.array( self._imshape )
        imshape = np.floor( imshape * self.scale ).astype( int )
        return imshape
