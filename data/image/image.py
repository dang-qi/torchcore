import numpy as np
import copy

from PIL import Image
from PIL import ImageOps

from tools import annot_tools
from tools import bbox_tools

class image :
    def __init__( self, cfg, image_info, mirrored=False ):
        self._cfg = cfg
        self._image_info = image_info
        self._name = image_info['name']
        self._path = self._cfg.IMAGES_TMP % ( self._name )
        self._imshape = image_info['im_shape'][:2]
        self._mirrored = mirrored

        self._data = {}
        self._properties = {}

        self.scale = 1.0
        self.padding = 0.0
        self.label = -1

    @property
    def is_mirrored( self ):
        return self._mirrored

    def mirrored( self ):
        mimage = type(self)( self._cfg, self._image_info, not self._mirrored )
        mimage.label = self.label
        return mimage

    @property
    def name( self ):
        return self._name

    @property
    def im( self ):
        scale = self.scale
        img = Image.open( self._path )

        if self._mirrored :
            img = ImageOps.mirror( img )

        if scale != 1.0 :
            w = int(np.floor( img.size[0]*scale ))
            h = int(np.floor( img.size[1]*scale ))
            img = img.resize( [w,h], Image.BILINEAR )

        if self.padding > 0 :
            img = ImageOps.expand( img, border=self.padding )

        np_img = np.array( img )
        del img
        if len( np_img.shape ) == 2 :
            np_img = np.repeat(np_img[:, :, np.newaxis], 3, axis=2)

        return np_img

    @property
    def im_PIL( self ):
        scale = self.scale
        img = Image.open( self._path )

        if self._mirrored :
            img = ImageOps.mirror( img )

        if scale != 1.0 :
            w = int(np.floor( img.size[0]*scale ))
            h = int(np.floor( img.size[1]*scale ))
            img = img.resize( [w,h], Image.BILINEAR )

        if self.padding > 0 :
            img = ImageOps.expand( img, self.padding )

        return img

    @property
    def shape(self):
        # Output [ height, width ]
        imshape = np.array( self._imshape )
        imshape = np.floor( imshape * self.scale ).astype( int )
        imshape = imshape + 2*self.padding

        return imshape


    @property
    def gtboxes( self ):
        if 'gtboxes' in self._data :
            scale = self.scale
            gtboxes = copy.deepcopy( self._data['gtboxes'] )
            if self._mirrored and len(gtboxes) > 0 :
                gtboxes = annot_tools.mirror_boxes( gtboxes, self._imshape )
            gtboxes *= scale
            gtboxes = gtboxes + self.padding

            if len( gtboxes ) > 0 :
                bbo = bbox_tools()
                gtboxes = bbo.clip( gtboxes, self.shape )

            return gtboxes
        else :
            return np.array([])

    @property
    def gtlabels( self ):
        if 'gtlabels' in self._data :
            gtlabels = copy.deepcopy( self._data['gtlabels'] )
            return np.array( gtlabels )
        else :
            return np.array([])

    def size_check( self, min_size ):
        gtboxes = self.gtboxes

        check = False
        for box in gtboxes :
            h = box[3] - box[1]
            w = box[2] - box[0]

            if h >= min_size[0] and w >= min_size[1] :
                check = True
                break

        return check
