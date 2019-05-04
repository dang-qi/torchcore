import numpy as np

from .dataset import dataset
from ..image import modanet_image

class modanet_dataset( dataset ):
    def __init__( self, cfg, part, img_type=modanet_image ):
        super().__init__( cfg, img_type )
        self._dset_tag = 'modanet'
        self._part = part

        self._dset_hash = 'Moda%s' % ( self._part )
        self._data_name = 'modanet'
        self._images = []

    def load( self, settings=None ):
        super().load( settings=settings )

        keep = []
        for idx, image in enumerate( self._original_images ):
            if image.has_file :
                keep.append(idx)

        keep = np.array(keep)
        self._original_images = self._original_images[keep]
