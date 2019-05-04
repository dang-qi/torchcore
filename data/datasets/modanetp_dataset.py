import numpy as np

from .dataset import dataset
from ..image import modanetp_image

class modanetp_dataset( dataset ):
    def __init__( self, cfg, part, img_type=modanetp_image ):
        super().__init__( cfg, img_type )
        self._dset_tag = 'modanetp'
        self._part = part

        self._dset_hash = 'Modap%s' % ( self._part )
        self._data_name = 'modanetp'
        self._images = []

    def load( self, settings=None ):
        super().load( settings=settings )

        keep = []
        for idx, image in enumerate( self._original_images ):
            if image.has_file :
                keep.append(idx)

        keep = np.array(keep)

        if len(keep) > 0 :
            self._original_images = self._original_images[keep]
