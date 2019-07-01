from .dataset import dataset
from ..image import deepfashion2_image

class deepfashion2_dataset( dataset ):
    def __init__( self, cfg, part, img_type=deepfashion2_image ):
        super().__init__( cfg, img_type )
        self._dset_tag = 'deepfashion2' 
        self._part = part

        self._dset_hash = 'DF2%s' % ( self._part )
        self._data_name = 'deepfashion2' 
        self._images = []

    def load( self, settings=None ):
        super().load( settings=settings )
