from .dataset import dataset
from ..image import fire_image

class fire_dataset( dataset ):
    def __init__( self, cfg, part, img_type=fire_image ):
        super().__init__( cfg, img_type )
        self._dset_tag = 'fire'
        self._part = part

        self._dset_hash = 'F%s' % ( self._part )
        self._data_name = 'fire'
        self._images = []

    def load( self, settings=None ):
        super().load( settings=settings )
