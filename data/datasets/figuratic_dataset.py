from .dataset import dataset
from ..image import figuratic_image

class figuratic_dataset( dataset ):
    def __init__( self, cfg, part, img_type=figuratic_image ):
        super().__init__( cfg, img_type )
        self._dset_tag = 'figuratic'
        self._part = part

        self._dset_hash = 'FG%s' % ( self._part )
        self._data_name = 'figuratic'
        self._images = []

    def load( self, setting=None ):
        super().load( setting=setting )
