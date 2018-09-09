from .dataset import dataset
from ..image import deepfashion_image

class deepfashion_dataset( dataset ):
    def __init__( self, cfg, cls, part, img_type=deepfashion_image ):
        super().__init__( cfg, img_type )
        self._dset_tag = 'deepfashion_%s' % ( cls )
        self._part = part

        self._dset_hash = 'DF%s%s' % ( cls, self._part )
        self._data_name = 'deepfashion_%s' % ( cls )
        self._images = []

    def load( self, setting=None ):
        super().load( setting=setting )
