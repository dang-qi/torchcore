from .dataset import dataset
from ..image import deepfashionp_image

class deepfashionp_dataset( dataset ):
    def __init__( self, cfg, cls, part, img_type=deepfashionp_image ):
        super().__init__( cfg, img_type )
        self._dset_tag = 'deepfashionp_%s' % ( cls )
        self._part = part

        self._dset_hash = 'DFP%s%s' % ( cls, self._part )
        self._data_name = 'deepfashionp_%s' % ( cls )
        self._images = []

    def load( self, settings=None ):
        super().load( settings=settings )
