from .dataset import dataset
from ..patch import aflw_patch

class aflw_patch_dataset( dataset ):
    def __init__( self, cfg, part, img_type=aflw_patch ):
        super().__init__( cfg, img_type )

        self._dset_tag = 'aflw_patch'
        self._part = part

        if part == 'train' :
            p = 'tr'
        else :
            p = 'te'

        self._dset_hash = 'AFLWPatch%s' % ( p )
        self._data_name = 'aflw_patch'
        self._images = []

    def load( self, settings=None, **kwargs ):
        super().load( settings=settings )
        # Updating the ground truth labels
