from .dataset import dataset
from ..image import coco_image

class cocowider_dataset( dataset ):
    def __init__( self, cfg, cls, part, img_type=coco_image ):
        super().__init__( cfg, img_type )

        self._dset_tag = 'cocowider_%s_%s' % ( cls, part )

        c = cls[:2]
        if part == 'tr14tr' :
            p = 'tr14'
        elif part == 'tr17tr' :
            p = 'tr17'
        elif part == 'va14va' :
            p = 'va14'
        elif part == 'va17va' :
            p = 'va17'
        else :
            p = 'te'

        self._dset_hash = 'CW%s' % ( p )
        self._data_name = 'cocowider'
        self._images = []

    def load( self, settings=None ):
        super().load( settings=settings )
