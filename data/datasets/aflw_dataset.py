from .dataset import dataset
from ..image import aflw_image

class aflw_dataset( dataset ):
    def __init__( self, cfg, part, img_type=aflw_image ):
        super().__init__( cfg, img_type )

        self._dset_tag = 'aflw'
        self._part = part

        if part == 'train' :
            p = 'tr'
        else :
            p = 'te'

        self._dset_hash = 'AFLW%s' % ( p )
        self._data_name = 'aflw'
        self._images = []

    def load( self, setting=None ):
        super().load( setting=setting )

        # Updating the ground truth labels
        cls_labels = self._cfg.CLS_LABELS
        gt_label = self._cfg.GT_LABEL
        gt_label = cls_labels[ gt_label ]

        for image in self._original_images :
            image.label = gt_label
