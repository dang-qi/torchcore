import numpy as np
from .dataset import dataset
from ..patch import coco_person_patch

class coco_person_patch_dataset( dataset ):
    def __init__( self, cfg, part, img_type=coco_person_patch ):
        super().__init__( cfg, img_type )

        self._dset_tag = 'coco_person_patch'
        self._part = part

        if part == 'train' :
            p = 'tr'
        elif part == 'val' :
            p = 'va'
        else :
            p = 'te'

        self._dset_hash = 'CPEPatch%s' % ( p )
        self._data_name = 'coco_person_patch'
        self._images = []

    def _prune( self ):
        images = self.images

        keep = []
        for idx, image in enumerate(images) :
            labels = image.keypoint_labels.ravel()

            if np.max( labels ) > 0 :
                keep.append(idx)

        self._workset = images[keep]

    def load( self, settings=None, **kwargs ):
        super().load( settings=settings )
        # Updating the ground truth labels
