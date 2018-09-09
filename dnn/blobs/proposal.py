#import config
import numpy as np
#from config import dnn_cfg
#from proposal_module import proposal
from .blobs import data_blobs


# Images in the dataset are labeled accourding to the object that is present in them.
# The index of the objects are incremented from 1.
# The invalid objects are labeled as -1
# Since this section of the algorithm does not require any negative boxes, the label
# of the gtboxes will be -1, 1, 2, 3 and no label 0 will apear.
#
# This is to make multi-class classification easier. In a multiclass setup, we will
# have 0 : bg, 1 : cls1, 2 : cls2, ...
#
# This enables using softmax for multiclass classification, when we want to decide
# to which class a certain box blongs

class proposal_blobs( data_blobs ):
    def __init__( self, cfg ):
        super().__init__( cfg.dnn )
        #self._proposal = proposal()
        self._cfg = cfg
        self._dnn_cfg = self._cfg.dnn

    def _remove_small_gtboxes( self, blobs, images ):
        min_obj_size = self._cfg.dataset.MIN_OBJ_SIZE

        gtboxes = blobs['gtboxes']
        gtlabels = blobs['gtlabels']
        validboxes = blobs['validboxes']

        for ind, box in enumerate( gtboxes ):
            h = box[3] - box[1]
            w = box[2] - box[0]

            if h < min_obj_size[0] or w < min_obj_size[1] :
                gtlabels[ ind ] = -1
                validboxes[ ind ] = -1

        blobs['gtlabels'] = gtlabels
        blobs['validboxes'] = validboxes

    def get_blobs( self, images, remove_difficults=None ):
        if remove_difficults is None :
            remove_difficults = self._dnn_cfg.PROPOSAL.TRAIN.REMOVE_DIFFICULTS

        blobs = {}

        # Converting the data to the propper format
        self._add_data( images, blobs )

        # Adding ground truth boxes
        self._add_gtboxes( images, blobs, remove_difficults=remove_difficults )

        # Setting the label of small boxes to -1
        self._remove_small_gtboxes( blobs, images )

        return blobs

    def get_blobs_deploy( self, images, max_size=None ):
        blobs = {}
        self._add_data( images, blobs, max_size=max_size )
        return blobs

    def get_analytics_blobs( self, images, remove_difficults=None ):
        if remove_difficults is None :
            remove_difficults = self._dnn_cfg.PROPOSAL.TRAIN.REMOVE_DIFFICULTS

        assert len(images) == 1
        blobs = {}
        self._add_data_deploy( images[0], blobs )
        self._add_gtboxes( images, blobs, remove_difficults=remove_difficults )
        self._add_targets( images, blobs )
        return blobs

class proposal_blobs_deploy( proposal_blobs ):
    def get_blobs( self, image ):
        blobs = {}
        self._add_data_deploy( image, blobs )
        return blobs

    def get_analytics_blobs( self, images, remove_difficults=None ):
        if remove_difficults is None :
            remove_difficults = self._dnn_cfg.PROPOSAL.TRAIN.REMOVE_DIFFICULTS

        assert len(images) == 1
        blobs = {}
        self._add_data_deploy( images[0], blobs )
        self._add_gtboxes( images, blobs, remove_difficults=remove_difficults )
        self._add_targets( images, blobs )
        return blobs
