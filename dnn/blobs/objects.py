import numpy as np
from .blobs import data_blobs

class objects_blobs( data_blobs ):
    def __init__( self, cfg ):
        super().__init__( cfg.dnn )
        #self._proposal = proposal()
        self._cfg = cfg
        self._dnn_cfg = self._cfg.dnn

    def _remove_small_gtboxes( self, blobs, images ):
        min_obj_size = self._cfg.dataset.MIN_OBJ_SIZE

        gtboxes = blobs['gtboxes']
        gtparts = blobs['gtparts']

        for ind, box in enumerate( gtboxes ):
            h = box[3] - box[1]
            w = box[2] - box[0]

            if h < min_obj_size[0] or w < min_obj_size[1] :
                gtparts[ ind ] = -1

        blobs['gtparts'] = gtparts

    def _gen_proposal_labels( self, blobs, images ):
        gtboxes = blobs['gtboxes']
        labels = np.ones((len(gtboxes,)))
        blobs['gtparts'] = labels.astype( np.float32 ).reshape([-1,1])

    def _add_pose( self, blobs, images ):
        keypoints = []
        labels = []

        for image in images :
            l,k = image.keypoints
            labels.append( l )
            keypoints.append( k )

        keypoints = np.concatenate( keypoints )
        labels = np.concatenate( labels )

        blobs['gtpose'] = keypoints
        blobs['gtposelabels'] = labels

    def get_blobs( self, images ):
        blobs = {}

        # Converting the data to the propper format
        self._add_data( images, blobs )

        # Adding ground truth boxes
        self._add_gtboxes( images, blobs )

        # Generating proposal labels
        self._gen_proposal_labels( blobs, images )

        # Setting the label of small boxes to -1
        self._remove_small_gtboxes( blobs, images )

        # Adding keypoints
        # self._add_pose( blobs, images )

        return blobs
