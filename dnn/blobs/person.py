import numpy as np
from .blobs import blobs

class person_blobs( blobs ):
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

        def bin_data( values, thresholds ):
            bins = []
            bins.append(np.where( values < thresholds[0] )[0])
            for i in range( len(thresholds)-1 ):
                bins.append(np.where( ( values >= thresholds[i] ) & ( values < thresholds[i+1] ) )[0])
            bins.append(np.where(values >= thresholds[-1])[0] )
            return bins

        min_obj_size = self._cfg.dataset.MIN_OBJ_SIZE
        gtboxes = blobs['gtboxes']
        labels = np.zeros((len(gtboxes,)))

        heights = gtboxes[:,3] - gtboxes[:,1]
        widths = gtboxes[:,2] - gtboxes[:,0]

        small_inds = np.where( ( heights < min_obj_size[0] ) | (widths < min_obj_size[1] ) )[0]
        large_inds = np.where( ( heights >= min_obj_size[0] ) & ( widths >= min_obj_size[1] ) )[0]

        labels[ small_inds ] = -1

        heights = heights[ large_inds ]
        widths = widths[ large_inds ]

        areas = heights * widths
        aspects = heights / widths

        area_thresh = self._dnn_cfg.PROPOSAL.AREA_THRESH
        aspect_thresh = self._dnn_cfg.PROPOSAL.ASPECT_THRESH

        labels = np.zeros((len(gtboxes,)))

        area_bins = bin_data( areas, area_thresh )
        aspect_bins = bin_data( aspects, aspect_thresh )

        for i, arbin in enumerate( area_bins ):
            for j, asbin in enumerate( aspect_bins ):
                set_arbin = set( arbin )
                set_asbin = set( asbin )
                inds = list( set_arbin.intersection( set_asbin ) )

                # Adding one because objects start at 1 and 0 is background
                labels[ large_inds[inds] ] =  i * len(aspect_bins) + j + 1

        blobs['gtparts'] = labels.astype( np.float32 ).reshape([-1,1])

    def _add_pose( self, blobs, images ):
        keypoints = []
        labels = []

        for image in images :
            ll,kk = image.keypoints

            for l,k in zip( ll,kk ):
                if np.max(l) == 0 :
                    l.fill(-1)

                labels.append(l)
                keypoints.append(k)

        keypoints = np.array( keypoints )
        labels = np.array( labels )

        blobs['posepoints'] = keypoints
        blobs['poselabels'] = labels

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

    def get_blobs_deploy( self, images ):
        blobs = {}

        # Converting the data to the propper format
        self._add_data( images, blobs )

        return blobs
