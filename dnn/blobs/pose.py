import numpy as np
from tools import bbox_tools
from .blobs import data_blobs

from tools import pil_tools
from PIL.ImageDraw import Draw

class pose_blobs( data_blobs ):
    def __init__( self, dnn_cfg, deploy ):
        super().__init__( dnn_cfg )
        self._detector_deploy = deploy

    def _add_data( self, images, blobs ):
        all_descriptors = []
        all_rois = []
        all_keypoints = []
        all_labels = []

        for image in images :
            dets = self._detector_deploy.eval( image, rescale=False )
            rois = dets['person']['rois']
            descs = dets['person']['descriptors']

            gtboxes = image.gtboxes
            labels, keypoints = image.keypoints

            nboxes = len(gtboxes)
            valid_boxes = []
            for ii, ll in enumerate( labels ):
                if len( np.where( ll > 0 )[0] ) > 0 :
                    valid_boxes.append( ii )
            valid_boxes = np.array( valid_boxes )

            gtboxes = gtboxes[ valid_boxes ]
            labels = labels[ valid_boxes ]
            keypoints = keypoints[ valid_boxes ]

            if len( rois ) > 0 :
                bbo = bbox_tools()
                overlaps = bbo.overlap_gpu( rois, gtboxes )

                maxes = np.max( overlaps, axis=1 )
                argmaxes = np.argmax( overlaps, axis=1 )

                keep = np.where( maxes >= 0.5 )[0]

                if len( keep ) > 0 :
                    maxes = maxes[keep]
                    argmaxes = argmaxes[keep]

                    all_descriptors.append( descs[keep] )
                    all_rois.append( rois[keep] )
                    all_keypoints.append( keypoints[argmaxes] )
                    all_labels.append( labels[argmaxes] )

            if len( all_descriptors ) > 0 :
                blobs['descriptors'] = np.concatenate( all_descriptors ).astype( np.float32 )
                blobs['rois'] = np.concatenate( all_rois ).astype( np.float32 )
                blobs['keypoints'] = np.concatenate( all_keypoints ).astype( np.float32 )
                blobs['labels'] = np.concatenate( all_labels ).astype( np.float32 )

    def _normalize_keypoints( self, images, blobs ):
        keypoints = blobs['keypoints']
        rois = blobs['rois']
        labels = blobs['labels']

        target_keypoints = []
        target_labels = []

        for roi, kp, ll in zip( rois, keypoints, labels ):
            cx = (roi[2] + roi[0])*0.5
            cy = (roi[3] + roi[1])*0.5
            w = roi[2] - roi[0]
            h = roi[3] - roi[1]

            tkp = np.zeros( kp.shape )
            tkp[:,0] = ( kp[:,0] - cx ) / w
            tkp[:,1] = ( kp[:,1] - cy ) / h

            ll = ll.ravel()
            tl = np.zeros( ll.shape )

            inds = np.where( ll > 0 )[0]
            tl[ inds ] = 1.0

            target_keypoints.append( tkp.reshape((-1,17,2)) )
            target_labels.append( tl.reshape((-1,17,1)) )

        blobs['target_keypoints'] = np.concatenate( target_keypoints ).astype( np.float32 )
        blobs['target_labels'] = np.concatenate( target_labels ).astype( np.float32 )

    def _prune( self, images, blobs ):
        keypoints = blobs['target_keypoints']
        labels = blobs['target_labels']
        descriptors = blobs['descriptors']

        batch_size = self._dnn_cfg.POSE.TRAIN.SELECTION_BATCH_SIZE

        if len(keypoints) > batch_size :
            inds = np.arange(len(keypoints))
            keep = np.random.choice( inds, batch_size )

            blobs['descriptors'] = descriptors[keep]
            blobs['target_keypoints'] = keypoints[keep]
            blobs['target_labels'] = labels[keep]

    def _build_mask( self, images, blobs ):
        keypoints = blobs['target_keypoints']
        labels = blobs['target_labels']

        mask = np.zeros( keypoints.shape )

        mask = mask.reshape((-1,2))
        labels = labels.ravel()

        assert len(labels) == len(mask)

        inds = np.where( labels == 1 )[0]
        mask[ inds, : ] = 1.0

        mask = mask.reshape( keypoints.shape )

        blobs['target_mask'] = mask.astype( np.float32 )
        blobs['target_keypoints'] = np.multiply( blobs['target_keypoints'], blobs['target_mask'] )

    def _one_hot( self, images, blobs ):
        labels = blobs['target_labels']
        shape = list(labels.shape)

        labels = labels.ravel()
        oh = np.zeros( (len(labels),2),dtype=np.float32 )

        for i,l in enumerate(labels):
            oh[i,int(l)] = 1.0

        shape[-1] = 2
        oh = oh.reshape( shape )

        blobs['target_labels'] = oh

    def get_blobs( self, images ):
        blobs = {}

        self._add_data( images, blobs )

        if not 'descriptors' in blobs :
            blobs['valid'] = False
            return blobs

        self._normalize_keypoints( images, blobs )
        self._prune( images, blobs )
        self._build_mask( images, blobs )
        self._one_hot( images, blobs )

        blobs['dropout_prob'] = 0.5
        blobs['valid'] = True

        return blobs
