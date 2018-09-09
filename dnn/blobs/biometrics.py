#import config
import numpy as np
#fdnnrom config import dnn_cfg
from tools import bbox_tools
from .blobs import blobs

from tools import pil_tools
from PIL.ImageDraw import Draw

class biometrics_blobs( blobs ):
    def __init__( self, dnn_cfg, deploy ):
        super().__init__( dnn_cfg )
        self._detector_deploy = deploy

    def _one_hot( self, labels ):
        labels = labels.ravel()
        oh = np.zeros( (len(labels),2),dtype=np.float32 )

        for i,l in enumerate(labels):
            oh[i,int(l)] = 1.0

        return oh

    def _add_data( self, images, blobs ):
        descriptors = []
        pose = []
        gender = []

        for ind, image in enumerate(images) :
            dets = self._detector_deploy.eval( image )
            dets = dets['face']

            rois = dets['rois']
            descs = dets['descriptors']
            gtboxes = image.gtboxes

            if len( rois ) > 0 :
                bbo = bbox_tools()
                overlaps = bbo.overlap_gpu( rois, gtboxes )

                maxes = np.max( overlaps, axis=1 )
                argmaxes = np.argmax( overlaps, axis=1 )

                keep = np.where( maxes >= 0.5 )[0]

                if len( keep ) > 0 :
                    maxes = maxes[keep]
                    argmaxes = argmaxes[keep]

                    descriptors.append( descs[keep] )
                    pose.append( image.pose[ argmaxes ] )
                    gender.append( image.gender[ argmaxes ] )

        if len( descriptors ) > 0 :
            blobs['descriptors'] = np.concatenate( descriptors ).astype(np.float32)
            blobs['pose'] = np.concatenate( pose ).astype(np.float32)
            blobs['gender'] = self._one_hot( np.concatenate( gender ).astype(np.float32) )

    def _prune( self, images, blobs ):
        descriptors = blobs['descriptors']
        ndata = descriptors.shape[0]

        if ndata > self._dnn_cfg.BIOMETRICS.SELECTION_BATCH_SIZE :
            inds = np.arange( ndata )
            keep = np.random.choice( inds, self._dnn_cfg.BIOMETRICS.SELECTION_BATCH_SIZE )

            blobs['descriptors'] = blobs['descriptors'][ keep ]
            blobs['pose'] = blobs['pose'][ keep ]
            blobs['gender'] = blobs['gender'][ keep ]

    def get_blobs( self, images ):
        blobs = {}

        self._add_data( images, blobs )

        if not 'descriptors' in blobs :
            blobs['valid'] = False
            return blobs
        else :
            blobs['valid'] = True

        self._prune( images, blobs )

        blobs['dropout_prob'] = 0.5

        return blobs
