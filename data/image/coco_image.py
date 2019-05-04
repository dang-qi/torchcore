import numpy as np
import copy

from .image import image
from tools import annot_tools

keypoints_mirror_mapping = np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15])

class coco_image( image ):
    def __init__( self, cfg, image_info, mirrored=False ):
        super().__init__( cfg, image_info, mirrored )
        self.label = image_info.get('label')
        self._data['gtboxes'] = []
        self._data['gtlabels'] = []
        self._data['keypoints'] = []

        for obj in self._image_info['objects'] :
            self._data['gtboxes'].append( obj.get('bbox') )
            self._data['gtlabels'].append( obj.get('label') )
            self._data['keypoints'].append( obj.get('keypoints',np.zeros(51) ) )

        self._data['gtboxes'] = np.array( self._data['gtboxes'], dtype=np.float32 )
        self._data['gtlabels'] = np.array( self._data['gtlabels'], dtype=np.float32 )
        self._data['keypoints'] = np.array( self._data['keypoints'], dtype=np.float32 )

    @property
    def keypoints( self ):
        if 'keypoints' in self._data :
            scale = self.scale
            keypoints = copy.deepcopy( self._data['keypoints'] )
            keypoints = keypoints.reshape([-1,17,3])

            labels = keypoints[:,:,2]
            keypoints = keypoints[:,:,:2]

            nobjs = labels.shape[0]

            for i in range(nobjs):
                if np.max(labels[i]) == 0 :
                    labels[i] = -1

            if self.is_mirrored :
                annot_tools.mirror_keypoints( keypoints, self._imshape )
                # We need to apply mirror mapping to both labels and the
                # keypoints here

                labels = labels[:,keypoints_mirror_mapping]
                keypoints = keypoints[:,keypoints_mirror_mapping,:]

            keypoints = keypoints*scale + self.padding

            if self.is_mirrored :
                # We need to set keypoints that are not visible to 0

                labels_shape = labels.shape
                keypoints_shape = keypoints.shape

                labels = labels.reshape([-1,1]).ravel()
                keypoints = keypoints.reshape([-1,2])

                inds = np.where( labels<=0 )[0]
                keypoints[inds] = 0

                labels = labels.reshape(labels_shape)
                keypoints = keypoints.reshape(keypoints_shape)

            return labels, keypoints
        else :
            return []

    @property
    def has_keypoints( self ):
        if 'keypoints' in self._data :
            keypoints = copy.deepcopy( self._data['keypoints'] )
            keypoints = keypoints.reshape([-1,17,3])
            labels = keypoints[:,:,2].ravel()
            if len(labels) > 0 :
                if np.max(labels) > 0 :
                    return True
        return False
