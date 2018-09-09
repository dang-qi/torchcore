import numpy as np
import copy

from .image import image
from tools import annot_tools

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
            if self._mirrored :
                annot_tools.mirror_keypoints( keypoints, self._imshape )
            keypoints[:,:,:2] *= scale

            labels = keypoints[:,:,2]
            keypoints = keypoints[:,:,:2] + self.padding

            return labels, keypoints
        else :
            return []

    @property
    def has_keypoints( self ):
        if 'keypoints' in self._data :
            keypoints = copy.deepcopy( self._data['keypoints'] )
            keypoints = keypoints.reshape([-1,17,3])
            labels = keypoints[:,:,2].ravel()

            inds = np.where( labels > 0 )[0]

            if len( inds ) > 0 :
                return True

        return False
