import numpy as np
import copy
from pprint import pprint

from .image import image
from ...tools import annot_tools

# For landmarks
# Types :
#       1 : upper-body clothes, 6 Landmarks
#               ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]
#               mirror_mapping = [1,0,3,2,5,4]
#       2 : lower-body clothes, 4 Landmarks
#               ["left waistline", "right waistline", "left hem", "right hem"]
#               mirror_mapping = [1,0,3,2]
#       3 : full-body clothes, 8 Landmarks
#               ["left collar", "right collar", "left sleeve", "right sleeve", "left waistline", "right waistline", "left hem", "right hem"]
#               mirror_mapping = [1,0,3,2,5,4,7,6]
#

mirror_mapping = np.array([1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16,19,18,21,20])

class figuratic_image( image ):
    def __init__( self, cfg, image_info, mirrored=False ):
        super().__init__( cfg, image_info, mirrored )

        self.label = image_info.get('label')

        self._data['gtboxes'] = []

        self._data['landmarks_points'] = []
        self._data['landmarks_labels'] = []

        for obj in self._image_info['objects'] :
            self._data['gtboxes'].append( obj.get('bbox') )
            self._data['landmarks_points'].append( obj.get('keypoints') )
            self._data['landmarks_labels'].append( obj.get('keypoint_labels') )

        self._data['gtboxes'] = np.array( self._data['gtboxes'], dtype=np.float32 )
        self._data['landmarks_points'] = np.array( self._data['landmarks_points'], dtype=np.float32 )
        self._data['landmarks_labels'] = np.array( self._data['landmarks_labels'], dtype=np.float32 )

    @property
    def landmarks_points( self ):
        if 'landmarks_points' in self._data :
            keypoints = copy.deepcopy( self._data['landmarks_points'] )
            labels = copy.deepcopy( self._data['landmarks_labels'] )
            if self.is_mirrored :
                width = self._imshape[1]
                keypoints[:,:,0] = width - keypoints[:,:,0]
                keypoints = keypoints[:,mirror_mapping,:]
                labels = labels[:,mirror_mapping]

                for l,k in zip(labels, keypoints):
                    inds = np.where(l==0)
                    k[ inds,: ] = 0
                    

            keypoints = keypoints * self.scale + self.padding
            return keypoints
        else :
            return np.array([])

    @property
    def landmarks_labels( self ):
        if 'landmarks_labels' in self._data :
            labels = copy.deepcopy( self._data['landmarks_labels'] )

            if self.is_mirrored :
                labels = labels[:,mirror_mapping]

            return labels
        else :
            return np.array([])
