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

mirror_mapping = {}
mirror_mapping[1] = np.array([1,0,3,2,5,4])
mirror_mapping[2] = np.array([1,0,3,2])
mirror_mapping[3] = np.array([1,0,3,2,5,4,7,6])

class deepfashion_image( image ):
    def __init__( self, cfg, image_info, mirrored=False ):
        super().__init__( cfg, image_info, mirrored )

        self.label = image_info.get('label')

        self._data['gtboxes'] = []
        self._data['gtlabels'] = []
        self._data['category'] = []

        self._data['landmarks_points'] = []
        self._data['landmarks_labels'] = []
        self._data['landmarks_type'] = []
        self._data['landmarks_variation'] = []

        for obj in self._image_info['objects'] :
            self._data['gtboxes'].append( obj.get('bbox') )
            self._data['gtlabels'].append( image_info.get('label') )
            self._data['category'].append( obj.get('category') )

            landmarks = obj.get('landmarks')
            if landmarks is not None :
                ll = np.array( landmarks['landmarks'], dtype=np.float32 ).reshape([-1,3])
                self._data['landmarks_points'].append( ll[:,1:] )
                self._data['landmarks_labels'].append( ll[:,0] )
                self._data['landmarks_type'].append( landmarks['type'] )
                self._data['landmarks_variation'].append( landmarks['variation'] )

        self._data['gtboxes'] = np.array( self._data['gtboxes'], dtype=np.float32 )
        self._data['category'] = np.array( self._data['category'], dtype=np.float32 )
        self._data['category'] = self._data['category'].reshape([-1,1])

        self._data['landmarks_points'] = np.array( self._data['landmarks_points'], dtype=np.float32 )
        self._data['landmarks_labels'] = np.array( self._data['landmarks_labels'], dtype=np.float32 )
        self._data['landmarks_type'] = np.array( self._data['landmarks_type'], dtype=np.float32 )
        self._data['landmarks_variation'] = np.array( self._data['landmarks_variation'], dtype=np.float32 )

    @property
    def category( self ):
        if 'category' in self._data :
            return copy.deepcopy( self._data['category'] )
        else :
            return np.array([])

    @property
    def category( self ):
        if 'category' in self._data :
            return copy.deepcopy( self._data['category'] )
        else :
            return np.array([])

    @property
    def landmarks_points( self ):
        if 'landmarks_points' in self._data :
            keypoints = copy.deepcopy( self._data['landmarks_points'] )
            types = copy.deepcopy( self._data['landmarks_type'] )
            if self.is_mirrored :
                width = self._imshape[1]
                keypoints[:,:,0] = width - keypoints[:,:,0]

                assert( len(types) == len(keypoints) )

                for i in range(len(types)):
                    t = int(types[i])
                    k = keypoints[i][ mirror_mapping[t] ]
                    keypoints[i] = k

            keypoints = keypoints * self.scale + self.padding
            return keypoints
        else :
            return np.array([])

    @property
    def landmarks_labels( self ):
        if 'landmarks_labels' in self._data :
            labels = copy.deepcopy( self._data['landmarks_labels'] )
            types = copy.deepcopy( self._data['landmarks_type'] )

            if self.is_mirrored :
                assert( len(types) == len(labels) )
                for i in range(len(types)) :
                    t = int(types[i])
                    labels[i] = labels[i][ mirror_mapping[t] ]

            return labels
        else :
            return np.array([])

    @property
    def landmarks_type( self ):
        if 'landmarks_type' in self._data :
            return copy.deepcopy( self._data['landmarks_type'] )
        else :
            return np.array([])

    @property
    def landmarks_variation( self ):
        if 'landmarks_variation' in self._data :
            return copy.deepcopy( self._data['landmarks_variation'] )
        else :
            return np.array([])

    def show_info( self ):
        pprint( self._image_info )

    def show_info( self ):
        pprint( self._image_info )
