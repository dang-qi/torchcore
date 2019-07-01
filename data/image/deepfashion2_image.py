import numpy as np
import copy
from pprint import pprint

from .image import image
from tools import annot_tools

# TODO
# For landmarks this is just from deepfashion1, not deepfashion2
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
#mirror_mapping[1] = np.array([1,0,3,2,5,4])
#mirror_mapping[2] = np.array([1,0,3,2])
#mirror_mapping[3] = np.array([1,0,3,2,5,4,7,6])

class deepfashion2_image( image ):
    def __init__( self, cfg, image_info, mirrored=False ):
        super().__init__( cfg, image_info, mirrored )

        self.source = image_info['source']
        self.pair_id = image_info['pair_id']
        self.width = image_info['width']
        self.height = image_info['height']
        self._imshape = [self.height, self.width]

        self._data['gtboxes'] = []
        self._data['style'] = []
        self._data['category'] = []
        self._data['gtlabels'] = []

        self._data['landmarks_points'] = []
        self._data['landmarks_visibility'] = []
        self._data['landmarks_type'] = []
        #TODO
        self._data['landmarks_labels'] = [] # for mirror mapping, not available for now

        for obj in self._image_info['objects'] :
            self._data['gtboxes'].append( obj.get('bounding_box') )
            self._data['gtlabels'].append( obj['category_id'] )
            self._data['style'].append( obj.get('style') )

            landmarks = obj.get('landmarks')
            if landmarks is not None :
                ll = np.array( landmarks, dtype=np.float32 ).reshape([-1,3])
                self._data['landmarks_points'].append( ll[:,:2] )
                self._data['landmarks_visibility'].append( ll[:,2] ) # labels for visibility: v=2 visible; v=1 occlusion; v=0 not labeled
                self._data['landmarks_type'].append( obj['category_id'] )

        self._data['gtboxes'] = np.array( self._data['gtboxes'], dtype=np.float32 )
        self._data['gtlabels'] = np.array( self._data['gtlabels'], dtype=np.float32 )
        self._data['gtlabels'] = self._data['gtlabels'].reshape([-1,1])
        self._data['style'] = np.array( self._data['style'], dtype=np.float32 )

        #self._data['landmarks_points'] = np.array( self._data['landmarks_points'], dtype=np.float32 )
        #self._data['landmarks_visibility'] = np.array( self._data['landmarks_visibility'], dtype=np.float32 )
        self._data['landmarks_type'] = np.array( self._data['landmarks_type'], dtype=np.float32 )

    @property
    def styles(self):
        return self._data['style']

    @property
    def boxes(self):
        return self._data['gtboxes']

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
    def landmarks_visibility( self ):
        if 'landmarks_visibility' in self._data :
            return copy.deepcopy( self._data['landmarks_visibility'] )
        else :
            return np.array([])

    def show_info( self ):
        pprint( self._image_info )
