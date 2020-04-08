import numpy as np
import copy
from pprint import pprint
from ...tools import annot_tools

from .image import image

keypoints_mirror_mapping = np.array([5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 16, 15, 14, 13, 12, 19, 18, 17, 20])

class aflw_patch_image( image ):
    def __init__( self, cfg, image_info, mirrored=False ):
        super().__init__( cfg, image_info, mirrored )

        self.label = self._image_info.get('label',None)
        gender_dict = {'f':0, 'm':1}

        self._data['gtboxes'] = []
        self._data['pose'] = []
        self._data['gender'] = []
        self._data['glasses'] = []
        self._data['keypoints'] = []
        self._data['keypoint_labels'] = []

        for obj in self._image_info['objects'] :
            self._data['gtboxes'].append( obj['expand_roi'] )
            self._data['pose'].append( obj['pose'] )
            self._data['gender'].append( gender_dict[ obj['sex'] ] )
            self._data['glasses'].append( obj['glasses'] )
            self._data['keypoints'].append( obj['keypoints'] )
            self._data['keypoint_labels'].append( obj['keypoint_labels'] )

        self._data['gtboxes'] = np.array( self._data['gtboxes'], dtype=np.float32 )
        self._data['pose'] = np.array( self._data['pose'], dtype=np.float32 )
        self._data['keypoints'] = np.array( self._data['keypoints'], dtype=np.float32 )
        self._data['keypoint_labels'] = np.array( self._data['keypoint_labels'], dtype=np.float32 )

        self._data['gender'] = np.array( self._data['gender'], dtype=np.float32 )
        self._data['gender'] = self._data['gender'].reshape((-1,1))

        self._data['glasses'] = np.array( self._data['glasses'], dtype=np.float32 )
        self._data['glasses'] = self._data['glasses'].reshape((-1,1))

    @property
    def pose( self ):
        if 'pose' in self._data :
            pose = copy.deepcopy( self._data['pose'] )
            pose = pose.reshape([-1,3])

            #if self.is_mirrored :
            #    pose[:,2] *= -1
            return pose
        else :
            return None

    @property
    def gender( self ):
        if 'gender' in self._data :
            gender = copy.deepcopy( self._data['gender'] )
            return gender
        else :
            return None

    @property
    def glasses( self ):
        if 'glasses' in self._data :
            gender = copy.deepcopy( self._data['glasses'] )
            return gender
        else :
            return None

    @property
    def keypoints( self ):
        if 'keypoints' in self._data :
            keypoints = copy.deepcopy( self._data['keypoints'] )
            labels = copy.deepcopy( self._data['keypoint_labels'] )
            if self.is_mirrored :
                annot_tools.mirror_keypoints( keypoints, self._imshape )
            keypoints = keypoints * self.scale + self.padding

            shape = keypoints.shape

            keypoints = keypoints.reshape([-1,2])
            labels = labels.reshape([-1,1]).ravel()

            inds = np.where( labels==0 )[0]
            keypoints[ inds,: ] = 0
            keypoints = keypoints.reshape( shape )

            if self.is_mirrored :
                keypoints = keypoints[:,keypoints_mirror_mapping,:]

            return keypoints
        else :
            return None

    @property
    def keypoint_labels( self ):
        if 'keypoint_labels' in self._data :
            keypoint_labels = copy.deepcopy( self._data['keypoint_labels'] )

            if self.is_mirrored :
                keypoint_labels = keypoint_labels[:,keypoints_mirror_mapping,:]

            return keypoint_labels
        else :
            return None
