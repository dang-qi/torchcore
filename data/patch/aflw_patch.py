import numpy as np
import copy
from pprint import pprint
from tools import annot_tools

from .patch import patch

keypoints_mirror_mapping = np.array([5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 16, 15, 14, 13, 12, 19, 18, 17, 20])
gender_dict = {'f':0, 'm':1}

class aflw_patch( patch ):
    def __init__( self, cfg, patch_info, mirrored=False ):
        super().__init__( cfg, patch_info, mirrored )
        assert len(patch_info['objects']) == 1

        self._imshape = patch_info['im_shape']
        self._imname = patch_info['im_name']

        obj = patch_info['objects'][0]

        self._roi = obj['expand_roi']

        orig_h = self._roi[3] - self._roi[1]
        orig_w = self._roi[2] - self._roi[0]
        patch_h, patch_w = self._patchshape

        assert orig_h == orig_w
        assert patch_h == patch_w

        self._original_scale = float(patch_h) / orig_h
        self._scale = 1.0

        self._data['gender'] = gender_dict[obj['sex']]
        self._data['pose'] = np.array(obj['pose'])
        self._data['glasses'] = obj['glasses']
        self._data['keypoints'] = np.array(obj['keypoints'])
        self._data['keypoint_labels'] = np.array(obj['keypoint_labels'])

    @property
    def gender( self ):
        if 'gender' in self._data :
            gender = copy.deepcopy( self._data['gender'] )
            return gender
        else :
            return None

    @property
    def pose( self ):
        if 'pose' in self._data :
            pose = copy.deepcopy( self._data['pose'] )

            if self.is_mirrored :
                pose[0] *= -1
                pose[2] *= -1

            return pose
        else :
            return None

    @property
    def glasses( self ):
        if 'glasses' in self._data :
            glasses = copy.deepcopy( self._data['glasses'] )
            return glasses
        else :
            return None

    @property
    def keypoint_labels( self ):
        if 'keypoint_labels' in self._data :
            keypoint_labels = copy.deepcopy( self._data['keypoint_labels'] )

            if self.is_mirrored :
                keypoint_labels = keypoint_labels[keypoints_mirror_mapping,:]

            return keypoint_labels
        else :
            return None

    @property
    def keypoints( self ):
        if 'keypoints' in self._data :
            keypoints = copy.deepcopy( self._data['keypoints'] )

            x0 = self._roi[0]
            y0 = self._roi[1]
            x1 = self._roi[2]
            y1 = self._roi[3]

            if self.is_mirrored :
                width = self._imshape[1]
                keypoints[:,0] = self._imshape[1] - keypoints[:,0]
                x0 = width - x1
                #annot_tools.mirror_keypoints( keypoints, self._imshape )

            keypoints[:,0] -= x0
            keypoints[:,1] -= y0

            if self.is_mirrored :
                keypoints = keypoints[keypoints_mirror_mapping,:]

            keypoints *= (self._original_scale * self._scale)

            labels = self.keypoint_labels.ravel()
            inds = np.where( labels==0 )[0]
            keypoints[inds,:] = 0

            return keypoints
        else :
            return None
