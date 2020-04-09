import numpy as np
import copy
from pprint import pprint
from ...tools import annot_tools

from PIL import Image
from PIL import ImageOps

from .patch import patch

keypoints_mirror_mapping = np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15])

class coco_person_patch( patch ):
    def __init__( self, cfg, patch_info, mirrored=False ):
        super().__init__( cfg, patch_info, mirrored )
        assert len(patch_info['objects']) == 1

        self._imshape = patch_info['im_shape']
        #self._imname = patch_info['im_name']

        obj = patch_info['objects'][0]

        self._roi = np.round(obj['roi'])
        #self._original_scale = patch_info['scale']
        self._scale = 1.0

        #keypoints_data = np.array(obj['keypoints']).reshape([-1,3])

        self._data['keypoints'] = obj['keypoints']
        self._data['keypoint_labels'] = obj['keypoint_labels'].reshape([-1,1])

    @property
    def shape(self):
        roi = self._roi * self._scale
        w = int(np.round(roi[2] - roi[0]))
        h = int(np.round(roi[3] - roi[1]))
        return [h,w]

    @property
    def roi( self ):
        roi = copy.deepcopy( self._roi )

        roi = np.round( roi )

        if self.is_mirrored :
            width = self._imshape[1]
            x0 = roi[0]
            y0 = roi[1]
            x1 = roi[2]
            y1 = roi[3]
            roi = np.array( [ width-x1, y0, width-x0, y1 ] )

        roi = roi * self.scale
        return roi

    @property
    def patch( self ):
        scale = self.scale
        img = Image.open( self._path )
        roi = np.round(self._roi)
        crop = img.crop( roi )

        if self._mirrored :
            crop = ImageOps.mirror( crop )
            width = self._imshape[1]
            x0 = roi[0]
            y0 = roi[1]
            x1 = roi[2]
            y1 = roi[3]
            roi = np.array( [ width-x1, y0, width-x0, y1 ] )

        if scale != 1.0 :
            h,w = self.shape
            crop = crop.resize( [w,h], Image.BILINEAR )

        np_crop = np.array( crop )
        del crop
        if len( np_crop.shape ) == 2 :
            np_crop = np.repeat(np_crop[:, :, np.newaxis], 3, axis=2)

        return np_crop

    @property
    def patch_PIL( self ):
        scale = self.scale
        img = Image.open( self._path )
        roi = self._roi
        crop = img.crop( roi )

        if self._mirrored :
            crop = ImageOps.mirror( crop )

        if scale != 1.0 :
            h,w = self.shape
            crop = crop.resize( [w,h], Image.BILINEAR )

        return crop

    @property
    def im( self ):
        scale = self.scale
        img = Image.open( self._path )

        if self._mirrored :
            img = ImageOps.mirror( img )

        if scale != 1.0 :
            w = int(np.round( img.size[0]*scale ))
            h = int(np.round( img.size[1]*scale ))
            img = img.resize( [w,h], Image.BILINEAR )

        np_img = np.array( img )
        del img
        if len( np_img.shape ) == 2 :
            np_img = np.repeat(np_img[:, :, np.newaxis], 3, axis=2)

        return np_img

    @property
    def im_PIL( self ):
        scale = self.scale
        img = Image.open( self._path )

        if self._mirrored :
            img = ImageOps.mirror( img )

        if scale != 1.0 :
            w = int(np.round( img.size[0]*scale ))
            h = int(np.round( img.size[1]*scale ))
            img = img.resize( [w,h], Image.BILINEAR )

        return img

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
            keypoints = copy.deepcopy( self._data['keypoints'] ).astype(np.float32)
            roi = np.round(self._roi)

            x0 = roi[0]
            y0 = roi[1]
            x1 = roi[2]
            y1 = roi[3]

            if self.is_mirrored :
                width = self._imshape[1]
                keypoints[:,0] = self._imshape[1] - keypoints[:,0]
                x0 = width - x1
                #annot_tools.mirror_keypoints( keypoints, self._imshape )

            keypoints[:,0] -= x0
            keypoints[:,1] -= y0

            if self.is_mirrored :
                keypoints = keypoints[keypoints_mirror_mapping,:]

            keypoints *= ( self._scale)

            labels = self.keypoint_labels.ravel()
            inds = np.where( labels==0 )[0]
            keypoints[inds,:] = 0

            return keypoints
        else :
            return None

    @property
    def keypoints_image( self ):
        if 'keypoints' in self._data :
            keypoints = copy.deepcopy( self._data['keypoints'] ).astype(np.float32)

            if self.is_mirrored :
                width = self._imshape[1]
                keypoints[:,0] = self._imshape[1] - keypoints[:,0]
                #annot_tools.mirror_keypoints( keypoints, self._imshape )

            if self.is_mirrored :
                keypoints = keypoints[keypoints_mirror_mapping,:]

            keypoints *= (self._scale)

            labels = self.keypoint_labels.ravel()
            inds = np.where( labels==0 )[0]
            keypoints[inds,:] = 0

            return keypoints
        else :
            return None
