import numpy as np
import copy
from pprint import pprint

from .image import image

class aflw_image( image ):
    def __init__( self, cfg, image_info, mirrored=False ):
        super().__init__( cfg, image_info, mirrored )

        self.label = self._image_info.get('label',None)
        gender_dict = {'f':0, 'm':1}

        self._data['gtboxes'] = []
        self._data['pose'] = []
        self._data['gender'] = []
        self._data['glasses'] = []

        for obj in self._image_info['objects'] :
            self._data['gtboxes'].append( obj['bbox'] )
            self._data['pose'].append( obj['pose'] )
            self._data['gender'].append( gender_dict[ obj['sex'] ] )
            self._data['glasses'].append( obj['glasses'] )

        self._data['gtboxes'] = np.array( self._data['gtboxes'], dtype=np.float32 )
        self._data['pose'] = np.array( self._data['pose'], dtype=np.float32 )

        self._data['gender'] = np.array( self._data['gender'], dtype=np.float32 )
        self._data['gender'] = self._data['gender'].reshape((-1,1))

        self._data['glasses'] = np.array( self._data['glasses'], dtype=np.float32 )
        self._data['glasses'] = self._data['glasses'].reshape((-1,1))

    @property
    def pose( self ):
        if 'pose' in self._data :
            pose = copy.deepcopy( self._data['pose'] )
            return pose
        else :
            return np.array([])

    @property
    def gender( self ):
        if 'gender' in self._data :
            gender = copy.deepcopy( self._data['gender'] )
            return gender
        else :
            return np.array([])

    @property
    def glasses( self ):
        if 'glasses' in self._data :
            gender = copy.deepcopy( self._data['glasses'] )
            return gender
        else :
            return np.array([])
