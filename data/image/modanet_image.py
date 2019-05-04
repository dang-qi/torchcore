import numpy as np
import copy
from .image import image

class modanet_image( image ):
    def __init__( self, cfg, image_info, mirrored=False ):
        super().__init__( cfg, image_info, mirrored )
        self._has_file = image_info['has_file']
        self._label = 1

        self._data['gtboxes'] = []
        self._data['gtlabels'] = []

        for obj in self._image_info['objects'] :
            self._data['gtboxes'].append( obj.get('bbox') )
            self._data['gtlabels'].append( obj.get('category_id') )

        self._data['gtboxes'] = np.array( self._data['gtboxes'], dtype=np.float32 )
        self._data['gtlabels'] = np.array( self._data['gtlabels'], dtype=np.float32 )

    @property
    def has_file( self ):
        return self._has_file
