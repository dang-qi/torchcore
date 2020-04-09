import numpy as np
import copy
from ...tools import annot_tools
from ...tools import bbox_tools

from .modanet_image import modanet_image

class modanetp_image( modanet_image ):
    def __init__( self, cfg, image_info, mirrored=False ):
        super().__init__( cfg, image_info, mirrored )

        detections = image_info['detections']

        self._data['rois'] = detections['rois']
        self._data['roiscores'] = detections['roiscores']
        self._data['roilabels'] = detections['roilabels']

    @property
    def rois( self ):
        if 'rois' in self._data :
            scale = self.scale
            rois = copy.deepcopy( self._data['rois'] )
            if self._mirrored and len(rois) > 0 :
                rois = annot_tools.mirror_boxes( rois, self._imshape )
            rois *= scale
            rois = rois + self.padding

            if len( rois ) > 0 :
                bbo = bbox_tools()
                rois = bbo.clip( rois, self.shape )

            return rois
        else :
            return np.array([])

    @property
    def roilabels( self ):
        if 'roilabels' in self._data :
            return copy.deepcopy( self._data['roilabels'] )
        else :
            return np.array([])

    @property
    def roiscores( self ):
        if 'roiscores' in self._data :
            return copy.deepcopy( self._data['roiscores'] )
        else :
            return np.array([])
