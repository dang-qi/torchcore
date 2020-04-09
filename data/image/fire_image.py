import numpy as np
import copy
from pprint import pprint

from .image import image
from ...tools import annot_tools

class fire_image( image ):
    def __init__( self, cfg, image_info, mirrored=False ):
        super().__init__( cfg, image_info, mirrored )
        self.label = image_info.get('label')
