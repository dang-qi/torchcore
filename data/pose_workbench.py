import numpy as np
#import config
from .workbench import workbench

class pose_workbench( workbench ):
    def __init__( self, cfg, params ):
        super().__init__( cfg, params )

    def load_dataset( self, workset_setting=None ):
        super().load_dataset( workset_setting=workset_setting )
        self._dataset.reduce_to_pose()
