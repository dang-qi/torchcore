import numpy as np
#import config
from pprint import pprint
from .datasets import selector as dataset_selector

class workbench :
    def __init__( self, cfg, dset_params, model_params=None ):
        self._cfg = cfg
        self._dataset = dataset_selector[ dset_params[0] ]( self._cfg.dataset, *( dset_params[1:] ) )
        if not model_params is None :
            self._model = dataset_selector[ model_params[0] ]( self._cfg.dataset, *( model_params[1:] ) )
        else :
            self._model = self._dataset

    def set_as_main( self ):
        config.set_dnn_cfg( self._cfg )

    def load_dataset( self, workset_setting=None ):
        if workset_setting is None :
            workset_setting = self._cfg.dataset.WORKSET_SETTING
        self._dataset._images = None
        self._dataset.load( workset_setting )

    def load_rois( self, path, roi_overlaps=False ):
        self._dataset.load_rois( path, roi_overlaps )

    def load_deploy_dataset( self, workset_setting=None ):
        if workset_setting is None :
            workset_setting = self._cfg.dataset.WORKSET_SETTING_DEPLOY
        self._dataset.load( workset_setting )

    @property
    def dataset( self ):
        return self._dataset

    @property
    def model( self ):
        return self._model
