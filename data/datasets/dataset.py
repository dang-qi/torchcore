import numpy as np
import pickle
import copy
from pprint import pprint
from ..image import image

class dataset :
    def __init__( self, cfg, img_type=image ):
        self._cfg = cfg
        self._image = img_type
        self._workset = None
        self._images = None

    def load( self, setting=None ):
        if setting is None :
            setting = self._cfg.WORKSET_SETTING

        self._workset_setting = setting
        dset_path = self._cfg.ANNOTS_TMP % ( self._dset_tag )
        with open( dset_path, 'rb' ) as ff :
            dset_data = pickle.load( ff )[0]

        if type(dset_data) is dict :
            dset_data = dset_data[ self._part ]

        add_mirrored = False
        if 'add_mirrored' in setting :
            add_mirrored = True

        self._original_images = []
        for data in dset_data :
            img = self._image( self._cfg, data, mirrored=False )

            self._original_images.append( img )
            if add_mirrored :
                self._original_images.append( img.mirrored() )

        self._original_images = np.array( self._original_images )

    def _randomize_scales( self, images ):
        s0 = self._cfg.SCALES[0]
        s1 = self._cfg.SCALES[1]

        for img in images :
            img.scale = np.random.uniform( s0, s1 )

    def _size_prune( self, images ):
        tmp = []
        min_size = self._cfg.MIN_OBJ_SIZE

        valid_inds = []

        for i,image in enumerate(images) :
            gtboxes = image.gtboxes
            if len( gtboxes ) > 0 :
                widths = gtboxes[:,2] - gtboxes[:,0]
                heights = gtboxes[:,3] - gtboxes[:,1]

                height_check = np.max( heights ) >= min_size[0]
                width_check = np.max( widths ) >= min_size[1]

                if height_check or width_check :
                    valid_inds.append(i)
            else :
                valid_inds.append(i)

        return np.array( valid_inds )

    def create_workset( self ):
        del self._workset
        self._workset = []

        images = copy.deepcopy( self._original_images )

        if 'randomize_scales' in self._workset_setting :
            self._randomize_scales( images )

        if 'prune_by_size' in self._workset_setting :
            inds = self._size_prune( images )
        else :
            inds = np.arange(0,len( images ) )

        self._workset = images[ inds ]
        self._selection = inds

    @property
    def ndata( self ):
        assert self._workset is not None, "workset is not initiated."
        return len( self._workset )

    def __len__( self ):
        if self._workset is None :
            self.create_workset()
        return len( self._workset )

    @property
    def images( self ):
        if self._workset is None :
            self.create_workset()
        return self._workset

    @property
    def hash( self ):
        return self._dset_hash
