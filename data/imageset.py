import numpy as np

class imageset :
    def _prepare( self ):
        raise NotImplementedError

    def _size_prune( self, size ):
        raise NotImplementedError

    def __init__( self, cfg, dataset, blobs_gen, randomize=False, is_training=True, size_prune=None ):
        self._cfg = cfg
        self._images = dataset.images
        self._blobs_gen = blobs_gen
        self._batchsize = self._cfg.dnn.NETWORK.BATCH_SIZE
        self._randomize = randomize
        self._is_training = is_training
        self._cur = -1

        if size_prune is not None :
            self._size_prune( size_prune )

        self._prepare()

    def __len__( self ):
        raise NotImplementedError

    def __getitem__( self, idx ):
        raise NotImplementedError

    def next( self ):
        raise NotImplementedError
