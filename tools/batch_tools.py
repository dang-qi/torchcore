import contextlib
import numpy as np

@contextlib.contextmanager
def temp_seed( seed ):
    state = np.random.get_state()
    np.random.seed( seed )
    try :
        yield
    finally :
        np.random.set_state(state)

class batch_generator :
    def __init__( self, dataset, batch_size ):
        self._images = dataset.images
        self._batch_size = batch_size
        self._chunks = []

    def _build_chunks( self ):
        ndata = len( self._images )
        inds = np.arange( ndata )
        bs = self._batch_size
        self._chunks = [ inds[i:i+bs] for i in np.arange(0,ndata,bs) ]
        self._epoch_size = len( self._chunks )

    def __len__( self ):
        return len( self._chunks )

    def __getitem__( self, index ):
        return self._chunks[ index ]
