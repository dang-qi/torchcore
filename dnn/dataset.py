import numpy as np

class dataset :
    def __init__( self, imageset, blobs_gen, batch_size, randomize=False, training=False, benchmark=None ):
        self._imageset = imageset
        self._blobs_gen = blobs_gen
        self._imageset = imageset
        self._training = training

        ndata = len(self._imageset)
        inds = np.arange(ndata)

        if randomize :
            np.random.shuffle(inds)

        self._chunks = [ inds[i:i+batch_size] for i in np.arange(0,ndata,batch_size) ]
        self._benchmark = benchmark

        self._cur = 0

    @property
    def benchmark( self ):
        return self._benchmark

    def __len__( self ):
        return len(self._chunks)

    def __getitem__( self, index ):
        images = self._imageset[self._chunks[index]]
        inputs, targets = self._blobs_gen.get_blobs(images, training=self._training)
        return inputs, targets

    def next( self ):
        if self._cur >= len( self._chunks ) :
            self._cur = 0
        idx = self._cur
        self._cur = self._cur + 1
        return self[ idx ]
