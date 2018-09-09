from multiprocessing import Queue
import threading
import time

class dataset_buffer(threading.Thread):
    def __init__( self, imageset, queue, buffer_size ):
        threading.Thread.__init__(self)
        self._imageset = imageset
        self._queue = queue
        self._buffer_size = buffer_size

        self.shutdown_flag = threading.Event()

    def run( self ):
        while not self.shutdown_flag.is_set() :
            if not self._queue.full() :
                self._queue.put( self._imageset.next() )
            time.sleep(0.01)

class data_feeder :
    def __init__( self, imageset, buffer_size=50 ):
        self._imageset = imageset
        self._buffer_size = buffer_size
        self._queue = Queue( maxsize=buffer_size )

        print('Data feeder is buffering ...')

        while not self._queue.full():
            self._queue.put( self._imageset.next() )

        self._buffer_thread = dataset_buffer( self._imageset, self._queue, self._buffer_size )
        self._buffer_thread.start()

    def __del__( self ):
        self._buffer_thread.shutdown_flag.set()

    def __len__( self ):
        return len(self._imageset)

    def next( self ):
        blobs = self._queue.get()
        return blobs
