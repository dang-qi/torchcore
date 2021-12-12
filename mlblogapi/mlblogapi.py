import requests
import json
import threading
import datetime

import requests
import json
import threading
import datetime
import copy

class MLBlogAPI :
    def __init__( self, reference, server='http://trooper.noip.me:56433', call_interval=5 ):
        self.reference = reference
        self.server = server
        self.call_interval = call_interval
        self._initialized = False
        self._init_time = None
        self._nepoch = None
        self._epoch_size = None
        self._total = None

        self._buffer = []
        self._last_call = None

    def _initialize( self, nepoch, epoch_size ):
        self._nepoch = nepoch
        self._epoch_size = epoch_size
        self._total = nepoch * epoch_size

        headers = {}

        data = {}
        data['reference'] = self.reference
        data['nepoch'] = nepoch
        data['epoch_size'] = epoch_size

        response = requests.post( '%s/api/init/' % ( self.server ), data=json.dumps(data), headers=headers )

        if response.status_code == 200 :
            self._initialized = True
            self._init_time = datetime.datetime.now()
            self._last_call = datetime.datetime.now()

    def _update( self, buffer_copy ):
        if not self._initialized :
            return

        headers = {}

        data = {}
        data['reference'] = self.reference
        data['buffer'] = buffer_copy

        response = requests.post( '%s/api/updatebatch/' % ( self.server ), data=json.dumps(data), headers=headers )

    def setup( self, nepoch, epoch_size ):
        t = threading.Thread(target=self._initialize(int(nepoch), int(epoch_size)))
        t.start()
        #t.join()

    def update( self, epoch_idx, iter_idx, loss_value ):
        ctime = datetime.datetime.now()

        percentage = ( epoch_idx * self._epoch_size + iter_idx + 1) / self._total

        data = {}
        data['epoch_idx'] = int(epoch_idx)
        data['iter_idx'] = int(iter_idx)
        data['loss_value'] = float(loss_value)
        data['time_elapsed'] = ( ctime - self._init_time ).total_seconds()
        data['percentage'] = percentage

        self._buffer.append( data )

        if (ctime - self._last_call).total_seconds() > self.call_interval :
            try :
                buffer_copy = copy.deepcopy( self._buffer )
                self._update( buffer_copy )

                self._last_call = ctime
                self._buffer = []
            except :
                pass

        #t = threading.Thread(target=self._update(int(epoch_idx), int(iter_idx), float(loss_value)))
        #t.start()

    def flush( self ):
        try :
            buffer_copy = copy.deepcopy( self._buffer )
            self._update( buffer_copy )
            self._last_call = datetime.datetime.now()
            self._buffer = []
        except:
            pass