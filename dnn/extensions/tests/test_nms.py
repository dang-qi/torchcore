import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import pickle

import torch
import tensorflow as tf

from modules import NMS

def py_nms( dets, scores, thresh ):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

class TestNMS :
    def _load_data( self ):
        pathdir = os.path.dirname( os.path.realpath(__file__))
        fname = os.path.join( pathdir, 'data/nms_test.pkl' )

        with open( fname, 'rb' ) as ff :
            data = pickle.load( ff )[0]

        self._boxes = data['boxes']
        self._scores = data['scores'][:,1].ravel()

    def _py_out( self, thresh ):
        keep = py_nms( self._boxes, self._scores, thresh )
        print( np.sort(keep) )

    def _pytorch_out( self, device, thresh ):
        boxes = torch.from_numpy( self._boxes ).to( device )
        scores = torch.from_numpy( self._scores ).to( device )

        _nms = NMS(thresh)
        keep = _nms( boxes, scores ).cpu().numpy()

        print( np.sort( np.array(keep) ) )

    def _tf_out( self, thresh ):
        inputs = {}
        inputs['boxes'] = tf.placeholder( tf.float32, shape=[None,4] )
        inputs['scores'] = tf.placeholder( tf.float32, shape=[None] )

        keep = tf.image.non_max_suppression( inputs['boxes'], inputs['scores'],
                                            max_output_size=2000, iou_threshold=thresh )

        feed_dict = {}
        feed_dict[ inputs['boxes'] ] = self._boxes[:,np.array([1,0,3,2])]
        feed_dict[ inputs['scores'] ] = self._scores

        with tf.Session() as sess :
            res = sess.run( keep, feed_dict=feed_dict )

        print( res )

    def __init__( self ):
        self._load_data()

    def perform_test( self, device ):
        device = torch.device( device )
        thresh = 0.3
        #self._pytorch_out( device )
        #self._tf_out( thresh=0.3 )
        self._py_out( thresh=0.3 )
        self._pytorch_out( device, thresh=0.3 )
