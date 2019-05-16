import torch
import numpy as np
from torch.nn import Module
from torch.autograd import Function
import nms_cpu

def py_cpu_nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    #scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    # sort the detections according to scores, high to low values.

    keep = []

    # keep doing until all bboxes are covered by THIS BBOX
    while order.size > 0:
        # choose the bbox with highest iou score, namely THIS BBOX
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

        # filter out all detections which have IOUs larger than the threshold, which means they are covered by THIS BBOX
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)

class NmsFunction( Function ):
    @staticmethod
    def forward( self, rois, scores, threshold, max_size ):
        device = rois.device

        rois = rois.cpu()#.detach().numpy()
        scores = scores.cpu()#.detach().numpy()
        #order = torch.argsort( scores, True )

        nms_keep = nms_cpu.forward_cpu( rois, scores, threshold )
        #nms_keep = py_cpu_nms( rois, scores, threshold )
        nms_keep = nms_keep[:max_size]#.astype( np.int64 )
        #nms_keep = torch.from_numpy( nms_keep ).to( device )

        return nms_keep.to( device )

    @staticmethod
    def backward( self, outgrad ):
        return None, None, None, None

class Nms( Module ):
    def __init__( self, threshold, max_size ):
        super().__init__()
        self._threshold = threshold
        self._max_size = max_size

    def forward( self, rois, scores ):
        return NmsFunction.apply(rois, scores, self._threshold, self._max_size)
