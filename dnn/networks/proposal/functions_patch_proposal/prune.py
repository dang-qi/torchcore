import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def batch_cls_selection( roibatches, roilabels, batch, cls ):
    roibatches = roibatches.ravel()
    roilabels = roilabels.ravel()
    keep = np.where( (roibatches==batch) & (roilabels==cls) )[0]
    keep = keep.reshape([-1,1]).astype( np.int32 )
    return keep

class PruneFunction( Function ) :
    @staticmethod
    def forward( self, labels, batch_size, fgratio ):
        device = labels.device
        labels = labels.detach().cpu().numpy()

        labels = labels.transpose([0,2,3,1])

        n,h,w,c = labels.shape
        labels = labels.ravel()

        pos_inds = np.where( labels == 1 )[0]
        neg_inds = np.where( labels == 0 )[0]

        pos_count = int(fgratio * batch_size)

        pos_selection = []
        if len(pos_inds) > 0 :
            pos_count = int(np.min([pos_count, len(pos_inds)]))
            pos_selection = np.random.choice(pos_inds, pos_count, replace=False)

        neg_count = batch_size - len(pos_selection)

        neg_selection = []
        if len(neg_inds) > 0 :
            neg_count = int(np.min([neg_count, len(neg_inds)]))
            neg_selection = np.random.choice(neg_inds, neg_count, replace=False)

        selection = np.concatenate([ pos_selection, neg_selection ]).astype(int)

        olabels = np.empty( labels.shape )
        olabels.fill(-1)

        olabels[selection] = labels[selection]
        valid = np.where( olabels==1 )[0]

        olabels = olabels.reshape([-1,1])
        mask = np.zeros( (len(labels),4), dtype=np.float32 )
        mask[valid] = 1


        olabels = olabels.reshape([n,h,w,c]).transpose([0,3,1,2])
        mask = mask.reshape([n,h,w,c*4]).transpose([0,3,1,2])

        olabels = torch.from_numpy( olabels.astype(np.int64) ).to( device )
        mask = torch.from_numpy( mask ).to( device )

        return olabels, mask

    @staticmethod
    def backward( self, outgrad ):
        return None, None, None
