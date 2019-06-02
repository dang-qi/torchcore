import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def _clip_gtboxes( nbatches, rois, roibatches, gtboxes, gtbatches, normalize=True ):
    ogtboxes = []

    gtbatches = gtbatches.ravel()
    roibatches = roibatches.ravel()

    for idx, b in enumerate( gtboxes ):
        batch_idx = gtbatches[idx]
        roi_idx = np.where( roibatches == batch_idx )[0][0]

        r = rois[ roi_idx ]

        gx0, gy0, gx1, gy1 = b
        rx0, ry0, rx1, ry1 = r

        x0 = np.max([gx0,rx0])
        y0 = np.max([gy0,ry0])
        x1 = np.min([gx1,rx1])
        y1 = np.min([gy1,ry1])

        if normalize :
            x0 -= rx0
            y0 -= ry0
            x1 -= rx0
            y1 -= ry0

        ogtboxes.append([x0,y0,x1,y1])

    return np.array(ogtboxes, dtype=np.float32)

def _intersection( b0, b1 ):
    ix = np.min( [ b0[2], b1[2] ] ) - np.max( [ b0[0], b1[0] ] )
    iy = np.min( [b0[3], b1[3] ] ) - np.max( [ b0[1], b1[1] ] )
    return np.max( [ix,0.0] ) * np.max( [iy,0.0] )

def _build_gt_map(h,w,r,b):
    x0,y0,x1,y1 = b
    roi_w = r[2] - r[0]
    roi_h = r[3] - r[1]

    yarange = np.linspace(0,roi_h,h+1)
    xarange = np.linspace(0,roi_w,w+1)

    tmp_x = np.zeros([h,w])
    tmp_y = np.zeros([h,w])

    xgrid = (xarange[1] - xarange[0])
    ygrid = (yarange[1] - yarange[0])

    x0 = np.max([ x0 - xgrid/2, xarange[0]] )
    y0 = np.max([ y0 - ygrid/2, yarange[0]] )

    if x1 - x0 < xgrid :
        x1 = np.min( [ x0 + xgrid + xgrid/2, xarange[-1] ] )
    if y1 - y0 < ygrid :
        y1 = np.min( [ y0 + ygrid + ygrid/2, yarange[-1] ] )

    xinds = np.where( (xarange>=x0) & (xarange<x1) )[0]
    yinds = np.where( (yarange>=y0) & (yarange<y1) )[0]

    tmp_x[:,xinds] = 1
    tmp_y[yinds,:] = 1

    tmp_x = tmp_x.ravel()
    tmp_y = tmp_y.ravel()

    valid = np.where( (tmp_x==1) & (tmp_y==1) )[0]

    gt_map = np.zeros([h*w])
    gt_map[valid] = 1

    return gt_map, valid, xarange, yarange

def _build_labels( feat_shape, gtrois, gtroibatches, gtboxes_norm, gtboxes, gtlabels, gtbatches, nclasses ):
    n,h,w,d = feat_shape
    gtbatches = gtbatches.ravel()
    gtroibatches = gtroibatches.ravel()
    labels = np.zeros([n,h,w,nclasses], dtype=np.float32)
    targets = np.zeros([n,h,w,4*nclasses], dtype=np.float32)

    for idx, b in enumerate( gtboxes_norm ):
        batch_idx = gtbatches[idx]
        roi_idx = np.where( gtroibatches == batch_idx )[0][0]
        r = gtrois[ roi_idx ]

        x0,y0,x1,y1 = b
        gt_map, valid, xarange, yarange = _build_gt_map(h,w,r,b)
        depth = int(gtlabels[idx]) - 1

        labels_map = labels[batch_idx,:,:,depth].ravel()
        labels_map[valid] = 1

        labels_map = labels_map.reshape([h,w])
        labels[batch_idx,:,:,depth] = labels_map

        yhalf = (yarange[1] - yarange[0])/2
        xhalf = (xarange[1] - xarange[0])/2

        xx, yy = np.meshgrid( xarange[:-1] + xhalf, yarange[:-1] + yhalf )
        g = np.stack([xx,yy], axis=2).reshape( -1,2 )

        invalid = np.where( gt_map == 0 )[0]
        valid = np.where( gt_map==1 )[0]

        bxc = (x0+x1)/2
        byc = (y0+y1)/2
        bw = (x1-x0)
        bh = (y1-y0)

        indices = np.arange(depth*4,(depth+1)*4)
        t = []
        for idx in indices :
            t.append( targets[batch_idx,:,:,idx].ravel() )

        t[0][valid] = (bxc - g[valid,0])/bw
        t[1][valid] = (byc - g[valid,1])/bh
        t[2][valid] = np.log(bw)
        t[3][valid] = np.log(bh)

        for ii,jj in enumerate(indices) :
            targets[batch_idx,:,:,jj] = t[ii].reshape((h,w))

    return labels, targets

def _build_exclusive_batches( scores_target, batch_labels, nclasses ):
    n,h,w,d = scores_target.shape
    batch_labels = batch_labels.ravel()

    for ii in range(n) :
        b = batch_labels[ii] - 1
        for jj in range(nclasses) :
            if jj != b :
                scores_target[ii,:,:,jj] = -1

    return scores_target

class BuildTargetsFunction( Function ):
    @staticmethod
    def forward( self, scores, gtrois, gtroibatches, gtboxes, gtlabels, gtbatches,
                 nclasses, batch_labels, exclusive_batches ):

        nbatches,channels,h,w = scores.shape
        device = scores.device

        gtrois = gtrois.detach().cpu().numpy()
        gtroibatches = gtroibatches.detach().cpu().numpy()
        gtboxes = gtboxes.detach().cpu().numpy()
        gtlabels = gtlabels.detach().cpu().numpy()
        gtbatches = gtbatches.detach().cpu().numpy()
        batch_labels = batch_labels.detach().cpu().numpy()

        gtboxes_norm = _clip_gtboxes( nbatches, gtrois, gtroibatches, gtboxes, gtbatches )
        feat_shape = [nbatches, h, w, channels]
        scores_target, rois_target = _build_labels( feat_shape, gtrois, gtroibatches, gtboxes_norm, gtboxes, gtlabels, gtbatches, nclasses )

        if exclusive_batches :
            scores_target = _build_exclusive_batches( scores_target, batch_labels, nclasses )

        scores_target = scores_target.transpose([0,3,1,2])
        rois_target = rois_target.transpose([0,3,1,2])

        scores_target = torch.from_numpy( scores_target ).to( device )
        rois_target = torch.from_numpy( rois_target ).to( device )

        return scores_target, rois_target

    @staticmethod
    def backward( self, outgrad ):
        return None, None, None, None, None, None, None, None, None
