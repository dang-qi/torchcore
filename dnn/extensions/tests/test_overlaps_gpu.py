import _init
import torch
import numpy as np
import sys
import overlaps_gpu
import pickle
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def intersection_area( b0, b1 ):
        ix = np.min( [ b0[2], b1[2] ] ) - np.max( [ b0[0], b1[0] ] )
        iy = np.min( [b0[3], b1[3] ] ) - np.max( [ b0[1], b1[1] ] )
        return np.max( [ix+1,0.0] ) * np.max( [iy+1,0.0] )

def area( b ):
    return ( b[2] - b[0] + 1 ) * ( b[3] - b[1] + 1 )

def py_overlap( rects0, rects1 ):
        out = np.zeros( (len(rects0), len(rects1) ), dtype=np.float32 )

        for i,r0 in enumerate( rects0 ):
            for j,r1 in enumerate( rects1 ):
                ii = intersection_area( r0, r1 )
                a0 = area( r0 )
                a1 = area( r1 )

                out[i,j] = ii/( a0+a1-ii+1e-10 )
        return out

if __name__=="__main__" :
    with open('test_data.pkl','rb') as ff :
        d = pickle.load(ff)[0]

    if torch.cuda.is_available() :
        device = torch.device("cuda")
        gtboxes = d['gtboxes']
        gtlabels = np.ones( len(gtboxes), dtype=np.float32 )
        gtbatches = np.ones( len(gtboxes), dtype=np.int64 )

        rois = d['rois']
        roilabels = np.ones( len(rois), dtype=np.float32 )
        roibatches = np.ones( len(rois), dtype=np.int64 )

        pt_gtboxes = torch.from_numpy( gtboxes ).to( device )
        pt_gtlabels = torch.from_numpy( gtlabels ).to( device )
        pt_gtbatches = torch.from_numpy( gtbatches ).to( device )

        pt_rois = torch.from_numpy( rois ).to( device )
        pt_roilabels = torch.from_numpy( roilabels ).to( device )
        pt_roibatches = torch.from_numpy( roibatches ).to( device )

        out = torch.zeros( (len(gtboxes), len(rois)), dtype=torch.float32 )
        out = out.to( device )

        overlaps_gpu.forward( pt_gtboxes, pt_gtlabels, pt_gtbatches,
                              pt_rois, pt_roilabels, pt_roibatches,
                              out )

        print( out.cpu().numpy() )
        print( py_overlap( gtboxes, rois ) )
