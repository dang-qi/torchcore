import numpy as np
#from bbox_overlap_module import bbox_overlap

class bbox_tools:
    def __init__( self ):
        self._bbo = None#bbox_overlap()

    def whc( self, b ):
        w = b[2] - b[0]
        h = b[3] - b[1]
        x = b[0] + w*0.5
        y = b[1] + h*0.5
        return np.array( [w,h,x,y] )

    def whc_inv( self, r ):
        x0 = r[2] - 0.5*r[0]
        y0 = r[3] - 0.5*r[1]
        x1 = x0 + r[0]
        y1 = y0 + r[1]
        return np.array( [x0,y0,x1,y1] )

    def whcs( self, bs ):
        print( bs.shape )

        ws = bs[:,2] - bs[:,0] # Width
        hs = bs[:,3] - bs[:,1] # Height
        xs = bs[:,0] + ws*0.5  # Center x
        ys = bs[:,1] + hs*0.5  # Center y

        return np.vstack( [ ws, hs, xs, ys ] ).transpose()

    def whcs_inv( self, rs ):
        x0 = rs[:,2] - 0.5*rs[:,0]
        y0 = rs[:,3] - 0.5*rs[:,1]
        x1 = x0 + rs[:,0]
        y1 = y0 + rs[:,1]

        return np.vstack( [ x0, y0, x1, y1 ] ).transpose()

    def forward_transform( self,r,t ):
        r_whc = self.whc( r )
        t_whc = self.whc( t )

        dx = ( t_whc[2] - r_whc[2] ) / r_whc[0]
        dy = ( t_whc[3] - r_whc[3] ) / r_whc[1]
        dw = np.log(t_whc[0] / r_whc[0])
        dh = np.log(t_whc[1] / r_whc[1])

        return np.array([ dx,dy,dw,dh ])

    def backward_transform( self,r,d ):
        rw, rh, rx, ry = self.whc( r )
        dx, dy, dw, dh = d

        dw = np.exp(dw)
        dh = np.exp(dh)

        tw = dw * rw
        th = dh * rh
        tx = dx * rw + rx
        ty = dy * rh + ry

        return self.whc_inv( np.array( [tw,th,tx,ty] ) )

    def transform( self, refs, targets ):
        #
        # Refs are the boxes that already exist
        # Targets are the boxes that we wish to predict
        #

        assert refs.shape == targets.shape, "The shape of the references and the targets do not match"

        #for r,t in zip( refs, targets ):
        #    d = forward( r,t )
        #    t2 = backward( r,d )
        #    print( t,t2 )

        delta = []

        for r,t in zip( refs, targets ):
            delta.append( self.forward_transform(r,t) )

        return np.array( delta, dtype=np.float32 )

    def transform_inv( self, refs, deltas ):
        assert refs.shape == deltas.shape, "The shape of the references and the targets do not match"

        refs = refs.astype( deltas.dtype, copy=False )

        targets = []
        for r,d in zip( refs, deltas ):
            targets.append( self.backward_transform(r,d) )
        return np.array( targets, dtype=np.float32)

    def clip( self, boxes, im_shape ):
        # im_shape = [ height, width ]
        def maxmin( values, min_v, max_v ):
            return np.maximum( np.minimum( values, min_v ), max_v )

        boxes[:,0::4] = maxmin( boxes[:,0::4], im_shape[1]-1, 0 )
        boxes[:,1::4] = maxmin( boxes[:,1::4], im_shape[0]-1, 0 )
        boxes[:,2::4] = maxmin( boxes[:,2::4], im_shape[1]-1, 0 )
        boxes[:,3::4] = maxmin( boxes[:,3::4], im_shape[0]-1, 0 )

        return boxes

    def size_filter( self, boxes, min_size ):
        heights = (boxes[:,3] - boxes[:,1]).ravel()
        widths = (boxes[:,2] - boxes[:,0]).ravel()
        return np.where( (heights >= min_size) & ( widths >= min_size ) )[0]


    def overlap( self, rects0, rects1 ):
        overlaps = np.array( self._bbo.overlap( rects0.ravel().tolist(), rects1.ravel().tolist() ), dtype=np.float32 )
        return overlaps

    def overlap_gpu( self, rects0, rects1 ):
        n0 = len( rects0 )
        n1 = len( rects1 )

        overlaps = np.array( self._bbo.overlap_gpu( rects0.ravel().tolist(), rects1.ravel().tolist() ), dtype=np.float32 )
        return overlaps.reshape( n0,n1 )

    def nms( self, rects, scores, thresh, sort=False ):
        if sort :
            sinds = np.argsort( scores )[::-1]
            rects = rects[ sinds ]
            scores = scores[ sinds ]
        # Assuming that boxes are sorted
        keep = self._bbo.nms( rects.ravel().tolist(), scores.ravel().tolist(), thresh )
        return np.array( keep )

    def cpp_nms( self, dets, scores, thresh ):
        order = scores.argsort()[::-1]
        keep = self._bbo.nms( dets.ravel().tolist(), scores.ravel().tolist(), order.tolist(), thresh )
        return keep

    def py_nms(self, dets, scores, thresh):
        """Pure Python NMS baseline."""
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

    def mirror_boxes( self, boxes, width ):
        boxes = boxes[:,(2,1,0,3)]
        boxes[:,0] = width - boxes[:,0]
        boxes[:,2] = width - boxes[:,2]
        return boxes

    def intersection_area( self, b0, b1 ):
        ix = np.min( [ b0[2], b1[2] ] ) - np.max( [ b0[0], b1[0] ] )
        iy = np.min( [b0[3], b1[3] ] ) - np.max( [ b0[1], b1[1] ] )
        return np.max( [ix,0.0] ) * np.max( [iy,0.0] )

    def area( self, b ):
        return ( b[2] - b[0] ) * ( b[3] - b[1] )

    def py_overlap( self, rects0, rects1 ):
        out = np.zeros( (len(rects0), len(rects1) ), dtype=np.float32 )

        for i,r0 in enumerate( rects0 ):
            for j,r1 in enumerate( rects1 ):
                ii = self.intersection_area( r0, r1 )
                a0 = self.area( r0 )
                a1 = self.area( r1 )

                out[i,j] = ii/( a0+a1-ii + 1e-10 )
        return out


    def is_inside_mask( self, rects, masks, thresh ):
        check = self._bbo.is_inside_mask( rects.ravel().tolist(), masks.ravel().tolist(), thresh )
        return np.array( check )
