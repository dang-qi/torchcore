import numpy as np
#:wqimport cv2

def mirror_boxes( boxes, im_shape ):
    width = im_shape[1]
    tmp = boxes.copy()

    tmp[:,2] = width - boxes[:,0]
    tmp[:,0] = width - boxes[:,2]

    return tmp

def mirror_keypoints( keypoints, im_shape ):
    width = im_shape[1]

    for kk in keypoints :
        inds = np.where( kk[:,2]>0 )[0]
        kk[inds,0] = width - kk[inds,0]

def draw_boxes( image ):
    tmp = image.im
    gtboxes = image.gtboxes

    for b in gtboxes :
        b = np.round(b).astype( int )
        cv2.rectangle( tmp, (b[0],b[1]), (b[2],b[3]), (0,0,255),2 )

    return tmp

def draw_keypoints( image ):
    tmp = image.im
    keypoints = image.keypoints

    for kps in keypoints :
        kps = np.round( kps ).astype( int )
        for i in range( kps.shape[0] ):
            if kps[i,2] > 0 :
                k = kps[i,:].ravel()
                cv2.circle( tmp, (k[0],k[1]), 2, (0,0,255), 2 )

    return tmp
