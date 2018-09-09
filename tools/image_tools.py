import numpy as np
#import cv2

#color_codes = {}
#color_codes['hsv'] = cv2.COLOR_RGB2HSV

def image_scale( imsize, frame_size ):
    scale = 1.0
    if np.max( imsize ) >= frame_size :
        scale = (frame_size) / np.max( imsize )
    return scale

def image_scale_fit( imsize, frame_size ):
    max_size = np.max( imsize )
    scale = (frame_size) / max_size
    return scale

def prepare_image_embed( image, pixel_means, frame_size, colorspace='rgb' ):
    im = image.im # RGB Image

    if colorspace is not 'rgb' :
        im = cv2.cvtColor( im, color_codes[colorspace] )

    scale = image_scale( im.shape[:2], frame_size )

    if scale != 1.0 :
        im = cv2.resize( im, None, None, scale, scale, cv2.INTER_LINEAR )

    im = im.astype( np.float32, copy=False ) - pixel_means
    h,w = im.shape[:2]

    frame = np.zeros(( frame_size, frame_size, 3 )).astype( np.float32 )
    frame[0:h,0:w,:] = im

    return frame, np.array([h,w]), scale

def prepare_image_scale( image, pixel_means, frame_size, colorspace='rgb' ):
    im = image.im # RGB Image

    if colorspace is not 'rgb' :
        im = cv2.cvtColor( im, color_codes[colorspace] )

    scale = image_scale( im.shape[:2], frame_size )

    if scale != 1.0 :
        im = cv2.resize( im, None, None, scale, scale, cv2.INTER_LINEAR )

    im = im.astype( np.float32, copy=False ) - pixel_means
    return im, scale

def prepare_image_fit( image, pixel_means, frame_size, colorspace='rgb' ):
    im = image.im

    if colorspace is not 'rgb' :
        im = cv2.cvtColor( im, color_codes[colorspace] )

    scale = image_scale_fit( im.shape[:2], frame_size )
    im = cv2.resize( im, None, None, scale, scale, cv2.INTER_LINEAR )

    h,w = im.shape[:2]
    im = im.astype( np.float32, copy=False ) - pixel_means

    frame = np.zeros(( frame_size, frame_size, 3 )).astype( np.float32 )
    frame[0:h,0:w,:] = im

    return frame, np.array([h,w]), scale
