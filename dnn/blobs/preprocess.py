import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter

class preprocess_func :
     def __call__( self, image ):
         return image

class identity( preprocess_func ):
    def __call__( self, image ):
        return image.astype( np.float32 )

class vgg( preprocess_func ):
    def __init__( self, vgg_mean=None ):
        if vgg_mean is None :
            vgg_mean = np.array( [123.68, 116.78, 103.94], dtype=np.float32 )
        self._vgg_mean = vgg_mean

    def __call__( self, image ):
        return image.astype( np.float32 ) - self._vgg_mean

class rgb2gray( preprocess_func ):
    def __call__( self, image ):
        image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
        image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        return image

class rgb2gray_rand( preprocess_func ):
    def __init__( self, prob ):
        self._prob = prob
        self._rgb2gray = rgb2gray()

    def __call__( self, image ):
        r = np.random.rand()
        if r < self._prob :
            image = self._rgb2gray( image )
        return image

class img2float( preprocess_func ):
    def __call__( self, image ):
        return image.astype( np.float32 )/255.0

class blur( preprocess_func ):
    def __init__( self, kernel ):
        self._kernel = kernel

    def __call__( self, image ):
        return cv2.GaussianBlur(image,self._kernel,0)

class preprocess_class :
    def __init__( self ):
        self._funcs = {}
        self._funcs['identity'] = identity
        self._funcs['vgg'] = vgg
        self._funcs['rgb2gray'] = rgb2gray
        self._funcs['rgb2gray_rand'] = rgb2gray_rand
        self._funcs['img2float'] = img2float
        self._funcs['blur'] = blur

    def __call__( self, preproc_list ):
        out = []

        for item in preproc_list :
            name = item[0]
            args = item[1:]
            out.append( self._funcs[name]( *args ) )

        return out

def preprocess( preproc_list ):
    pp = preprocess_class()
    return pp( preproc_list )
