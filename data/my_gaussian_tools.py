import numpy as np
def ellipse_gaussian_radius(width, height, IoU=0.7):
    ratio = 1 - np.sqrt(1-(1-IoU)/(1+IoU))
    r_h = height * ratio 
    r_w = width * ratio
    return r_w, r_h

def ellipse_gaussian_2D(r_w, r_h, sigma_x, sigma_y):
    r_w, r_h = int(r_w), int(r_h)
    y, x = np.ogrid[-r_h:r_h+1, -r_w:r_w+1]

    h = np.exp(-0.5*(x*x/ (sigma_x*sigma_x) + y*y/(sigma_y*sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_ellipse_gaussian(heatmap, c_x, c_y, r_w, r_h, sigma_x, sigma_y):
    '''c:center, r:radius'''
    gaussian_map = ellipse_gaussian_2D(r_w, r_h, sigma_x, sigma_y)

    c_x, c_y = int(c_x), int(c_y)
    r_w, r_h = int(r_w), int(r_h)
    height, width = heatmap.shape
    left = min(c_x, r_w)
    top = min(c_y, r_h)
    right = min(r_w, width-c_x)
    down = min(r_h, height-c_y)

    masked_heatmap = heatmap[c_y-top:c_y+down, c_x-left:c_x+right]
    masked_gaussian = gaussian_map[r_h-top:r_h+down, r_w-left:r_w+right]
    np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap
