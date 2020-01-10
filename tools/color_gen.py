import random
import colorsys

def random_color(n):
    ret = []
    for _ in range(n):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        ret.append((r,g,b)) 
    return ret

def random_color_fix(n, seed=123):
    random.seed(seed)
    ret = []
    for _ in range(n):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        ret.append((r,g,b)) 
    return ret

# from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: tuple(int(x*255) for x in colorsys.hsv_to_rgb(*c)), hsv))
    #random.shuffle(colors)
    return colors