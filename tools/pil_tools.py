import numpy as np

def draw_rectangle( draw, b, thickness=3, color=(255,0,0) ):
    b = np.round( b ).astype( int ).tolist()
    ht = np.floor( thickness / 2 )
    offset = np.arange( -ht, ht, 1 )

    for o in offset :
        draw.rectangle( (np.array(b)+o).tolist(), outline=color )

def draw_circle( draw, p, r, color=(255,0,0) ):
    p = np.round( p ).astype( int ).tolist()

    y = p[1]
    x = p[0]

    draw.ellipse( (x-r,y-r,x+r,y+r), fill=color )

def draw_line( draw, p0, p1, color=(0,0,255) ):
    p0 = np.round( p0 ).astype( int ).tolist()
    p1 = np.round( p1 ).astype( int ).tolist()

    p0 = tuple( p0 )
    p1 = tuple( p1 )

    draw.line( [ p0, p1 ], fill=color, width=3 )
