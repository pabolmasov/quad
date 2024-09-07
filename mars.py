import numpy
from numpy import *

'''
realization of the marching square method on a toroidal 2D-array.
'''

def caseindex(w):
    '''
    for a Boolean 2D toroidal array, calculates a 'case index' map required by the marching squares algorithm
    '''

    c = zeros(shape(w), dtype = int)
    
    w0 = roll(w, 1, axis=0)
    w1 = roll(w, 1, axis=1)
    w2 = roll(roll(w, 1, axis=0), 1, axis=1)

    # Cases 0-15
    c = w + 2 * w0 + 4 * w2 + 8 * w1
    
    return c

def curvestep(casa, prevdxy = [-1, -1]):
    '''
    sets the sequence of the squared
    prevdxy is the direction towards the previous square (+-1, +-1)
    '''
    if casa==0 or casa==15:
        dx = prevdxy[0] # back!!!
        dy = prevdxy[1]
        ddx = 0. ; ddy = 0.
    elif casa==1 or casa==4 or casa==11 or casa==14:
        dx = prevdxy[1]
        dy = prevdxy[0]
        ddx = -prevdxy[0]
        ddy = -ddx
    elif casa==2 or casa==5 or casa==7 or casa==8 or casa==10 or casa==13:
        dx = -prevdxy[1]
        dy = -prevdxy[0]
        ddx = -prevdxy[1]
        ddy = ddx
    elif casa==3 or casa==12:
        dx = -prevdxy[0]
        dy = 0
        ddx = double(dx) ; ddy = 0.
    elif casa==6 or casa==9:
        dx = 0
        dy = -prevdxy[1] # up if direction=1
        ddx = 0. ; ddy = double(dy)
    else:
        print("casa = ", casa)
    return dx, dy, ddx, ddy

def whirl(casemap, xstart, ystart, startcase):
    '''
    marching square contouring on a sphere
    stops when the contour
    '''
    nx, ny = shape(casemap)
    xs = arange(nx) ; ys = arange(ny)
    xs2, ys2 = meshgrid(xs, ys, indexing='ij')
        
    x = xstart ; y = ystart
    
    kx = x ; ky = y
    
    # print("starting case = ", startcase)
        
    if startcase==3 or startcase==12 or startcase==1 or startcase==5 or startcase==7 or startcase==8 or startcase==10 or startcase==14:
        prevdxy = [1, 0]
    elif startcase==6 or startcase==9 or startcase==2 or startcase==4 or startcase==11 or startcase==13:
        prevdxy = [0,1]
    
    # prevdxy = [xs2[wtransition][1]-xs2[wtransition][0], ys2[wtransition][1]-ys2[wtransition][0]]
    # print("prev. square: ", prevdxy)
    
    avail = ones([nx, ny], dtype = bool)
    xlist = [xstart] ; ylist = [ystart]
    kxlist  = [] ;  kylist  = []
        
    while(avail[kx,ky] & (casemap[kx,ky]>0) & (casemap[kx,ky]<15)) :
    
        # print("coords and case: ", x, y, casemap[kx,ky])
        avail[kx,ky] = 0
        dx, dy, ddx, ddy = curvestep(casemap[kx,ky], prevdxy = prevdxy)
        # print("coords and case at halfstep: ", (kx+dx)%nx, (ky+dy)%ny, casemap[(kx+dx)%nx, (ky+dy)%ny])
        # print("curve x, y = ", x + ddx, y + ddy)
        '''
        if (1-avail[kx+dx,ky+dy]) | (casemap[kx+dx, ky+dy]==0) |(casemap[kx+dx, ky+dy]==15):
            print("direction change")
            # direction = -direction
            dx, dy, ddx, ddy = curvestep(casemap[kx,ky], prevdxy = prevdxy)
        '''
        prevdxy = [-dx, -dy]
        
        kx += dx ; ky += dy
        x += ddx ; y += ddy
        # x = x%nx ; y = y%ny # designed for toroidal
        kx = kx%nx ; ky = ky%ny # designed for toroidal
        # print("coords and case after the step: ", x, y, casemap[kx,ky])
        xlist.append(x) ; ylist.append(y)
        kxlist.append(kx) ; kylist.append(ky)
        
    return kxlist, kylist, xlist, ylist
    
def allwhirls(w):
    '''
    finds all the loops in the plot for a given casemap
    '''
    
    casemap = caseindex(w)
    
    nx, ny = shape(casemap)
    xs = arange(nx) ; ys = arange(ny)
    xs2, ys2 = meshgrid(xs, ys, indexing='ij')

    wtransition = (casemap > 0) & (casemap < 15)

    kxlistlist = [] ; kylistlist = []
    xlistlist = [] ; ylistlist = []

    while (wtransition.sum() > 0):
        xstart = xs2[wtransition][0]
        ystart = ys2[wtransition][0]
        startcase = casemap[wtransition][0]
        kxlist, kylist, xlist, ylist =  whirl(casemap, xstart, ystart, startcase)
        kxlistlist.append(kxlist)  ; kylistlist.append(kylist)
        xlistlist.append(xlist) ; ylistlist.append(ylist)
        casemap[kxlist, kylist] = 0
        wtransition = (casemap > 0) & (casemap < 15)

    return kxlistlist, kylistlist, xlistlist, ylistlist

# output are 4 lists of lists:
# kx coordinates of the squares,
# ky coordinates of the squares
# x normalized coordinates of the contours
# y normalized coordinates of the contours
