import matplotlib
from matplotlib import rc
import numpy
from numpy import *
import numpy.ma as ma
import numpy.linalg as LA
from pylab import *
from scipy.optimize import root_scalar
from scipy.integrate import *
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.signal import *

import h5py

#Uncomment the following if you want to use LaTeX in figures
rc('font',**{'family':'serif','serif':['Times']})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"]
# plotting:
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager
# from matplotlib.patches import Wedge
from matplotlib import use
# ioff()
# use('Agg')

# for normal distribution fit:
# import matplotlib.mlab as mlab
# from scipy.stats import norm
# for curve fitting:
from scipy.optimize import curve_fit
import os

from numpy.ma import masked_array

from math import factorial

# ifmayavi = True
# if ifmayavi:
import sys
sys.path.append('/usr/local/lib/python3.9')
import mayavi.mlab as maya
from mlabtex import mlabtex # Sebastian Muller's mlabtex
maya.options.offscreen = False

import gc

import mars as mars

cmap = 'Oranges'

def burnaway(w):
    '''
    w is a boolean 2d-array
    '''
    
    wnew = copy(w)
    neighbours = copy(w)

    # print((w00&w10&w01&w11).sum())

    while(((roll(wnew,1,axis=0)|roll(wnew, -1, axis=0)|roll(wnew, 1, axis=1)|roll(wnew, -1, axis=1))&wnew).sum()>0):
        # first: first side
        neighbours1 = roll(wnew,1,axis=0)|roll(wnew,-1,axis=1)
        neighbours = roll(wnew,1,axis=0)|roll(wnew, 1, axis=1)|roll(wnew,-1,axis=0)|roll(wnew, -1, axis=1)
        wnew = (wnew&neighbours1)|(wnew&(1-neighbours))
        neighbours2 = roll(wnew,1,axis=0)|roll(wnew,1,axis=1)
        neighbours = roll(wnew,1,axis=0)|roll(wnew, 1, axis=1)|roll(wnew,-1,axis=0)|roll(wnew, -1, axis=1)
        wnew = (wnew&neighbours2)|(wnew&(1-neighbours))
        neighbours3 = roll(wnew,-1,axis=0)|roll(wnew,-1,axis=1)
        neighbours = roll(wnew,1,axis=0)|roll(wnew, 1, axis=1)|roll(wnew,-1,axis=0)|roll(wnew, -1, axis=1)
        wnew = (wnew&neighbours3)|(wnew&(1-neighbours))
        neighbours4 = roll(wnew,-1,axis=0)|roll(wnew,1,axis=1)
        neighbours = roll(wnew,1,axis=0)|roll(wnew, 1, axis=1)|roll(wnew,-1,axis=0)|roll(wnew, -1, axis=1)
        wnew = (wnew&neighbours4)|(wnew&(1-neighbours))

        print("number of points = ", wnew.sum())
        # neighbours = roll(wnew,1,axis=0)|roll(wnew, 1, axis=1)|roll(wnew,-1,axis=0)|roll(wnew, -1, axis=1) # |roll(roll(wnew,1, axis=0),1,axis=1)|roll(roll(wnew,1, axis=1),1,axis=0)|roll(roll(wnew,-1, axis=1),1,axis=0)|roll(roll(wnew,1, axis=1),-1,axis=0)
    
        # wnew = (wnew&roll(wnew,1,axis=0)&roll(wnew, 1, axis=1)&roll(wnew,-1,axis=0)&roll(wnew, -1, axis=1))|((1-neighbours)&wnew)
        
    return wnew.sum()

def burntester(nx, ny):

    x = arange(nx)/double(nx)  ;   y = arange(ny)/double(ny)
    
    x2, y2 = meshgrid(x, y)
    
    r = sqrt(minimum((x2-0.2)**2 + (y2-0.5)**2, (x2-0.8)**2 + (y2-0.5)**2))
    
    r = sqrt((x2-0.5)**2 + (y2-0.5)**2)
    
    r = r + 1./(r+0.01)
    print("r = ", r.min(), '..', r.max())
    ii = input('r')
    w  = (r < 10.)
    
    ww = burnaway(w)
    print('number of regions: ', ww) # x2[ww>0], y2[ww>0])

def hist3(lists, radii, nbins=10, vmin = 0., vmax = 1.0, blog = False, xtitle = None, ytitle = None, filename = 'dist3.png'):
# [pdist1, pdist2, pdist3], nbins = 10, vmin = 0.0, vmax = 10.0, xlabel = r'$P / R_{\rm *}$', ylabel = r'${\rm d} N / {\rm d} P$', filename = dirname+'/pdist.png'):
    if blog:
        bins = (vmax/vmin)**(arange(nbins+1)/double(nbins)) * vmin
        if vmin <= 0.0:
            print(ytitle, ": vmin = ", vmin)
            print("attempted logarithmic scale with a non-positive normalization")
            exit(1)
        bins[0] = 0. # let first bin start with 0
    else:
        bins = arange(nbins+1)/double(nbins) * (vmax-vmin) + vmin
        
    list1 = lists[0]
    list2 = lists[1]
    list3 = lists[2]

    dist1 = zeros(nbins) ; dist2 = zeros(nbins) ; dist3 = zeros(nbins)

    for q in arange(nbins):
        w = (list1 >= bins[q]) * (list1 < bins[q+1])
        dist1[q] += w.sum()
        w = (list2 >= bins[q]) * (list2 < bins[q+1])
        dist2[q] += w.sum()
        w = (list3 >= bins[q]) * (list3 < bins[q+1])
        dist3[q] += w.sum()

    bins_c = (bins[1:]+bins[:-1])/2.
    bins_s = (bins[1:]-bins[:-1])/2.
    
    '''
    if blog:
        dist1 = ma.masked_array(dist1, mask = dist1 <= 0.)
        dist2 = ma.masked_array(dist2, mask = dist2 <= 0.)
        dist3 = ma.masked_array(dist3, mask = dist3 <= 0.)
    '''

    w1 = dist1>0.
    w2 = dist2>0.
    w3 = dist3>0.

    clf()
    plot(bins_c[w1], (dist1/bins_s)[w1], 'k.', label = r'$r = '+str(round(radii[0],2))+'$')
    errorbar(bins_c[w1], (dist1/bins_s)[w1], xerr = bins_s[w1], yerr = (sqrt(dist1)/bins_s)[w1], fmt = 'k-')
    plot(bins_c[w2], (dist2/bins_s)[w2], 'rd', label = r'$r = '+str(round(radii[1],2))+'$')
    errorbar(bins_c[w2], (dist2/bins_s)[w2], xerr = bins_s[w2], yerr = (sqrt(dist2)/bins_s)[w2], fmt = 'r-')
    plot(bins_c[w3], (dist3/bins_s)[w3], 'b*', label = r'$r = '+str(round(radii[2],2))+'$')
    errorbar(bins_c[w3], (dist3/bins_s)[w3], xerr = bins_s[w3], yerr = (sqrt(dist3)/bins_s)[w3], fmt = 'b-')
    legend()
    if blog:
        xscale('log')
        yscale('log')
    xlabel(xtitle)
    ylabel(ytitle)
    savefig(filename)

def toSpherical(x, y, z):
                
    r = sqrt(x**2+y**2+z**2)
    th = arccos(z / r)
    sth = sin(th) ; cth = cos(th)
    ph = arctan2(y,x) % (2.*pi)

    return r, th, ph, sth, cth
    
def dq(dvector, qmatrix, r):
    '''
    r is the radius vector
    qmatrix is the quadruple moment
    r is a single radius vector
    '''
  
    musum = (dvector * r).sum()
    rsq = (r**2).sum()
    qsum = 0.
    b_q = zeros(3)
    for k in arange(3):
        qsum += (qmatrix[k,:] * r[:]).sum() * r[k]
    b_d = dvector * rsq - 3. * musum * r # dipolar component normalized by r^5
    for k in arange(3):
        b_q[k] = (qmatrix[:,k] * r[:]).sum() - 2.5 * qsum * r[k] / rsq
    
    return b_d[0]+b_q[0], b_d[1]+b_q[1], b_d[2]+b_q[2]

def dq_array(dvector, qmatrix, x, y, z, r1 = 1.0):
    '''
    x, y, and z are radius-vector coordinate arrays, each setting a point in space
    '''
    s = shape(x)
    nx = size(x)
    
    xf = x.flatten() ; yf = y.flatten() ; zf = z.flatten()
    
    bx = zeros(nx) ; by = zeros(nx) ; bz = zeros(nx)
    
    for k in arange(nx):
        rsq = xf[k]**2 + yf[k]**2 + zf[k]**2
        if (rsq >= r1**2):
            bx[k], by[k], bz[k] = dq(dvector, qmatrix, asarray([xf[k], yf[k], zf[k]]))
    
    bx = reshape(bx, s) ; by = reshape(by, s) ; bz = reshape(bz, s)
    
    return bx, by, bz

def plotquad(dvector, qmatrix, r2 = 10., r1 = 1.):
    '''
    calculates fields on a regular array
    '''

    xmax = r2
    
    nx = 50 ; ny = 51 ; nz = 52
    
    x = xmax * (arange(nx)/double(nx-1)-0.5)
    y = xmax * (arange(ny)/double(ny-1)-0.5)
    z = xmax * (arange(nz)/double(nz-1)-0.5)

    x3, y3, z3 = meshgrid(x,y,z, indexing='ij')
    
    bx, by, bz = dq_array(dvector, qmatrix, x3, y3, z3)

    return x3, y3, z3, bx, by, bz


def onetrack_outwards(dvector, qmatrix, r1, th1, phi1, th2 = pi/2., ifpath = False):
    '''
    traces the photon trajectory from the initial point on the surface towards the surface theta=th2
    '''
    
    dt = 1e-2
    dtout = 0.2
    tstore = 0.
    t = 0.

    tlist = [] ; xlist = [] ; ylist = [] ; zlist = [] ; blist = []

    rr = r1
    thth = th1
    phph = phi1
    
    sth = sin(thth) ; cth = cos(thth)
    xx = rr * sth * cos(phph)
    yy = rr * sth * sin(phph)
    zz = rr * cth

    tlist.append(0.) ; xlist.append(xx) ; ylist.append(yy) ; zlist.append(zz)
    bx, by, bz = dq(dvector, qmatrix, asarray([xx, yy, zz]))
    rr = sqrt(xx**2+yy**2+zz**2)
    b = sqrt(bx**2+by**2+bz**2)
    bbx = bx/b ; bby = by/b ; bbz = bz/b # normalized components
    b /= rr**2 # now we have a field normalized by r^3, const for dipole
    
    if (bx * xx + by * yy + bz * zz) > 0.:
        brsign = 1.0
    else:
        brsign = -1.0
    
    if ifpath:
        blist.append(b)
        
    if th1==th2:
        print(th1, ' = ', th2)
        exit(1)

    while(((th1-th2)*(thth-th2))>0.):
        bx, by, bz = dq(dvector, qmatrix, asarray([xx, yy, zz]))
        rr, thth, phph, sth, cth = toSpherical(xx, yy, zz)
        b = sqrt(bx**2+by**2+bz**2)
        bbx = bx/b ; bby = by/b ; bbz = bz/b # normalized components
        b /= rr**2 # now we have a field normalized by r^3, const for dipole
            
        xx1 = xx + bbx * rr * dt/2. * brsign
        yy1 = yy + bby * rr * dt/2. * brsign
        zz1 = zz + bbz * rr * dt/2. * brsign
        rr1 = sqrt(xx**2+yy**2+zz**2)

        bx1, by1, bz1 = dq(dvector, qmatrix, asarray([xx1, yy1, zz1]))
        b1 = sqrt(bx1**2+by1**2+bz1**2)
        bbx1 = bx1/b1 ; bby1 = by1/b1 ; bbz1 = bz1/b1 # normalized components
        b1 /= rr1**2 # now we have a field normalized by r^3, const for dipole

        xprev = xx ; yprev = yy ; zprev = zz

        xx += bbx1 * rr1 * dt * brsign
        yy += bby1 * rr1 * dt * brsign
        zz += bbz1 * rr1 * dt * brsign
        
        t += dt
        
        rr, thth, phph, sth, cth = toSpherical(xx, yy, zz)
        if (rr < r1):
            # closed line:
            return sqrt(-1), sqrt(-1)
        
        if t > tstore:
            # print("theta = ", thth, "; r = ", rr)
            if ifpath:
                xlist.append(xx)
                ylist.append(yy)
                zlist.append(zz)
                tlist.append(t)
            
            tstore = t+dtout
            # rr = sqrt(xx**2+yy**2+zz**2)
            bx, by, bz = dq(dvector, qmatrix, asarray([xx, yy, zz]))
            b = sqrt(bx**2+by**2+bz**2)
            # bbx = bx/b ; bby = by/b ; bbz = bz/b # normalized components
            b /= rr**2 # now we have a field normalized by r^3, const for dipole
            if ifpath:
                blist.append(b)

    # rprev = sqrt(xprev**2+yprev**2+zprev**2)
    rr, thth, phph, sth, cth = toSpherical(xx, yy, zz)
    rprev, thprev, phprev, sth_prev, cth_prev = toSpherical(xprev, yprev, zprev)
    dtlast = (th2-thth)/(thth-thprev) * dt

    xx = xprev + bbx * rr * (dt+dtlast) * brsign
    yy = yprev + bby * rr * (dt+dtlast) * brsign
    zz = zprev + bbz * rr * (dt+dtlast) * brsign

    rr, thth, phph, sth, cth = toSpherical(xx, yy, zz)

    bx, by, bz = dq(dvector, qmatrix, asarray([xx, yy, zz]))
    b = sqrt(bx**2+by**2+bz**2)
    bbx = bx/b ; bby = by/b ; bbz = bz/b # normalized components
    b /= rr**2 # now we have a field normalized by r^3, const for dipole

    rr, thth, phph, sth, cth = toSpherical(xx, yy, zz)
    # print("last radius = ", rprev, rr)

    if ifpath:
        return rr, phph, xlist, ylist, zlist, blist
    else:
        return rr, phph

def qrand(ifeigen = False):
    '''
    generates a random quad matrix
    '''

    cqth1 = random() * 2. - 1.  ; cqth2 = random() * 2. -1.
    sqth1 = sqrt(1.-cqth1**2) ; sqth2 = sqrt(1.-cqth2**2)
    qphi1 = random() * 2. * pi ; qphi2 = random() * 2. * pi
    v1 = asarray([sqth1 * cos(qphi1), sqth1 * sin(qphi1), cqth1])
    v2 = asarray([sqth2 * cos(qphi2), sqth2 * sin(qphi2), cqth2])
    
    # now let us orthogonalize the first two vectors:
    v2 = v2 - v1 * (v1 * v2).sum()
    v2 = v2 / sqrt((v2**2).sum())

    v3 = asarray([v1[1]*v2[2]-v2[1]*v1[2], v1[2]*v2[0]-v2[2]*v1[0], v1[0]*v2[1]-v2[0]*v1[1]]) # vector product of v1 and v2
    v3 /= sqrt((v3**2).sum()) # normalized
    
    # eigenvalues:
    lcth = random() * 2. - 1. ; lphi =  random() * 2. * pi
    lsth = sqrt(1.-lcth**2)
    l1 = (lsth * cos(lphi))**2-1./3. ; l2 = (lsth * sin(lphi))**2-1./3. ; l3 = lcth**2-1./3.
    ldiag = zeros([3,3])
    ldiag[0,0] = l1 ; ldiag[1,1] = l2 ; ldiag[2,2] = l3

    # now turning the tensor along the eigenvectors
    vecmatrix = zeros([3,3])
    vecmatrix[:,0] = v1 ; vecmatrix[:,1] = v2 ; vecmatrix[:,2] = v3

    qmatrix = matmul(matmul(vecmatrix, ldiag), LA.inv(vecmatrix)) # tensor of quad momentum

    if ifeigen:
        return qmatrix, v1, v2, v3, [l1, l2, l3]
    else:
        return qmatrix

def skymap(alpha = 0., psi = 0., qscale = 1.0, qsave = None, rrange = [10.,1e4]):
    '''
    constructs the map of the NS surface, calculating the radius-at-pi/2 and phi-at-pi/2 for every surface element
    the array of rrange is used to calculate the number of ``polar caps'', their perimeters and surface areas
    after that, one can make statistics
    '''
    
    # inclined dipole (fixed)
    dvector = asarray([sin(alpha)*cos(psi), sin(alpha)*sin(psi), cos(alpha)])

    # random quadrupole:
    if qsave is None:
        qmatrix, v1, v2, v3, ls = qrand(ifeigen=True)
    else:
        qmatrix, v1, v2, v3, ls = qsave
    
    # mesh on the surface
    cth1 = 1.; cth2 = -1.0 ; nth = 302
    cth = (cth2-cth1) * (arange(nth)+0.5)/double(nth)+cth1
    cthext = (cth2-cth1) * arange(nth+1)/double(nth)+cth1 # uniform in cos(theta)
    th = arccos(cth)
    thext = arccos(cthext)
    phi1 = 0.; phi2 = 2.*pi ; nphi = 303
    phi = (phi2-phi1)* (arange(nphi)+0.5)/double(nphi)+phi1
    phiext = (phi2-phi1)* (arange(nphi+1))/double(nphi)+phi1
    
    dphi = 2.*pi / double(nphi)
    dctheta = 2. / double(nth)

    th2, phi2 = meshgrid(thext, phiext)
    
    rstart = zeros([nth, nphi])
    phistart = zeros([nth, nphi])
    bsurf = zeros([nth, nphi])

    for kth in arange(nth):
        # print('working on theta = ', th[kth])
        for kphi in arange(nphi):
            rstart_tmp, phistart_tmp =  onetrack_outwards(dvector, qmatrix * qscale, 1., th[kth], phi[kphi], th2 = pi/2.)
            rstart[kth, kphi] = rstart_tmp   ;   phistart[kth, kphi] = phistart_tmp
            bx, by, bz = dq(dvector, qmatrix*qscale, asarray([sin(th[kth])*cos(phi[kphi]), sin(th[kth])*sin(phi[kphi]), cos(th[kth])]))
            b = sqrt(bx**2+by**2+bz**2)
            bbx = bx/b ; bby = by/b ; bbz = bz/b # normalized components
            b /= 1.**2 # now we have a field normalized by r^3, const for dipole
            bsurf[kth, kphi] = b

    rstart = transpose(rstart)
    phistart = transpose(phistart)
    # bsurf = transpose(bsurf)
    
    # estimate the number of polar caps
    # w = (rstart>=rrange[0])&(rstart<rrange[1])
    # print(shape(bsurf), shape(w), " = shapes")
    # ncaps = burnaway(w)
    
    # print(ncaps)
    phifun = interp1d(arange(nphi), phi, bounds_error=False, fill_value='extrapolate')
    cthfun = interp1d(arange(nth), cth, bounds_error=False, fill_value='extrapolate')

    nr = size(rrange)
    ncaps = zeros(nr, dtype=int)
    perimeters = zeros(nr)
    areas = zeros(nr)
    npoints = zeros(nr)

    clf()
    fig = figure()
    subplot(211)
    pcolormesh(phi2, th2, log10(rstart), cmap = 'jet', shading='flat')
    cb = colorbar()
    cb.set_label(r'$\log_{10}R/R_*$', fontsize=14)
    contour(phi, th, bsurf, colors='w')
    contour(phi, th, bsurf, colors='k', linestyles='dotted')
    for kr in arange(nr):
        w = rstart > rrange[kr]
        kxsquares, kysquares, xcontours, ycontours = mars.allwhirls(w)
        areas[kr] = dctheta * dphi * double(w).sum()
        npoints[kr] = w.sum()
        ncaps[kr] = len(kxsquares)
        # cs = contour(phi, th, transpose(rstart > rrange[0]), levels=[0.5])
        # polys = cs.allsegs[0]
        print("found ", ncaps[kr], " contours")
    
        # print(shape(polys))
        # ncaps = shape(polys)[0]
        perimeter = 0.0
        # phiclist = [] ; thclist = []
        for q in arange(len(kxsquares)):
            phi_c = phifun(xcontours[q])
            cth_c = cthfun(ycontours[q])
            # phiclist.append(phi_c)  ; thclist.append(th_c)
            dca = (roll(cth_c,1)*cth_c + roll(sqrt(maximum(1.-cth_c**2, 0.)),1)*sqrt(maximum(1. - cth_c**2,0.)) * cos(roll(phi_c,1)-phi_c))
            da = arccos(minimum(maximum(dca, -1.), 1.))
            # print(shape(phi_c), shape(cth_c))
            perimeter += da.sum()
            #print(da)
            plot(phifun(xcontours[q]), arccos(cthfun(ycontours[q])), 'r--')
            # ii = input('i')
        if ncaps[kr]>0:
            perimeters[kr] = perimeter.sum() # total perimeter for given rrange
            print("perimeter (R = "+str(rrange[kr])+") = ", perimeters[kr], "+/-", perimeters[kr]/sqrt(npoints[kr]))
        
    # contour(phi, th, transpose(w), colors='r', linestyles='dashed', levels=[0.5])
    plot([psi, psi+pi, psi+2.*pi, psi+3.*pi], [alpha, pi-alpha, alpha, pi-alpha], '*k') # dipole axis
    # quaqrupolar eigenvectors:
    plot([arctan2(v1[1],v1[0]), arctan2(v1[1],v1[0])+pi], [arccos(v1[2]), pi-arccos(v1[2])], 'og')
    plot([arctan2(v2[1],v2[0]), arctan2(v2[1],v2[0])+pi], [arccos(v2[2]), pi-arccos(v2[2])], 'og')
    plot([arctan2(v3[1],v3[0]), arctan2(v3[1],v3[0])+pi], [arccos(v3[2]), pi-arccos(v3[2])], 'og')
    ylabel(r'$\theta$', fontsize=14)
    ylim(pi, 0.) ; xlim(0.,2.*pi)
    subplot(212)
    pcolormesh(phi2, th2, phistart, cmap = 'hsv', shading='flat')
    cb1 = colorbar()
    cb1.set_label(r'$\varphi_{\rm disc}$', fontsize=14)
    contour(phi, th, bsurf, colors='w')
    contour(phi, th, bsurf, colors='k', linestyles='dotted')
    plot([psi, psi+pi, psi+2.*pi, psi+3.*pi], [alpha, pi-alpha, alpha, pi-alpha], '*k') # dipole axis
    # quaqrupolar eigenvectors:
    plot([arctan2(v1[1],v1[0]), arctan2(v1[1],v1[0])+pi], [arccos(v1[2]), pi-arccos(v1[2])], 'og')
    plot([arctan2(v2[1],v2[0]), arctan2(v2[1],v2[0])+pi], [arccos(v2[2]), pi-arccos(v2[2])], 'og')
    plot([arctan2(v3[1],v3[0]), arctan2(v3[1],v3[0])+pi], [arccos(v3[2]), pi-arccos(v3[2])], 'og')
    ylabel(r'$\theta$', fontsize=14)
    xlabel(r'$\varphi$', fontsize=14)
    ylim(pi, 0.) ; xlim(0.,2.*pi)
    fig.set_size_inches(8.,10.)
    suptitle(r'$Q$ scale = '+str(round(qscale,4)))
    savefig('skymap.png')
    
    if nr > 2:
        # plotting perimeters
        clf()
        fig = figure()
        plot(rrange, perimeters, 'k.')
        plot(rrange, 4.*pi / sqrt(rrange), 'r:')
        errorbar(rrange, perimeters, yerr = perimeters / sqrt(npoints), fmt = 'k.')
        xlabel(r'$R_{\rm out}/R_*$')  ; ylabel(r'$\Pi$')
        xscale('log') ; yscale('log')
        fig.set_size_inches(5.,4.)
        savefig('perimeters.png')
        
        # plotting areas
        clf()
        fig = figure()
        plot(rrange, areas, 'k.')
        plot(rrange, 4.*pi * (1. -  sqrt(1. - 1./asarray(rrange))), 'r:')
        errorbar(rrange, areas, yerr = areas / sqrt(npoints), fmt = 'k.')
        xlabel(r'$R_{\rm out}/R_*$')  ; ylabel(r'$A$')
        xscale('log') ; yscale('log')
        fig.set_size_inches(5.,4.)
        savefig('areas.png')

    
    return qmatrix, v1, v2, v3, ls, ncaps, perimeters, areas
    
    
def onetrack(dvector, qmatrix, r2, r1, th2 = pi/2., phi2 = 0., ifpath = False, brsign = 1.0):
    '''
    traces the motion along a single field line, but takes dipole and quadrupole moments instead of harmonics
    '''
    dt = 0.01
    dtout = 0.2
    tstore = 0.
    t = 0.
    
    tlist = [] ; xlist = [] ; ylist = [] ; zlist = [] ; blist = []
    
    rr = r2
    phph = phi2 # random() * 2. * pi
    # htor = 0.0 # disc thickness
    thth = th2
    
    sth = sin(thth) ; cth = cos(thth)
    xx = rr * sth * cos(phph)
    yy = rr * sth * sin(phph)
    zz = rr * cth

    tlist.append(0.) ; xlist.append(xx) ; ylist.append(yy) ; zlist.append(zz)
    bx, by, bz = dq(dvector, qmatrix, asarray([xx, yy, zz]))
    rr = sqrt(xx**2+yy**2+zz**2)
    b = sqrt(bx**2+by**2+bz**2)
    bbx = bx/b ; bby = by/b ; bbz = bz/b # normalized components
    b /= rr**2 # now we have a field normalized by r^3, const for dipole
    blist.append(b)

    while (rr>(r1*0.95)): # (((brsign>0.)*(rr > r1))|((brsign<0.)*(rr<=r1))) * (t < 20.):
        bx, by, bz = dq(dvector, qmatrix, asarray([xx, yy, zz]))
        rr = sqrt(xx**2+yy**2+zz**2)
        b = sqrt(bx**2+by**2+bz**2)
        bbx = bx/b ; bby = by/b ; bbz = bz/b # normalized components
        b /= rr**2 # now we have a field normalized by r^3, const for dipole
            
        xx1 = xx + bbx * rr * dt/2. * brsign
        yy1 = yy + bby * rr * dt/2. * brsign
        zz1 = zz + bbz * rr * dt/2. * brsign
        rr1 = sqrt(xx**2+yy**2+zz**2)
        
        bx1, by1, bz1 = dq(dvector, qmatrix, asarray([xx1, yy1, zz1]))
        b1 = sqrt(bx1**2+by1**2+bz1**2)
        bbx1 = bx1/b1 ; bby1 = by1/b1 ; bbz1 = bz1/b1 # normalized components
        b1 /= rr1**2 # now we have a field normalized by r^3, const for dipole

        xprev = xx ; yprev = yy ; zprev = zz

        xx += bbx1 * rr1 * dt * brsign
        yy += bby1 * rr1 * dt * brsign
        zz += bbz1 * rr1 * dt * brsign
        
        t += dt
        # ii = input('b')
        if t > tstore:
            xlist.append(xx)
            ylist.append(yy)
            zlist.append(zz)
            tstore = t+dtout
            tlist.append(t)
            
            rr = sqrt(xx**2+yy**2+zz**2)
            bx, by, bz = dq(dvector, qmatrix, asarray([xx, yy, zz]))
            b = sqrt(bx**2+by**2+bz**2)
            # bbx = bx/b ; bby = by/b ; bbz = bz/b # normalized components
            b /= rr**2 # now we have a field normalized by r^3, const for dipole
            blist.append(b)

    rprev = sqrt(xprev**2+yprev**2+zprev**2)
    rr = sqrt(xx**2+yy**2+zz**2)
    dtlast = (r1-rr)/(rr-rprev) * dt

    xx = xprev + bbx * r1 * (dt+dtlast) * brsign
    yy = yprev + bby * r1 * (dt+dtlast) * brsign
    zz = zprev + bbz * r1 * (dt+dtlast) * brsign

    rr = sqrt(xx**2+yy**2+zz**2)

    bx, by, bz = dq(dvector, qmatrix, asarray([xx, yy, zz]))
    b = sqrt(bx**2+by**2+bz**2)
    bbx = bx/b ; bby = by/b ; bbz = bz/b # normalized components
    b /= rr**2 # now we have a field normalized by r^3, const for dipole

    rr, thth, phph, sth, cth = toSpherical(xx, yy, zz)
    # print("last radius = ", rprev, rr)

    if ifpath:
        return thth, phph, xlist, ylist, zlist, blist
    else:
        return thth, phph, b
    
def mapping(alpha = pi / 3., psi = 0., qscale = 10., mayaplot = False, bplot = False, surfaceplot = False, movie=False, qsave = None):
    '''
    dipole + quadrupole magnetic field mapping
    the dipole is fixed,
    while the quadrupole component is isotropic, only its amplitude with respect to the dipole is fixed
    qscale is the relative amplitude of the quad at the surface
    '''

    r1 = 1. # radius of the NS

    # the mesh of the starting points in the disc:
    rstart1 = 3. ; rstart2 = 300. ; nrstart = 3 # radii
    rstart = (rstart2/rstart1) ** (arange(nrstart) / double(nrstart-1)) * rstart1
    
    phistart1 = 0.01 ; phistart2 = 2.*pi+0.02 ; nphistart = 150 # azimuths
    phistart = (phistart2-phistart1) * arange(nphistart) / double(nphistart-1) + phistart1
    
    rmesh2, phimesh2 = meshgrid(rstart, phistart)

    # inclined dipole (fixed)
    dvector = asarray([sin(alpha)*cos(psi), sin(alpha)*sin(psi), cos(alpha)])

    # quadrupole:
    if qsave is None:
        qmatrix, v1, v2, v3, ls = qrand(ifeigen = True)
        l1, l2, l3 = ls
        #if mayaplot:
           # print("should be diagonal: ", matmul(matmul(LA.inv(vecmatrix), qmatrix), vecmatrix))
           #  print("and traceless: l1+l2+l3 = ", l1+l2+l3)
    else:
        v1, v2, v3, ls = qsave
        l1, l2, l3 = ls
        ldiag = zeros([3,3])
        ldiag[0,0] = l1 ; ldiag[1,1] = l2 ; ldiag[2,2] = l3
        vecmatrix = zeros([3,3])
        vecmatrix[:,0] = v1 ; vecmatrix[:,1] = v2 ; vecmatrix[:,2] = v3
        qmatrix = matmul(matmul(vecmatrix, ldiag), LA.inv(vecmatrix))

    qstore = v1, v2, v3, [l1, l2, l3] # store and output for the eigenvectors and eigenvalues
    qmatrix *= qscale * r1**2

    if bplot or surfaceplot:
        fig = figure()
        clf()
        
    # now let us make a mayavi plot
    if mayaplot:
        fig = maya.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size = (500, 500))
        maya.clf()

        # coordinate axes
        maya.plot3d([0.,0.], [0.,0.,], [-2.*r1, 2.*r1], color=(0,0,0))
        maya.plot3d([0.,0.], [-2.*r1, 2.*r1], [0.,0.,], color=(0,0,0))
        maya.plot3d([-2.*r1, 2.*r1], [0.,0.], [0.,0.,], color=(0,0,0))

        # dipole axis:
        maya.plot3d([0., 2. * r1 * sin(alpha)*cos(psi)], [0., 2. * r1 * sin(alpha)*sin(psi)], [0. , 2. * r1 * cos(alpha)], color=(1.0,0.0,0.0))
        # quadrupole axes:
        maya.plot3d([-v1[0]*1.2,v1[0]*1.2], [-v1[1]*1.2,v1[1]*1.2], [-v1[2]*1.2, v1[2]*1.2], color=(0.0,0.5,0.0))
        maya.plot3d([-v2[0]*1.2,v2[0]*1.2], [-v2[1]*1.2,v2[1]*1.2], [-v2[2]*1.2, v2[2]*1.2], color=(0.0,0.5,0.0))
        maya.plot3d([-v3[0]*1.2,v3[0]*1.2], [-v3[1]*1.2,v3[1]*1.2], [-v3[2]*1.2, v3[2]*1.2], color=(0.0,0.5,0.0))
    
        # NS surface:
        nth = 200 ; nphi = 201

        theta = pi * (arange(nth)+0.5)/double(nth)
        phi = 2. * pi * arange(nphi)/double(nphi-1)

        theta2, phi2 = meshgrid(theta, phi, indexing='ij')

        m = maya.mesh(r1 * sin(theta2) * cos(phi2), r1 * sin(theta2) * sin(phi2), r1 * cos(theta2), color = (0.8, 0.8, 0.8), representation='surface', opacity = 1.0)

    # the landing points of the field lines (upper and lower hemispheres)
    surf_th_pos = zeros([nrstart, nphistart])
    surf_phi_pos = zeros([nrstart, nphistart])
    surf_th_neg = zeros([nrstart, nphistart])
    surf_phi_neg = zeros([nrstart, nphistart])

    perimeter = zeros(nrstart) ; area = zeros(nrstart)
    nmean = zeros(nrstart) ; nrms = zeros(nrstart)
    ns = zeros(nphistart*2)

    # calculating individual tracks:
    for kr in arange(nrstart):
        for kphi in arange(nphistart):
            # positive integration direction from the disc:
            th1, phi1, x, y, z, b1 = onetrack(dvector, qmatrix, rstart[kr], r1, th2 = pi/2., phi2 = phistart[kphi], ifpath = True, brsign=1.0)
            if mayaplot:
                maya.plot3d(x, y, z, line_width=0.01, tube_radius=0.01, color=(double(kr)/double(nrstart),double(kr)/double(nrstart),1))
            surf_th_pos[kr, kphi] = th1 ; surf_phi_pos[kr, kphi] = phi1
            x = asarray(x) ; y = asarray(y) ; z = asarray(z)
            r = sqrt(x**2+y**2+z**2)
            ns[kphi] = -log(b1[-1]/b1[-2])/log(r[-1]/r[-2]) + 3.
            # negative direction:
            th1, phi1, x, y, z, b1 = onetrack(dvector, qmatrix, rstart[kr], r1, th2 = pi/2., phi2 = phistart[kphi], ifpath = True, brsign=-1.0)
            if mayaplot:
                maya.plot3d(x, y, z, line_width=0.01, tube_radius=0.01, color=(double(kr)/double(nrstart),double(kr)/double(nrstart),1))
            surf_th_neg[kr, kphi] = th1 ; surf_phi_neg[kr, kphi] = phi1
            x = asarray(x) ; y = asarray(y) ; z = asarray(z)
            r = sqrt(x**2+y**2+z**2)
            ns[kphi+nphistart] = -log(b1[-1]/b1[-2])/log(r[-1]/r[-2]) + 3.

            # print(r[-1], ns[kphi])
            #            w = r<(rstart[kr]*20.)
            #           if w.sum()>0 and mayaplot:
            #              maya.plot3d(x[w], y[w], z[w], line_width=0.01, tube_radius=0.01, color=(double(kr)/double(nrstart),double(kr)/double(nrstart),1))
            if bplot:
                plot(r, b1, color=(double(kr)/double(nrstart),double(kr)/double(nrstart),1))
        # calculating the size of the hot spot:
        # da is the array of arc lengths. If da >~ R* sqrt(R*/Rstart) * dphi, it is suspicious
        dphi = (surf_phi_pos[kr,1:]-surf_phi_pos[kr,:-1] + pi ) %(2.*pi)-pi
        da = arccos(cos(surf_th_pos[kr,1:])*cos(surf_th_pos[kr,:-1]) + sin(surf_th_pos[kr,1:])*sin(surf_th_pos[kr,:-1]) * cos(dphi))
        # print("positive da = ", da)
        # print("compared to ", 2.*pi / double(nphistart-1) * sqrt(1./rstart[kr]))
        wint = da < (4.* 2.*pi / double(nphistart-1) * sqrt(1./rstart[kr]))
        perimeter[kr] = da[wint].sum()
        # print('area pos = ', (2. - cos(surf_th_pos[kr,1:]) - cos(surf_th_pos[kr,:-1])) * dphi * double(wint))
        area[kr] = ((2.- cos(surf_th_pos[kr,1:]) - cos(surf_th_pos[kr,:-1])) * dphi * double(wint)).sum() /2.
        # perimeter[kr] - double(wint.sum()-2)*pi
        #
        # now for negative brsign:
        da = arccos(cos(surf_th_neg[kr,1:])*cos(surf_th_neg[kr,:-1]) + sin(surf_th_neg[kr,1:])*sin(surf_th_neg[kr,:-1]) * cos(dphi))
        # print("negative da = ", da)
        # print("compared to ", 2.*pi / double(nphistart-1) * sqrt(1./rstart[kr]))
        wint = da < (4.* 2.*pi / double(nphistart-1) * sqrt(1./rstart[kr]))
        perimeter[kr] += da[wint].sum()
        # print('area neg = ', (2. - cos(surf_th_pos[kr,1:]) - cos(surf_th_pos[kr,:-1])) * dphi * double(wint))
        area[kr] += ((2. - cos(surf_th_pos[kr,1:]) - cos(surf_th_pos[kr,:-1])) * dphi * double(wint)).sum() /2.

        # the b slope
        # print("n slopes = ", ns)
        # both directions included in averaging:
        nmean[kr] = ns.mean() ; nrms[kr] = ns.std()
        
        if surfaceplot:
            plot(surf_phi_neg[kr, :], surf_th_neg[kr, :], '.', color=(double(kr)/double(nrstart),0,0), mfc = 'none')
            plot(surf_phi_pos[kr, :], surf_th_pos[kr, :], '.', color=(double(kr)/double(nrstart),0,0), mfc = 'none')
            plot(surf_phi_neg[kr, :]+2.*pi, surf_th_neg[kr, :], '.', color=(double(kr)/double(nrstart),0,0), mfc = 'none')
            plot(surf_phi_pos[kr, :]+2.*pi, surf_th_pos[kr, :], '.', color=(double(kr)/double(nrstart),0,0), mfc = 'none')
        if mayaplot:
            print("plotting radius R = ", rstart[kr])
            # image of the Rstart = const annulus on the surface:
            maya.plot3d(r1*1.01 * sin(surf_th_pos[kr, :]) * cos(surf_phi_pos[kr,:]), r1*1.01 * sin(surf_th_pos[kr, :]) * sin(surf_phi_pos[kr, :]), r1*1.01 * cos(surf_th_pos[kr, :]), color=(double(kr)/double(nrstart),0,0), tube_radius=0.015)
            maya.plot3d(r1*1.01 * sin(surf_th_neg[kr, :]) * cos(surf_phi_neg[kr,:]), r1*1.01 * sin(surf_th_neg[kr, :]) * sin(surf_phi_neg[kr, :]), r1*1.01 * cos(surf_th_neg[kr, :]), color=(double(kr)/double(nrstart),0,0), tube_radius=0.015)
            # the Rstart = const annulus itself
            maya.plot3d(rstart[kr] * cos(phistart), rstart[kr] * sin(phistart), 0.*phistart, color=(double(kr)/double(nrstart),0,0))
            # maya.plot3d(rstart[kr] * ((cos(phistart[1:])+cos(phistart[:-1]))/2.)[wint], rstart[kr] * ((sin(phistart[1:])+sin(phistart[:-1]))/2.)[wint], 0.*da[wint], color=(double(kr)/double(nrstart),0,0))
            # print("chain being integrated: ", arccos(cos(surf_th[kr,1:])*cos(surf_th[kr,:-1]) + sin(surf_th[kr,1:])*sin(surf_th[kr,:-1]) * cos(surf_phi[kr,1:]-surf_phi[kr,:-1])), surf_th[kr,1:] * surf_th[kr,:-1] * sin(surf_phi[kr,1:]-surf_phi[kr,:-1]))

    print("perimeters = ", perimeter)
    print("surface areas = ", area)
    print('n slope = ', nmean, "+/-", nrms)
    
    print('dipole perimeters = ', 4.*pi / sqrt(rstart))
    print('dipole surface areas = ', 4.*pi * (1.- sqrt(1. - 1./rstart)))

    if mayaplot or surfaceplot:
        # phistart = const images:
        for kphi in arange(nphistart):
            if surfaceplot:
                plot(surf_phi_pos[:,kphi], surf_th_pos[:,kphi], ',', color=(0., (sin(phistart[kphi])+1.)/2.,(cos(phistart[kphi])+1.)/2.))
                plot(surf_phi_neg[:,kphi], surf_th_neg[:,kphi], ',', color=(0., (sin(phistart[kphi])+1.)/2.,(cos(phistart[kphi])+1.)/2.))
            if mayaplot:
                maya.plot3d(r1*1.01 * sin(surf_th_pos[:, kphi]) * cos(surf_phi_pos[:, kphi]), r1*1.01 * sin(surf_th_pos[:, kphi]) * sin(surf_phi_pos[:, kphi]), r1*1.01 * cos(surf_th_pos[:, kphi]), color=(0., (sin(phistart[kphi])+1.)/2.,(cos(phistart[kphi])+1.)/2.), tube_radius=0.005)
                maya.plot3d(r1*1.01 * sin(surf_th_neg[:, kphi]) * cos(surf_phi_neg[:, kphi]), r1*1.01 * sin(surf_th_neg[:, kphi]) * sin(surf_phi_neg[:, kphi]), r1*1.01 * cos(surf_th_neg[:, kphi]), color=(0., (sin(phistart[kphi])+1.)/2.,(cos(phistart[kphi])+1.)/2.), tube_radius=0.005)
                maya.plot3d(rstart * cos(phistart[kphi]), rstart * sin(phistart[kphi]), 0.*rstart, color=(0., (sin(phistart[kphi])+1.)/2.,(cos(phistart[kphi])+1.)/2.))
        
        if mayaplot:
            # mayavi plotting of the field lines
            x3, y3, z3, bx, by, bz = plotquad(dvector, qmatrix, r2 = 10.)
            maya.flow(x3, y3, z3, bx, by, bz, seedtype='sphere', seed_resolution=20,seed_scale=1.5, colormap = cmap, seed_visible=False, integration_direction = 'both', opacity = 1.0, scalars = 0.5*log10(bx**2+by**2+bz**2+0.01), line_width=5.0)

            maya.view(azimuth=60., elevation=50, distance = r1 * 20., focalpoint = (0.,0.,0.0))
            if movie:
                nframes = 100
                for k in arange(nframes):
                    maya.view(azimuth=double(k)/double(nframes) * 400., elevation=50, distance = r1 * 5., focalpoint = (0.,0.,0.0))
                    maya.savefig('mapping_surface{:04d}.png'.format(k), magnification = 2)
                maya.close()
            else:
                maya.view(azimuth=60., elevation=50, distance = r1 * 20., focalpoint = (0.,0.,0.0))
                maya.savefig('mapping_surface.png', magnification = 2)
                if qsave is None:
                    maya.show()
        if surfaceplot:
            plot([psi, psi+pi, psi+2.*pi, psi+3.*pi], [alpha, pi-alpha, alpha, pi-alpha], 'xr') # dipole axis
        
            # quaqrupolar eigenvectors:
            plot([arctan2(v1[1],v1[0]), arctan2(v1[1],v1[0])+pi], [arccos(v1[2]), pi-arccos(v1[2])], 'og')
            plot([arctan2(v2[1],v2[0]), arctan2(v2[1],v2[0])+pi], [arccos(v2[2]), pi-arccos(v2[2])], 'og')
            plot([arctan2(v3[1],v3[0]), arctan2(v3[1],v3[0])+pi], [arccos(v3[2]), pi-arccos(v3[2])], 'og')
            xlabel(r'$\varphi$') ; ylabel(r'$\theta$')
            ylim(pi, 0. ) ; xlim(0.,3.*pi)
            savefig('surfaceplot.png')
    if bplot:
        plot(r, r*0. + 2., 'r:')
        plot(r[r<10.], qscale/r[r<10.], 'r:')
        xscale('log')
        yscale('log')
        xlabel(r'$r/r_{\rm NS}$')
        ylabel(r'$B r^3$')
        fig.set_size_inches(5.*log10(rstart.max()), 5.)
        savefig('bplot.png')
    # output: perimeters, areas, and b slopes
    return rstart, perimeter, area, nmean, nrms, qstore

def scaleshow(alpha = 0.0, psi=0.0):

    qstore = skymap(alpha = alpha, qscale=0., psi=psi)

    os.system('cp skymap.png surscale{:04d}.png'.format(0))
    # rstart, perimeter, area, nmean, nrms, qstore = mapping(alpha=pi/3., surfaceplot=True, qscale = 0.0, movie = False)
    
    print("eigenvectors = ", qstore[:-1])
    print("eigenvalues = ", qstore[-1])
    
    nsc = 100
    qmax = 20. ; qmin = 0.2
    qscales = (qmax/qmin)**(arange(nsc)/double(nsc-1))*qmin
    
    # qscales = asarray([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    # nsq = size(qscales)
    
    for k in arange(nsc):
        print("qscale = ", qscales[k])
        qstore = skymap(alpha = alpha, qscale=qscales[k], qsave = qstore, psi=psi)
        # rstart, perimeter, area, nmean, nrms, q0 = mapping(alpha=pi/3., surfaceplot=True, qscale = qscales[k], movie = False, qsave = qstore)
        os.system('cp skymap.png surscale{:04d}.png'.format(k+1))

# let us make N trials
def dists(alpha, qscale, nocalc = False):
    ntrials = 1000
    dirname = 'alpha'+str(alpha)+'_q'+str(qscale)
    os.system('mkdir '+dirname)
    
    # lists for distributions
    plist1 = [] ; alist1 = [] ;  nmlist1 = [] ; nslist1 = []
    plist2 = [] ; alist2 = [] ;  nmlist2 = [] ; nslist2 = []
    plist3 = [] ; alist3 = [] ;  nmlist3 = [] ; nslist3 = []

    # creating output files:
    if not(nocalc):
        fout1 = open(dirname+'/r1.dat', 'w') # first radius value, rstart = 3
        fout2 = open(dirname+'/r2.dat', 'w') # first radius value, rstart = 10
        fout3 = open(dirname+'/r3.dat', 'w') # first radius value, rstart = 30
        
    r, p, a, nm, ns, qsave = mapping(alpha = alpha, qscale = qscale, bplot = False, mayaplot=False)
    r1 = r[0] ; r2 = r[1] ; r3 = r[2]

    for k in arange(ntrials):
        if nocalc:
            lines = loadtxt(dirname+'/r1.dat')
            plist1 = lines[:,0] ; alist1 = lines[:,1] ; nmlist1 = lines[:,2] ; nslist1 = lines[:,3]
            lines = loadtxt(dirname+'/r2.dat')
            plist2 = lines[:,0] ; alist2 = lines[:,1] ; nmlist2 = lines[:,2] ; nslist2 = lines[:,3]
            lines = loadtxt(dirname+'/r3.dat')
            plist3 = lines[:,0] ; alist3 = lines[:,1] ; nmlist3 = lines[:,2] ; nslist3 = lines[:,3]
        else:
            print('trial ', k, '/ ', ntrials)
            r, p, a, nm, ns = mapping(alpha = alpha, qscale = qscale, bplot = False, mayaplot=False)
            fout1.write(str(p[0])+' '+str(a[0])+' '+str(nm[0])+' '+str(ns[0])+'\n')
            fout2.write(str(p[1])+' '+str(a[1])+' '+str(nm[1])+' '+str(ns[1])+'\n')
            fout3.write(str(p[2])+' '+str(a[2])+' '+str(nm[2])+' '+str(ns[2])+'\n')
            plist1.append(p[0]) ; plist2.append(p[1]) ; plist3.append(p[2])
            alist1.append(a[0]) ; alist2.append(a[1]) ; alist3.append(a[2])
            nmlist1.append(nm[0]) ; nmlist2.append(nm[1]) ; nmlist3.append(nm[2])
            nslist1.append(ns[0]) ; nslist2.append(ns[1]) ; nslist3.append(ns[2])

    if not(nocalc):
        plist1 = asarray(plist1) ;  plist2 = asarray(plist2) ; plist3 = asarray(plist3)
        alist1 = asarray(alist1) ;  alist2 = asarray(alist2) ; alist3 = asarray(alist3)
        nmlist1 = asarray(nmlist1) ;  nmlist2 = asarray(nmlist2) ; nmlist3 = asarray(nmlist3)
        nslist1 = asarray(nslist1) ;  nslist2 = asarray(nslist2) ; nslist3 = asarray(nslist3)

        fout1.flush() ; fout2.flush() ; fout3.flush()
        fout1.close() ; fout2.close() ; fout3.close()
    
    alist1 = abs(alist1) ; alist2 = abs(alist2) ; alist3 = abs(alist3)

    # now, let us make some histograms:
    hist3([plist1, plist2, plist3], r, nbins = 10, vmin = plist3.min(), vmax = plist1.max(), blog = True, xtitle = r'$P / R_{\rm *}$', ytitle = r'${\rm d} N / {\rm d} P$', filename = dirname+'/pdist.png')
    hist3([alist1, alist2, alist3], r, nbins = 10, vmin = alist3.min(), vmax = alist1.max(), blog = True, xtitle = r'$A / R_{\rm *}^2$', ytitle = r'${\rm d} N / {\rm d} A$', filename = dirname+'/adist.png')
    hist3([nmlist1, nmlist2, nmlist3], r, nbins = 10, vmin = minimum(minimum(nmlist1, nmlist2),nmlist3).min(), vmax = maximum(maximum(nmlist1, nmlist2),nmlist3).max(), xtitle = r'$n$', ytitle = r'${\rm d} N / {\rm d} n$', filename = dirname+'/nmdist.png')
    hist3([nslist1, nslist2, nslist3], r, nbins = 10, vmin = minimum(minimum(nslist1, nslist2),nslist3).min(), vmax = maximum(maximum(nslist1, nslist2),nslist3).max(), blog = True, xtitle = r'$\sigma_{\rm n}$', ytitle = r'${\rm d} N / {\rm d} \sigma_{\rm n}$', filename = dirname+'/nsdist.png')
    # perimeter distribution:

# dists(0., 10., nocalc=True)
# dists(pi/4., 10.)
# dists(pi/2., 10.)
# mapping(alpha=pi/3., mayaplot=True, qscale = 10.0, movie = False)
# mapping(surfaceplot=True, qscale = 10.0)
# scaleshow(alpha=pi/3., psi = pi/2.)
qpack = qrand(ifeigen=True)
skymap(alpha = 0., psi=pi/2., qscale=0.0, qsave = qpack, rrange=[3.,10.,30.,100.])
# burntester(200,200)

# ffmpeg -f image2 -r 15 -pattern_type glob -i 'surscale*.png' -pix_fmt yuv420p -b 4096k scales.mp4
