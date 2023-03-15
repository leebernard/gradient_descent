#!/usr/bin/python3
"""
psf_lasso.py fits_file psf_file [nomodel] [sigma]
    sparse psf image reconstruction (python 3 version using cython)
"""

from psf_lasso_fun import psf_lasso_fun,psf_lasso_rebuild
from numpy import array,sqrt,zeros,arange,exp,where,median,log,newaxis,hstack,roll,ceil
from scipy.special import sinc
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter

import sys,os
from astropy.io.fits import getdata,getheader,writeto

def usage():
    print (__doc__)
    sys.exit()

def oversamp_psf(psf,nover):
    """
     Lanczos-4 interpolation to shift PSF for oversampling
    """
    if (nover<1): nover=1
    s=len(psf)
    inover = int(ceil(nover))
    if (nover!=inover):
        if (inover%2==0): inover+=1
    psf_out = zeros((inover*s,inover*s),dtype='float32')
    n2 = inover//2

    pp0 = sqrt((psf**2).sum())

    xx1 = arange(8)-3.
    xx2 = arange(8)-4.
    def f(a):
        if (a>=0): return hstack((0,sinc((a-xx1))*sinc((a-xx1)/4.)))
        else: return hstack((sinc((a-xx2))*sinc((a-xx2)/4.),0))

    for i in range(inover):
        dx = (n2-i)/nover
        for j in range(inover):
            dy = (n2-j)/nover
            im = f(dx)[:,newaxis]*f(dy)[newaxis,:]
            if (dx==0 and dy==0): psf_out[i::inover,j::inover] = psf
            else:
                pp = fftconvolve(psf,im,mode='same')
                psf_out[i::inover,j::inover] = pp*pp0/sqrt((pp**2).sum())

    return psf_out


def psf_lasso(data,psf,eps=0,return_model=True,skip=0,nover=0):
    """
    For an input data image (2d array), find an optimal sum of psf's using Lasso regression:
       data[i,j] = Sum_k,l beta[k,l]*psf[i-k,j-l]

    Inputs:
       data: input data image (2d, numpy array)
       psf: input data point-spread-function (2d, numpy array)
       eps: regularization until lambda_max*eps (default 0 means derive)
       return_model: returns the model, otherwise the beta parameters (True)
       skip: evaluate the model on a skipped grid (default 0, estimate)
       nover: oversampling (default 0, estimate)
    Outputs:
       The beta model or coeficients (2d, numpy array)
    """
    nx,ny = data.shape
    d0 = median(data)

    # we will circularize the psf (remove square edges) and use only the core of the psf for fitting
    sz=len(psf)
    sz1 = sz//2
    xx = arange(sz)-sz1
    h = sqrt( (xx**2)[:,newaxis]+(xx**2)[newaxis,:] ) > sz1
    psf1 = (1.*psf).astype('float32')
    psf1 -= median(psf1[h])
    psf1[h]=0
    norm = sqrt((psf1**2).sum())
    psf1/=norm

    psf10 = 1.*psf1
    # automatic oversampling estimate varies between 1 and 9 (sub-pixels), targetting 1% precision
    if (nover<1): nover = int( sqrt( log( (psf10[1:]*psf10[:-1]).sum().clip(0.4) )/log(0.99) ) )
    if (nover<1): nover=1
    n2 = nover//2

    if (nover>1): psf1 = oversamp_psf(psf1,nover)

    # correlation of the data with the psf
    if (nover==1): c = fftconvolve(data-d0,psf1[::-1,::-1],mode='same').astype('float32')
    else:
        data1 = zeros((nx*nover,ny*nover),dtype='float32')
        data1[n2::nover,n2::nover] = data-d0
        c = fftconvolve(data1,psf1[::-1,::-1],mode='same').astype('float32')
        data1 = 0

    # autocorrelation of the psf
    sz1 = 2*len(psf1)-1
    psf2 = zeros((nover,nover,sz1,sz1),dtype='float32')
    if (nover==1): psf2[0,0] = fftconvolve(psf1,psf1[::-1,::-1]).astype('float32')
    else:
        for i in range(nover):
            for j in range(nover):
                psfij0 = 0*psf1
                psfij0[i::nover,j::nover] = psf1[i::nover,j::nover]
                psf2[i,j] = fftconvolve(psfij0,psf1[::-1,::-1])

    # the regularization parameters starts at the max observed value
    #  and proceeds to the max times eps
    # to estimate eps (if eps=0), estimate the noise level in the image
    lambda_init = c.max()
    if (eps<=0):
        delta_x = abs(data-median(data))
        h = delta_x<2*1.48*median(delta_x)
        if (h.sum()>0): dx = 1.48*median(delta_x[delta_x<2*1.48*median(delta_x)])
        else: dx=0.
        eps = dx/lambda_init
        if (eps<=0): eps=1.e-3

    # automatic estimate of skip size set to minimize psf grid autocorrelation
    if (skip<=0):
        skip=1
        for skip in range(1,sz//2):
            if ((psf10[skip:]*psf10[:-skip]).sum()<0.5): break

    # step logarithmically through regularization parameters
    psfm = 0.5*( (psf10[1:]*psf10[:-1]).sum() + (psf10[skip:]*psf10[:-skip]).sum() )
    if (psfm<=0.5): psfm = 0.5
    nsteps = 1 + int( log(eps)/log(psfm) )
    lambda_ar = zeros(nsteps+1,dtype='float32')
    lambda_ar[:-1] = lambda_init*exp(log(eps)*arange(nsteps)/(nsteps-1))

    # now, iteratively determine the beta values using Lasso
    beta = zeros((nx*nover,ny*nover),dtype='float32')
    psf_lasso_fun(lambda_ar,psf2,c,beta,skip)

    if (return_model):
        psf_lasso_rebuild(psf1,beta,c) #same as c=fftconvolve(beta,psf1,mode='same')
        if (nover==2):
            return roll(roll(c[n2-1::nover,n2-1::nover],-1,axis=0),-1,axis=1)
        else:
            return c[n2::nover,n2::nover]
    else:
        # multiply by psf.sum() for units of flux
        return beta*psf10.sum()


if __name__ == "__main__":
    """
    """
    if (len(sys.argv)<3): usage()

    file=sys.argv[1]
    if (os.path.exists(file)==False): 
        print ("cannot find file:",file)
        usage()
    psf_file=sys.argv[2]
    if (os.path.exists(psf_file)==False): 
        print ("cannot find file:",psf_file)
        usage()

    return_model=True
    if (len(sys.argv)>3):
        if (sys.argv[3]=="nomodel"): return_model=False

    sigma=0.
    if (len(sys.argv)>4):
        sigma = float(sys.argv[4])
    
    data = getdata(file)
    hdr = getheader(file)
    psf = getdata(psf_file)
    model = psf_lasso(data,psf,return_model=return_model)

    if (return_model==False):
        nover = len(model)//len(data)
        if (sigma>0): model = gaussian_filter(model,sigma*nover)
        try:
            hdr['CRPIX1']*=nover
            hdr['CRPIX2']*=nover
            hdr['CD1_1']/=nover
            hdr['CD1_2']/=nover
            hdr['CD2_1']/=nover
            hdr['CD2_2']/=nover
        except:
            print ("Unable to read input file WCS, ignoring...")

    writeto('lasso_'+file,model,hdr,overwrite=True)
