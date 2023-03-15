from psf_glmnet_fun import psf_glmnet_fun
from numpy import array,sqrt,zeros,arange,exp,where,median,log,newaxis
from scipy.signal import fftconvolve

"""
sparse psf image reconstruction (python 3 version using cython)
"""

def psf_glmnet(data,psf,eps=0,alpha=0.99,return_model=True):
    """
    For an input data image (2d array), find an optimal sum of psf's using Glmnet regression:
       data[i,j] = Sum_k,l beta[k,l]*psf[i-k,j-l]

    Inputs:
       data: input data image (2d, numpy array)
       psf: input data point-spread-function (2d, numpy array)
       eps: regularization until lambda_max*eps (default 0 means derive)
       alpha: elasticnet parameter (alpha=1 is LASSO, alpha=0 is ridge regression)
       return_model: returns the model, otherwise the beta parameters (True)
    Outputs:
       The beta model or coeficients (2d, numpy array)
    """
    nx,ny = data.shape
    d0 = median(data)

    # we will circularize the psf (remove square edges) and use only the core of the psf for fitting
    sz=len(psf)
    s1 = sz//2
    s2 = s1//2
    xx = arange(sz)-s1
    h = sqrt( (xx**2)[:,newaxis]+(xx**2)[newaxis,:] ) > s1
    psf1 = 1.*psf
    psf1 -= median(psf1[h])
    psf1[h]=0
    psf1 = psf1.clip(0)
    norm = sqrt((psf1**2).sum())
    psf1/=norm

    # correlation of the data with the psf
    c = fftconvolve(data-d0,psf1[::-1,::-1],mode='same').astype('float32')
    # autocorrelation of the psf
    psf2 = fftconvolve(psf1,psf1[::-1,::-1]).astype('float32')
    sz=len(psf2)

    # the regularization parameters starts at the max observed value
    #  and proceeds to the max times eps
    # to estimate eps (if eps=0), estimate the noise level in the image
    lambda_init = c.max()/alpha
    if (eps<=0):
        delta_x = abs(data-median(data))
        dx = 1.48*median(delta_x[delta_x<2*1.48*median(delta_x)])
        eps = dx/(alpha*lambda_init)

    # step logarithmically through regularization parameters
    psfm = psf2.max()
    psfm1 = psf2[psf2<psfm].max()
    if (psfm1<=0): psfm1 = 0.5*psfm
    nsteps = 1 + int( -log(eps)/log(psfm/psfm1) )
    lambda_ar = zeros(nsteps+1,dtype='float32')
    lambda_ar[:-1] = lambda_init*exp(log(eps)*arange(nsteps)/(nsteps-1))

    beta = zeros((nx,ny),dtype='float32')

    # now, iteratively determine the beta values using Glmnet
    psf_glmnet_fun(lambda_ar,psf2,c,beta,alpha)

    if (return_model):
        return  fftconvolve(beta,psf1,mode='same')
    else:
        # multiply by psf1.sum() for units of flux
        return beta*psf1.sum()
