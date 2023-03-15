#cython: language_level=3
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def psf_lasso_fun(float[:] lambda_ar, float[:,:,:,::1] psf2, float[:,::1] c, float[:,::1] beta, int skip):
    """ function to carry-out lasso regression """

    cdef Py_ssize_t nsteps = lambda_ar.shape[0], sz = psf2.shape[2], nover = psf2.shape[0], nx = c.shape[0], ny = c.shape[1]
    cdef int k,i1,i1a,i1b,j1,j1a,j1b,imax,jmax,imax1,jmax1,ix0,iy0,ix,iy,ix1,iy1,sz1=sz//2,n2,nx0,ny0
    cdef float dbeta, alam, lam, cmax
  
    n2 = (skip*nover)//2 
    #n2 = (nover//2) + nover*(skip//2)
    nx0=nx//(nover*skip)
    ny0=ny//(nover*skip)

    for k in range(nsteps-1): # loop through regularization parameters, large to small
        lam = lambda_ar[k+1]

        for ix0 in range(nx0): # loop through psf's, one-at-a-time
            ix = n2+ix0*skip*nover
            for iy0 in range(ny0):
                iy = n2+iy0*skip*nover
                if (c[ix,iy]<lam): continue

                # use oversampling to find the best-fit sub-pixel psf
                #  skip>1 treats regular pixels as sub-pixels
                cmax=c[ix,iy]
                imax=n2
                jmax=n2
                if (nover*skip>1):
                    for i1 in range(nover*skip):
                        i1a = ix-n2+i1
                        for j1 in range(nover*skip):
                            j1a = iy-n2+j1
                            if (c[i1a,j1a]>cmax):
                                cmax = c[i1a,j1a]
                                imax=i1
                                jmax=j1

                # otherwise adjust its amplitude beta -> beta - dbeta
                dbeta = beta[ix-n2+imax,iy-n2+jmax]
                alam = cmax+dbeta-lam
                if (alam>0): dbeta -= alam
                beta[ix-n2+imax,iy-n2+jmax] -= dbeta

                i1a = max(0,ix-n2+imax-sz1)
                i1b = min(nx,ix-n2+imax+sz1+1)
                j1a = max(0,iy-n2+jmax-sz1)
                j1b = min(ny,iy-n2+jmax+sz1+1)
                iy1 = iy-n2+jmax+sz1
                imax1 = imax%nover
                jmax1 = jmax%nover

                # correct c[i1,j1] for psf[ix,iy] impact at i1,j1
                for i1 in range(i1a,i1b):
                    ix1 = ix-n2+imax+sz1-i1
                    for j1 in range(j1a,j1b):
                        c[i1,j1] += dbeta*psf2[imax1,jmax1,ix1,iy1-j1]


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def psf_lasso_rebuild(float[:,::1] psf, float[:,::1] beta, float[:,::1] c):
    """ function to rebuild the image from the lasso results """

    cdef Py_ssize_t sz = psf.shape[0], nx = beta.shape[0], ny = beta.shape[1]
    cdef int i,j,i1,i1a,i1b,j1,j1a,j1b,i2,j2,sz1=sz//2
    cdef float b
  
    for i in range(nx):
        for j in range(ny):
            c[i,j]=0

    for i in range(nx):
        for j in range(ny):
            if (beta[i,j]>0):
                b=beta[i,j]

                i1a=max(0,sz1-i)
                i1b=min(sz,nx+sz1-i)
                j1a=max(0,sz1-j)
                j1b=min(sz,ny+sz1-j)

                j2=j-sz1
                for i1 in range(i1a,i1b):
                    i2=i+i1-sz1
                    for j1 in range(j1a,j1b):
                        c[i2,j2+j1] += b*psf[i1,j1]
