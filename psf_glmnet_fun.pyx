cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

def psf_glmnet_fun(float[:] lambda_ar, float[:,::1] psf2, float[:,::1] c, float[:,::1] beta, float alpha):
    """ function to carry-out glmnet """
    # float[:]  1d pointer
    # float[:, :] 2d pointer
    # float[:, ::1] 2d pointer to C-contiguous memory

    cdef Py_ssize_t nsteps = lambda_ar.shape[0], sz = psf2.shape[0], nx = c.shape[0], ny = c.shape[1]
    cdef int k,i1,i1a,i1b,j1,j1a,j1b,ix,iy,ix1,iy1,sz1=sz//2
    cdef float dbeta, alam, lam, lam1i, thresh

    for k in range(nsteps-1): # loop through regularization parameters, large to small
        lam = lambda_ar[k+1]
        lam1i = 1./(1 + (1-alpha)*lam)
        thresh = alpha*lam

        for ix in range(nx): # loop through psf's, one-at-a-time
            i1a = max(0,ix-sz1)
            i1b = min(nx,ix+sz1+1)
            for iy in range(ny):
                if (c[ix,iy]<thresh): continue

                # otherwise adjust it's amplitude beta
                dbeta = beta[ix,iy]
                alam = thresh-c[ix,iy]-dbeta
                if (alam<0): dbeta += alam*lam1i
                beta[ix,iy] -= dbeta

                j1a = max(0,iy-sz1)
                j1b = min(ny,iy+sz1+1)
                iy1 = iy+sz1

                # correct c[i1,j1] for psf[ix,iy] impact at i1,j1
                for i1 in range(i1a,i1b):
                    ix1 = ix+sz1-i1
                    for j1 in range(j1a,j1b):
                        c[i1,j1] += dbeta*psf2[ix1,iy1-j1]
