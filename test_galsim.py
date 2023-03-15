import matplotlib.pyplot as plt
import os

from astropy.io.fits import getdata
from astropy.io import fits
from psf_glmnet import psf_glmnet
from psf_lasso import psf_lasso


# x = getdata('small.fits')
# psf = getdata('small_psf.fits')
file_name = os.path.join('output', 'demo3.fits')
file_name_epsf = os.path.join('output','demo3_epsf.fits')

x = getdata(file_name)
psf = getdata(file_name_epsf)

mdl = psf_glmnet(x, psf)

lasso_output = psf_lasso(x, psf)

print('Regression finished, displaying results...')

# global figure varables
vmax = x.max()
vmin = 0
fig, (rawax, lassoax, lasso_residax) = plt.subplots(figsize=(6, 15), nrows=3)
fig.suptitle('Galsim Demo 3 galaxy')

rawax.set_title('raw_data')
pos = rawax.imshow(x, cmap='twilight') #, vmin=vmin, vmax=vmax)
fig.colorbar(pos, ax=rawax)

lassoax.set_title('lasso_output')
pos = lassoax.imshow(lasso_output, cmap='twilight') #, vmin=vmin, vmax=vmax)
fig.colorbar(pos, ax=lassoax)

lasso_residax.set_title('residuals_lasso')
pos = lasso_residax.imshow(x - lasso_output, cmap='twilight') #, vmin=vmin, vmax=vmax)
fig.colorbar(pos, ax=lasso_residax)


fullfig, fullax = plt.subplots()
pos = fullax.imshow(lasso_output, cmap='twilight') #, vmin=vmin, vmax=vmax)
fullfig.colorbar(pos, ax=fullax)

file_name = os.path.join('output', 'galsim_halfsize2_results.png')
fig.savefig(file_name)
# plt.show()



# test section
'''
mosaic_data = fits.open('k4m_160531_050920_ori.fits.fz')

region_def = ((1080, 928), (1112, 960))

psf_data = x[934:952, 1094:1112]

plt.imshow(psf_data, cmap='twilight')

plt.imshow(psf, cmap='twilight')

psf_hdu = fits.open('small_psf.fits')[0]

mosaic_data[1].header

mosaic_psf = mosaic_data[1]
mosaic_psf.data = psf_data
plt.imshow(mosaic_psf.data, cmap='twilight')
mosaic_psf.header['comment'] = 'Sliced for characterizing the psf'
mosaic_psf.header['comment'] = 'slice taken from k4m_160531_050920_ori[im1]'

mosaic_psf.writeto('mosaic_psf.fits')
'''
