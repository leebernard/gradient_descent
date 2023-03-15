import matplotlib.pyplot as plt

from astropy.io.fits import getdata
from astropy.io import fits
from psf_glmnet import psf_glmnet
from psf_lasso import psf_lasso


# x = getdata('small.fits')
# psf = getdata('small_psf.fits')

x = getdata('k4m_160531_050920_ori.fits.fz', 1)
psf = getdata('mosaic_psf.fits')

mdl = psf_glmnet(x, psf)

lasso_output = psf_lasso(x, psf)

print('Regression finished, displaying results...')

# global figure varables
vmax = x[1120:1220, 1020:1120].max()
vmin = 0
fig, ((rawax, rawax2), (glmnetax, lassoax), (glm_residax, lasso_residax)) = plt.subplots(figsize=(8, 20), nrows=3, ncols=2)
fig.suptitle('Just a arbitrary galaxy')

rawax.set_title('raw_data')
pos = rawax.imshow(x[1135:1195, 1045:1105], cmap='twilight', vmin=vmin, vmax=vmax)
fig.colorbar(pos, ax=rawax)
rawax2.set_title('raw_data')
pos = rawax2.imshow(x[1135:1195, 1045:1105], cmap='twilight', vmin=vmin, vmax=vmax)
fig.colorbar(pos, ax=rawax2)

glmnetax.set_title('glmnet_output')
pos = glmnetax.imshow(mdl[1135:1195, 1045:1105], cmap='twilight', vmin=vmin, vmax=vmax)
fig.colorbar(pos, ax=glmnetax)

lassoax.set_title('lasso_output')
pos = lassoax.imshow(lasso_output[1135:1195, 1045:1105], cmap='twilight', vmin=vmin, vmax=vmax)
fig.colorbar(pos, ax=lassoax)

glm_residax.set_title('residuals_glmnet')
pos = glm_residax.imshow(x[1135:1195, 1045:1105] - mdl[1135:1195, 1045:1105], cmap='twilight', vmin=vmin, vmax=vmax)
fig.colorbar(pos, ax=glm_residax)

lasso_residax.set_title('residuals_lasso')
pos = lasso_residax.imshow(x[1135:1195, 1045:1105] - lasso_output[1135:1195, 1045:1105], cmap='twilight', vmin=vmin, vmax=vmax)
fig.colorbar(pos, ax=lasso_residax)


fullfig, fullax = plt.subplots()
pos = fullax.imshow(lasso_output[1000:2000, 1000:2000], cmap='twilight') #, vmin=vmin, vmax=vmax)
fullfig.colorbar(pos, ax=fullax)

plt.show()



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
