"""
Below is the way I centroided in undergrad. It has severe limitations: it relies on least-squares fitting, it can only
find one centroid, and the data needs to be a postage stamp. Downstream from those limitations, the data has to be
fairly clean (background subtracted, etc), it needs error estimates on the pixels, and the rough area of the centroid
needs to be manually found for both the postage stamp coordinates and the initial guess.

Upgrades: enable multiplexing. Fit the model with a proper Nested Sampling algorthm, or lasso regression. Lasso
regression is probably the fasted way; it's photometric data loss doesn't matter for centroiding.

Nat's lasso regression using numerical data to build a psf model. Essentially, he takes all the point sources in the
image, flux-normalizes them, then stacks and averages them. My goal is slightly different from Nat's. Instead of
extracting data, I want to use the data to create a vector between two images, and drive that vector to zero. This is
presumably simular to telescope pointing.

Given that, I should force the algorthm to have only one centroid, which means lasso regression isn't the correct path.
I also need to throw an error if the centroid fit doesn't have a clear solution, which means I need to define a criteria
for a clear solution.
"""

import numpy as np
import matplotlib.pyplot as plt

# from scipy.optimize import curve_fit
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from skimage.draw import line as draw_line
from matplotlib import cm


def draw_box(imdata, c1, c2, c3, c4, mag=255):
    """
    Draws an arbitrary box, using four pixel coordinate pairs.
    Treats c1 as the origin, and then draws the box through c2, c3, and c4, in that order
    Parameters
    ----------
    c1
    c2
    c3
    c4

    Returns
    -------
    Image data with a box added to it.
    """
    imdata[draw_line(*c1, *c2)] = mag
    imdata[draw_line(*c2, *c3)] = mag
    imdata[draw_line(*c3, *c4)] = mag
    imdata[draw_line(*c4, *c1)] = mag


def draw_slit(imdata, origin, length, angle, aspect_ratio=30, mag=255, debug=False):
    # define the corners
    c1 = origin  # set first corner

    # walk to 2nd corner
    origin = np.rint((length*np.sin(-angle) + origin[0], length*np.cos(-angle) + origin[1])).astype(int)
    c2 = origin  # set second corner

    # walk to 3rd corner
    angle += np.pi / 2  # rotate 90 degrees
    origin = np.rint((length/aspect_ratio*np.sin(-angle) + origin[0], length/aspect_ratio*np.cos(-angle) + origin[1])).astype(int)
    c3 = origin  # set third corner

    # walk to the 4th corner
    angle += np.pi / 2  # rotate 90 degrees
    origin = np.rint((length*np.sin(-angle) + origin[0], length*np.cos(-angle) + origin[1])).astype(int)
    c4 = origin  # set 4th corner

    # test
    if debug:
        angle += np.pi / 2  # rotate 90 degrees
        origin = np.rint((length/aspect_ratio*np.sin(-angle) + origin[0], length/aspect_ratio*np.cos(-angle) + origin[1])).astype(int)
        print('starting location:', c1)
        print('Final location:', origin)
        print('Sucessful walk?', origin == c1)

    draw_box(imdata, c1, c2, c3, c4, mag=mag)

# test data:
origin, length, angle = (1202, 800), 350, np.radians(-2)


# Constructing test image
fake_slit = np.zeros((2048, 2448))
# idx = np.arange(25, 175)
# image[idx, idx] = 255
# image[draw_line(45, 25, 25, 175)] = 255
# image[draw_line(25, 135, 175, 155)] = 255

# corners = ((45, 45), (25, 175), (175, 155), (150, 35))
# draw_box(image, *corners)
draw_slit(fake_slit, origin, length, angle)
# end create test data

# # create some test data
# im_path = 'image_data/slit_nofilter_screenshot.png'
# from PIL import Image
# slit_im = Image.open(im_path)
# # convert to numpy array
# slit_data = np.asarray(slit_im.convert('L'))
# # crop the top
# slit_data = slit_data[150:, 500:1500]
#
# plt.imshow(slit_data)


# Classic straight-line Hough transform
# Set a precision of 0.5 degree.
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
h, theta, d = hough_line(fake_slit, theta=tested_angles)

# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(fake_slit, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

angle_step = 0.5 * np.diff(theta).mean()
d_step = 0.5 * np.diff(d).mean()
bounds = [np.rad2deg(theta[0] - angle_step),
          np.rad2deg(theta[-1] + angle_step),
          d[-1] + d_step, d[0] - d_step]
ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(fake_slit, cmap=cm.gray)
ax[2].set_ylim((fake_slit.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=None, num_peaks=10)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))

plt.tight_layout()
plt.show()


# try again, but with some filtering
edges = canny(fake_slit, 2, 1, 25)
# edges = canny(slit_data, sigma=1.0, low_threshold=10, high_threshold=20, use_quantiles=False)
# plt.imshow(edges)

lines = probabilistic_hough_line(edges, threshold=10, line_length=3, line_gap=10)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(fake_slit, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, fake_slit.shape[1]))
ax[2].set_ylim((fake_slit.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()

'''
old code below
'''
#
# def flat_Gaussian_2d(indata, amplitude, x0, y0, sigma_x, sigma_y, offset):
#     """Define gaussian function, assuming no correlation between x and y.
#
#     Uses a flattened input, and gives a flattened output
#
#     Parameters
#     ----------
#     indata: array int
#         indata is a pair of arrays, each array corresponding to the x indice or y indice, in the form (x, y)
#     amplitude: float
#         represents the total flux of the object being fitted
#     x0: float
#         horizontal center of the object
#     y0: float
#         vertical center of the object
#     sigma_x: float
#         half width half maximum of the object along the horizontal
#     sigma_y: float
#         half width half maximum of the object along the vertical
#     offset: float
#         represents the background around the object
#     """
#     import numpy as np
#     x, y = indata
#     normalize = 1 / (sigma_x * sigma_y * 2 * np.pi)
#
#     gaussian_fun = offset + amplitude * normalize * np.exp(
#         -(x - x0) ** 2 / (2 * sigma_x ** 2) - (y - y0) ** 2 / (2 * sigma_y ** 2))
#
#     return gaussian_fun.ravel()
#
#
# def Gaussian_2d(indata, amplitude, x0, y0, sigma_x, sigma_y, offset):
#     """same as Guassian_2D, but does not flatten the result.
#
#     This function is used for producing a 2d array of the result from the fit
#
#     Parameters
#     ----------
#     indata: array int
#         indata is a pair of arrays, each array corresponding to the x indice or y indice, in the form (x, y)
#     amplitude: float
#         represents the total flux of the object being fitted
#     x0: float
#         horizontal center of the object
#     y0: float
#         vertical center of the object
#     sigma_x: float
#         half width half maximum of the object along the horizontal
#     sigma_y: float
#         half width half maximum of the object along the vertical
#     offset: float
#         represents the background around the object
#     """
#     import numpy as np
#     x, y = indata
#     normalize = 1 / (sigma_x * sigma_y * 2 * np.pi)
#
#     gaussian_fun = offset + amplitude * normalize * np.exp(
#         -(x - x0) ** 2 / (2 * sigma_x ** 2) - (y - y0) ** 2 / (2 * sigma_y ** 2))
#
#     return gaussian_fun
#
#
# # curve fit
# try:
#     m_fit, m_cov = curve_fit(flat_Gaussian_2d(), (x, y), aperture.ravel(), sigma=aperture_err.ravel(), p0=guess,
#                              bounds=bounds)
#
# except RuntimeError:
#     print('Unable to find fit.')
# else:
#
#     error = np.sqrt(np.diag(m_cov))
#     # print('Error on parameters')
#     # print(error)
#
#
#     x_center = m_fit[1]
#     y_center = m_fit[2]
#     x_width = m_fit[4]
#     y_width = m_fit[5]
