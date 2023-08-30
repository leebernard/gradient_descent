import numpy as np

from scipy.optimize import curve_fit

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
def flat_Gaussian_2d(indata, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """Define gaussian function, assuming no correlation between x and y.

    Uses a flattened input, and gives a flattened output

    Parameters
    ----------
    indata: array int
        indata is a pair of arrays, each array corresponding to the x indice or y indice, in the form (x, y)
    amplitude: float
        represents the total flux of the object being fitted
    x0: float
        horizontal center of the object
    y0: float
        vertical center of the object
    sigma_x: float
        half width half maximum of the object along the horizontal
    sigma_y: float
        half width half maximum of the object along the vertical
    offset: float
        represents the background around the object
    """
    import numpy as np
    x, y = indata
    normalize = 1 / (sigma_x * sigma_y * 2 * np.pi)

    gaussian_fun = offset + amplitude * normalize * np.exp(
        -(x - x0) ** 2 / (2 * sigma_x ** 2) - (y - y0) ** 2 / (2 * sigma_y ** 2))

    return gaussian_fun.ravel()


def Gaussian_2d(indata, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """same as Guassian_2D, but does not flatten the result.

    This function is used for producing a 2d array of the result from the fit

    Parameters
    ----------
    indata: array int
        indata is a pair of arrays, each array corresponding to the x indice or y indice, in the form (x, y)
    amplitude: float
        represents the total flux of the object being fitted
    x0: float
        horizontal center of the object
    y0: float
        vertical center of the object
    sigma_x: float
        half width half maximum of the object along the horizontal
    sigma_y: float
        half width half maximum of the object along the vertical
    offset: float
        represents the background around the object
    """
    import numpy as np
    x, y = indata
    normalize = 1 / (sigma_x * sigma_y * 2 * np.pi)

    gaussian_fun = offset + amplitude * normalize * np.exp(
        -(x - x0) ** 2 / (2 * sigma_x ** 2) - (y - y0) ** 2 / (2 * sigma_y ** 2))

    return gaussian_fun


# curve fit
try:
    m_fit, m_cov = curve_fit(flat_Gaussian_2d(), (x, y), aperture.ravel(), sigma=aperture_err.ravel(), p0=guess,
                             bounds=bounds)

except RuntimeError:
    print('Unable to find fit.')
else:

    error = np.sqrt(np.diag(m_cov))
    # print('Error on parameters')
    # print(error)

    # save the results
    background_results.append([background_value, background_dev])
    aperture_data.append(aperture)
    fit_results.append(m_fit)
    fit_cov.append(m_cov)

    x_center = m_fit[1]
    y_center = m_fit[2]
    x_width = m_fit[4]
    y_width = m_fit[5]
