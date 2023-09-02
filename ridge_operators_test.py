import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian


# create some test data
im_path = 'image_data/Image__2023-09-01__22-02-26_slit_background.tiff'
from PIL import Image
slit_im = Image.open(im_path)
# convert to numpy array
slit_data = np.asarray(slit_im)
# crop the top
# slit_data = slit_data[150:, 500:1500]

# plt.imshow(slit_data)
# end create test data


def original(image, **kwargs):
    """Return the original image, ignoring any kwargs."""
    return image

image = np.asarray(slit_im)
cmap = plt.cm.gray

plt.rcParams["axes.titlesize"] = "medium"
axes = plt.figure(figsize=(10, 4)).subplots(2, 3)
for i, black_ridges in enumerate([True, False]):
    for j, (func, sigmas) in enumerate([
            (original, None),
            # (meijering, [1]),
            # (meijering, range(1, 5)),
            # (sato, [1]),
            # (sato, range(1, 5)),
            # (frangi, [1]),
            # (frangi, range(1, 5)),
            (hessian, [1]),
            (hessian, range(1, 5)),
    ]):
        result = func(image, black_ridges=black_ridges, sigmas=sigmas)
        axes[i, j].imshow(result, cmap=cmap)
        if i == 0:
            title = func.__name__
            if sigmas:
                title += f"\n\N{GREEK SMALL LETTER SIGMA} = {list(sigmas)}"
            axes[i, j].set_title(title)
        if j == 0:
            axes[i, j].set_ylabel(f'{black_ridges = }')
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.tight_layout()
plt.show()


