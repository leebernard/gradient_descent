import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from skimage.draw import polygon_perimeter as draw_polygon

import matplotlib
print('Pycharm interactive session. Changing to TkAgg')
# Pycharm seems to have problems with the matplotlib interactive background.
# Manually switching to TkAgg seems to fix this
matplotlib.use('TkAgg')


im_path = 'image_data/Image__2023-09-01__22-02-26_slit_background.tiff'

with Image.open(im_path) as slit_im:
    # slit_im = Image.open(im_path)
    # convert to numpy array
    slit_data = np.asarray(slit_im)


# subframe = slit_data[1344:1369, 1070:1450]
#
# subframe2 = slit_data[1320:1390, 1085:1432]
# plt.imshow(subframe2)

upper_left = (1344, 1087)
lower_left = (1360, 1087)
upper_right = (1350, 1432)
lower_right = (1367, 1431)

coords = np.asarray([upper_left, upper_right, lower_right, lower_left])
coords_center = coords.mean(axis=0)
print("center of slit:", coords_center)
# center of slit: [1355.25 1259.25]


fig, ax = plt.subplots()
ax.imshow(slit_data, cmap=plt.cm.gray)
# ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o', linestyle='None', markersize=6)
ax.plot(coords[:, 1], coords[:, 0], '+c', markersize=15)

ax.plot(coords_center[1], coords_center[0], '+r', markersize=15)
ax.plot (*draw_polygon(coords[:,1], coords[:,0]),'r-')
# ax.axis((1000, 1500, 1500, 1200))
plt.show()




