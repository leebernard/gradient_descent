import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import matplotlib
print('Pycharm interactive detected. Chaning to TkAgg')
# Pycharm seems to have problems with the matplotlib interactive background.
# Manually switching to TkAgg seems to fix this
matplotlib.use('TkAgg')

# filename = 'chessboard.png'
# img = cv.imread(filename)
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

im_path = 'image_data/Image__2023-09-01__22-02-26_slit_background.tiff'
from PIL import Image
# slit_im = Image.open(im_path)
slit_im = cv.imread(im_path, cv.IMREAD_ANYDEPTH)
# convert to numpy array
slit_data = np.float32(slit_im)

dst = cv.cornerHarris(slit_data,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
# slit_im[dst>0.01*dst.max()]=[0,0,255]
# cv.imshow('dst',slit_im)
# if cv.waitKey(0) & 0xff == 27:
#  cv.destroyAllWindows()

coords = np.nonzero(dst>0.1*dst.max())
fig, ax = plt.subplots()
ax.imshow(slit_data, cmap=plt.cm.gray)
ax.plot(coords[1], coords[0], color='cyan', marker='o',
        linestyle='None', markersize=6)

plt.show()

