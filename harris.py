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
slit_data = np.float32(slit_im)[1320:1390, 1070:1450]

# cornerHarris(src, blockSize, ksize, k[, dst[, borderType]]
# dst = cv.cornerHarris(slit_data,2,3,0.04)
# result is dilated for marking the corners, not important
# dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
# slit_im[dst>0.01*dst.max()]=[0,0,255]
# cv.imshow('dst',slit_im)
# if cv.waitKey(0) & 0xff == 27:
#  cv.destroyAllWindows()


# 	src, d, sigmaColor, sigmaSpace[, dst[, borderType]]
blur = cv.bilateralFilter(slit_data,9,6000,25)
plt.subplot(121),plt.imshow(slit_data,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur,cmap = 'gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.tight_layout()

# 	image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]]
corners = cv.goodFeaturesToTrack(blur,25,0.01,10, blockSize=10)
corners = np.intp(corners)  # convert to index integers (intp)

# coords = np.nonzero(dst>0.1*dst.max())
fig, ax = plt.subplots()
ax.imshow(slit_data, cmap=plt.cm.gray)
ax.plot(*corners.transpose(), color='cyan', marker='o',
        linestyle='None', markersize=6)

plt.show()

