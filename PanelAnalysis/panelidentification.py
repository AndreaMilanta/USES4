"""module devoted to image preprocessing and panel identification
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

import spimage as spi
import spconstants as spc

curr_image = spc.PATH_IMAGES_NICC + spc.FN_IMAGE
# curr_image = spc.PATH_IMAGES + spc.CF_IMAGE

# DISPLAYING OPTI
cntDefinition = 0.001    # Approximation level of contours (ignored if viewBox=True)
cntColor = '#FFFFFF'    # Color of contours
cntThick = 2            # Thickness of contours
cntMinArea = 4000       # Minimum Area of contour
linColor = '#0000FF'    # Color of lines
linThick = 1            # Thickness of lines
linMinLen = 800         # Minimum length of lines

# LOAD IMAGES
origimg = spi.spimage(cv2.imread(curr_image, cv2.IMREAD_COLOR))

# LINE COMPUTATION
classimg = origimg.toBW()
classimg.getCanny(50, 150, apSize=3, inplace=True)
lines = classimg.getHoughLines(1, spc.deg2rad(1), 200)
# origimg.show(view=True)

# ROTATION
lin = np.reshape(lines, (-1, 2))
avg_deg = np.average(lin, axis=0)[1]
origimg.rotate(spc.rad2deg(avg_deg), 'ccw', inplace=True)

# COLOR TUNING
img = origimg.enhanceColor('b', 1)
img.enhanceColor('g', 1, inplace=True)
img.enhanceColor('r', 0, inplace=True)
kernel_1 = np.ones((10, 10), np.uint8)
kernel_2 = np.ones((3, 3), np.uint8)
# img.filterColor('b', 1, True, inplace=True, subcolor=(255, 255, 255))
img.filterColor('g', 100, True, inplace=True)

# HISTOGRAM SINGLE AREA REPRESENTATION
# a_noise = spi.histoArea([500, 0], [50, 50], None)
# a_panel = spi.histoArea([300, 100], [50, 50], None)
# noise = img.getAreaHist(a_noise, 'g')
# panel = img.getAreaHist(a_panel, 'g')
# plt.figure(100)
# plt.title('Histograms')
# plt.plot(noise, color='#00FF00', label='Panel')    # , bins=255, label='Panel', rwidth=None)
# plt.plot(panel, color='#0000FF')    # , bins=255, label='Panel', rwidth=None)
# plt.xlim([0, 256])
# plt.show()

# HISTOGRAM CLASSIFICATION
m = 'euclidean'
cp = origimg.copy()

# 1 Iteration
perc = 1   # percentage of area to be black for exclusione
ksize = [5, 5]
thresh = ksize[0] * ksize[1] * perc / 100
kstep = ksize
# kstep = [int(ksize[0] / 2), int(ksize[1] / 2)]
try:
    areas, _ = img.histoClassification('g', ksize, kstep, dst_metric=m, numCluster=3, showDendogram=False, view=False)
except Exception as e:
    print('Computation with metric: ' + m + ' returned the following error: ' + str(e))
    exit()
for i, a in enumerate(areas):
    # print("Histogram " + str(i) + ": pos " + str(a.anchor))
    # print(str(a.hist))
    if a.hist[0] > thresh:
        cp.colorAreas([a], '#000000')

# 2 Iteration
perc = 30   # percentage of area to be black for exclusion
ksize = [cp.img.shape[1], 1]
thresh = ksize[0] * ksize[1] * perc / 100
kstep = ksize
try:
    areas, _ = cp.histoClassification('g', ksize, kstep, dst_metric=m, numCluster=3, showDendogram=False, view=False)
except Exception as e:
    print('Computation with metric: ' + m + ' returned the following error: ' + str(e))
    exit()
for i, a in enumerate(areas):
    # print("Histogram " + str(i) + ": pos " + str(a.anchor))
    # print(str(a.hist))
    if a.hist[0] > thresh:
        origimg.colorAreas([a], '#000000')

# RECOMPUTE LINES (Horizontal)
classimg = origimg.toBW()
classimg.getCanny(0, 50, apSize=3, inplace=True)
classimg.show(True)
exit()
lines = classimg.getHoughLines(1, spc.deg2rad(1), 100)
for l in lines:
    origimg.addLine(l)
origimg.show(True)


exit()

# NOISE REDUCTION
img.toBW(inplace=True)
dst = img.toBW()
dst = img.fastMeanDenoising(4)

lap = dst.laplacian(ksize=17, inplace=False)
lap.applyClahe(clipLimit=4, tileGridSize=(100, 100), inplace=True)
lap.getCanny(100, 100, inplace=True)
lineimg = origimg.copy()
# lap.show(view=True)

# GET LINES
lines = lap.getHoughLines(1, np.pi / 180, linMinLen)
for l in lines:
    lineimg.addLine(l, thick=linThick, color=linColor)

# GET CONTOURS
# lap.applyClahe(clipLimit=4, tileGridSize=(100, 100), inplace=True)
# thresh = lap.adaptiveThreshold('g', cv2.THRESH_BINARY, 111, 2)

# contours, hierarchy = thresh.getContours(cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE, filterfunc=lambda x: cv2.contourArea(x) > cntMinArea)
# print(str(len(contours)) + ' contours found with area larger than ' + str(cntMinArea))

# Add contours to original image
# for c in contours:
#     lineimg.addContour(c, cntDefinition, boxMode=False, color='#FF0000')

# lineimg.addContour(contours[0], cntDefinition, boxMode=False, thick=1, color='#FF0000')
# lineimg.addContour(contours[1], cntDefinition, boxMode=False, thick=1, color='#00FF00')
# lineimg.addContour(contours[2], cntDefinition, boxMode=False, thick=1, color='#0000FF')
# lineimg.addContour(contours[3], cntDefinition, boxMode=False, thick=1, color='#FFFFFF')

# Final Image Visualization
lineimg.show(view=True)
