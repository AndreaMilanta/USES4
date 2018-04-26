"""module devoted to image preprocessing and panel identification
"""
import cv2
import numpy as np

import spimage as spi
import spconstants as spc

curr_image = spc.PATH_IMAGES + spc.FN_IMAGE

p1 = [137, 42]  # [185, 242]
p2 = [92, 243]  # [490, 820]
p3 = [605, 240]  # [974, 603]
p4 = [606, 43]  # [711, 56]
p1_s = [0, 0]
p2_s = [0, 205]  # [0, 660]
p3_s = [469, 205]  # [530, 660]
p4_s = [469, 0]  # [530, 0]

# Define input and output triangles
rect_in = np.float32([[p1, p2, p3, p4]])
rect_out = np.float32([[p1_s, p2_s, p3_s, p4_s]])

# perMat = cv2.getPerspectiveTransform(rect_in, rect_out)

origimg = spi.spimage(cv2.imread(curr_image, cv2.IMREAD_COLOR))
img = origimg.enhanceColor('b', 1)
img.enhanceColor('g', 1, inplace=True)
img.enhanceColor('r', 0, inplace=True)
# img2 = spi.spimage(cv2.blur(img.img, (7, 7)))
# img2 = spi.spimage(cv2.GaussianBlur(img.img, (7, 7), sigmaX=5, sigmaY=5))
# img.filterColor('b', 70, True, inplace=True, subcolor=(255, 255, 255))
# img.filterColor('g', 50, True, inplace=True)
img2 = img.toBW()
img2.show()

thresh = cv2.adaptiveThreshold(img2.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 111, 3)
# ret, thresh = cv2.threshold(img2.img, 132, 255, cv2.THRESH_BINARY)
spi.spimage(thresh).show()

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(hierarchy)
conts = list(filter(lambda x: cv2.contourArea(x) > 5000, contours))
print(str(len(conts)) + ' contours found with area larger than 10000')
# cv2.drawContours(origimg.img, conts, 5, (0, 0, 0), 3)
for c in conts:
    # origimg.addPolygon(cv2.boxPoints(cv2.minAreaRect(c)))
    epsilon = 0.001 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    origimg.addPolygon(approx[:, 0], thick=1)
origimg.show(view=True)
# img2.show(view=True)
# plt.show()
