"""class to handle images
"""
from matplotlib import pyplot as plt
from matplotlib import patches as ptc
import cv2
import numpy as np
import copy

import spexceptions as spe
import csptools as csp

index = 0   # incremental index of images


class spimage:
    def __init__(self, image, iscv2=True):
        """Initialize class

        Arguments
            iscv2 {Boolean} -- checks if the image follows the cv2 convention (BGR) or the pyplot (RGB)
        """
        global index
        # Check image validity
        if image is None:
            raise spe.ImageNotFoundException("The specified image does not exists")
        # Check color format
        if len(image.shape) == 2:
            self.bw = True
        else:
            self.bw = False
        if not iscv2 and not self.bw:
            self.img = image[:, :, ::+1]
        else:
            self.img = image

        # Enhance contrast
        self.array_alpha = np.array([1.25])
        self.array_beta = np.array([-10.0])
        # add a beta value to every pixel
        cv2.add(self.img, self.array_beta, self.img)
        # multiply every pixel value by alpha
        cv2.multiply(self.img, self.array_alpha, self.img)

        # assign ID
        self.id = index
        index += 1
        # initiliatize variables
        self.points = []
        self.rects = []
        self.polys = []

    def toBW(self, inplace=False):
        """transform the image to black and white+

        Keyword Arguments:
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if inplace and not self.bw:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.bw = True
        elif self.bw:
            return spimage(copy.deepcopy(self.img))
        else:
            return spimage(cv2.cvtColor(copy.deepcopy(self.img), cv2.COLOR_BGR2GRAY))

    def addPoint(self, point, rad=5, color="#FFFFFF"):
        """add point to be drawn on image
        """
        if len(point) != 4:
            point = [point[0], point[1], rad, color]
        self.points.append(point)

    def addRectangle(self, rect, thick=2, color="#FFFFFF"):
        """add rectangle to be drawn on image
        """
        rct = [rect[0], rect[1], rect[2], rect[3], thick, color]
        self.rects.append(rct)

    def addPolygon(self, plg, thick=2, color="#FFFFFF"):
        """add polygon to be drawn on image
        """
        poly = [plg, thick, color]
        self.polys.append(poly)

    def show(self, view=False):
        """Gets image ready for display
        if view=True also displais it by calling plt.show()
        """
        plt.figure(self.id)
        ax = plt.gca()
        if not self.bw:
            plt.imshow(self.img[:, :, ::-1])
        else:
            plt.imshow(self.img, cmap="gray")
        for p in self.points:
            c = plt.Circle((p[0], p[1]), radius=p[2], color=p[3])
            ax.add_patch(c)
        for r in self.rects:
            rct = plt.Rectangle((r[0], r[1]), r[2], r[3], linewidth=r[4], edgecolor=r[5], facecolor='None')
            ax.add_patch(rct)
        for p in self.polys:
            plg = plt.Polygon(p[0], linewidth=p[1], edgecolor=p[2], facecolor='None')
            ax.add_patch(plg)
        plt.draw()
        if view:
            plt.show()

    def warpPerspective(self, mat, size, inplace=False):
        """Apply perspective transformation

        Arguments:
            mat {np.array[3, 3]} -- transformation matrix
            size {[int, int]} -- size of output image

        Keyword Arguments:
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if inplace:
            self.img = cv2.warpPerspective(self.img, mat, size)
        else:
            return spimage(cv2.warpPerspective(self.img, mat, size))

    def getCanny(self, minval, maxval, inplace=False):
        """Get canny edges

        Arguments:
            minval {float} -- minimum value for edge detection (below is not an edge)
            maxval {float} -- maximum value for edge detection (above is an edge)

        Keyword Arguments:
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if inplace:
            self.img = cv2.Canny(self.img, minval, maxval)
            self.bw = True
        else:
            return spimage(cv2.Canny(self.img, minval, maxval))

    def enhanceColor(self, color, value, inplace=False):
        """Enhances a particular color (R,G,B) on the image

        Arguments:
            color {string} -- color to enhance (R,r,Red,red)(G,g,Green,green)(B,b,Blue,blue)
            value {float} -- enhancing factor

        Keyword Arguments:
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if self.bw:
            return
        color.lower()
        if color == 'g' or color == 'green':
            if inplace:
                self.img[:, :, 1] = self.img[:, :, 1] * value
                self.img.astype(int)
            else:
                en_img = copy.deepcopy(self.img)
                en_img[:, :, 1] = en_img[:, :, 1] * value
                en_img.astype(int)
                return spimage(en_img)
        if color == 'b' or color == 'blue':
            if inplace:
                self.img[:, :, 0] = self.img[:, :, 0] * value
                self.img.astype(int)
            else:
                en_img = copy.deepcopy(self.img)
                en_img[:, :, 0] = en_img[:, :, 0] * value
                en_img.astype(int)
                return spimage(en_img)
        if color == 'r' or color == 'red':
            if inplace:
                self.img[:, :, 2] = self.img[:, :, 2] * value
                self.img.astype(int)
            else:
                en_img = copy.deepcopy(self.img)
                en_img[:, :, 2] = en_img[:, :, 2] * value
                en_img.astype(int)
                return spimage(en_img)
        return

    def filterColor(self, color, value, below, subcolor=[0, 0, 0], inplace=False):
        """Enhances a particular color (R,G,B) on the image

        Arguments:
            color {string} -- color to filter (R,r,Red,red)(G,g,Green,green)(B,b,Blue,blue) - also triple(RGB,rgb)(bgr,BGR)
            value {unsigned char} -- color filtering value
            below {boolean} -- filter if pixel value below (True) or above (True) value

        Keyword Arguments:
            subcolor {(float, float, float)}: color substituted to filtered ones (RGB). {Default: black (0, 0, 0)}
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if self.bw:
            return
        color = color.lower()
        col = np.zeros(3, np.dtype('B'))
        col[0] = subcolor[2]
        col[1] = subcolor[1]
        col[2] = subcolor[0]
        if color == 'b' or color == 'blue':
            if inplace:
                csp.threshold(self.img, value, 0, below, col)
            else:
                thr = spimage(copy.deepcopy(self.img))
                csp.threshold(thr.img, value, 0, below, col)
                return thr
        elif color == 'g' or color == 'green':
            if inplace:
                csp.threshold(self.img, value, 1, below, col)
            else:
                thr = spimage(copy.deepcopy(self.img))
                csp.threshold(thr.img, value, 1, below, col)
                return thr
        elif color == 'r' or color == 'red':
            if inplace:
                csp.threshold(self.img, value, 2, below, col)
            else:
                thr = spimage(copy.deepcopy(self.img))
                csp.threshold(thr.img, value, 2, below, col)
                return thr
