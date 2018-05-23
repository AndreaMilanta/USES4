"""class to handle images
"""
from matplotlib import pyplot as plt
import cv2
import numpy as np
import copy
from enum import Enum
from scipy.spatial import distance as dst
from scipy.cluster import hierarchy as cl_hier

import spexceptions as spe
import spconstants as spc
import csptools as csp

index = 1   # incremental index of images - starts from 1 otherwise issues on multiple shows of figure 0


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
        self.showcount = 0          # counts how many times .show() has been called
        # initiliatize variables
        self.points = []
        self.rects = []
        self.polys = []
        self.lines = []

        # contours variables
        self.contours = []
        self.hierarchy = []

    def copy(self):
        """Returns a full deepcopy of itslef
        """
        return spimage(copy.deepcopy(self.img))

    def toType(self, type, inplace=False):
        """Converts image to selected type

        Arguments:
            type {str} -- destination type

        Keyword Arguments:
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if 'f' in type and '32' in type:
            if inplace:
                self.img = np.float32(self.img)
            else:
                return spimage(np.float32(self.img))

    def toBW(self, inplace=False):
        """transform the image to black and white+

        Keyword Arguments:
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if inplace and not self.bw:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.bw = True
        elif not inplace and self.bw:
            return spimage(copy.deepcopy(self.img))
        elif not inplace and not self.bw:
            return spimage(cv2.cvtColor(copy.deepcopy(self.img), cv2.COLOR_BGR2GRAY))

    def toInvBW(self, inplace=False):
        """transform the image to inverted black and white (0 becomes 255)

        Keyword Arguments:
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if inplace and not self.bw:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.bw = True
        elif not inplace and self.bw:
            return spimage(copy.deepcopy(self.img))
        elif not inplace and not self.bw:
            return spimage(cv2.cvtColor(copy.deepcopy(self.img), cv2.COLOR_BGR2GRAY))

    def rotate(self, deg, dir='cw', pivot=None, inplace=False):
        """Rotate given image

        Arguments:
            deg{float} -- degrees of rotation

        Keyword Arguments:
            dir {str} -- direction of rotation ('cw', 'clockwide' or 'ccw', 'counterclockwise') {Default: 'cw'}
            pivot {(int, int)} -- pivot for rotation; if None the center is used {Default: None}
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        rows = self.img.shape[0]
        cols = self.img.shape[1]
        if 'ccw' in dir and 'counter' in dir:
            deg *= -1
        if pivot is None:
            pivot = (cols / 2, rows / 2)
        M = cv2.getRotationMatrix2D(pivot, deg, 1)
        if inplace:
            self.img = cv2.warpAffine(self.img, M, (cols, rows))
        else:
            return spimage(cv2.warpAffine(self.img, M, (cols, rows)))

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

    def addLine(self, line, thick=2, color="#FFFFFF"):
        """add line to be drawn on image

        Arguments:
            Line - two options: either (rho, theta) {(float, float)}  or   ((x1,y1), (x2,y2)) {((int, int), (int, int))}
        """
        if not isinstance(line[0], tuple) and not isinstance(line[0], list):
            rho = line[0]
            theta = line[1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            lin = [(x1, y1), (x2, y2), thick, color]
        else:
            lin = [line[0], line[1], thick, color]
        self.lines.append(lin)

    def addContour(self, con, approx, boxMode=False, thick=2, color="#FFFFFF"):
        """Add contour as polygon to be drawn

        Arguments:
            con {contour} -- contour
            approx {float} -- approximation factor (the lower the less approximation)

        Keywards Arguments:
            boxMode {Boolean} -- if true, the contour is approximate as a rectangle (approx is ignored) {Default: False}
        """
        if boxMode:
            # Box Visualization
            self.addPolygon(cv2.boxPoints(cv2.minAreaRect(con)), thick=thick, color=color)
        else:
            # Contour Visualization
            print("new line contour")
            epsilon = approx * cv2.arcLength(con, True)
            approx = cv2.approxPolyDP(con, epsilon, True)
            print(approx)
            self.addPolygon(approx[:, 0], thick=thick, color=color)

    def show(self, view=False, title=None):
        """Gets image ready for display
        if view=True also displais it by calling plt.show()
        """
        # figure ID handling                        The id is given by id repeated showcount times (eg 2, 22, 222, etc...)
        currId = self.id
        for i in range(0, self.showcount):
            currId += self.id * 10
        self.showcount += 1
        # Create figure
        for l in self.lines:
            cv2.line(self.img, l[0], l[1], spc.hex2bgr(l[3]), l[2])
        plt.figure(currId)
        if title is not None:
            plt.title(title)
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
            self.showcount = 0

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

    def getCanny(self, minval, maxval, apSize=3, inplace=False):
        """Get canny edges

        Arguments:
            minval {float} -- minimum value for edge detection (below is not an edge)
            maxval {float} -- maximum value for edge detection (above is an edge)

        Keyword Arguments:
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if inplace:
            self.img = cv2.Canny(self.img, minval, maxval, apertureSize=apSize)
            self.bw = True
        else:
            return spimage(cv2.Canny(self.img, minval, maxval, apertureSize=apSize))

    def applyClahe(self, clipLimit=40, tileGridSize=(8, 8), color='rgb', inplace=False):
        """Apply CLAHE - Contrast Limited Adaptive Histogram Equalization

        Arguments:
            clipLimit {float} -- contrast limiting value to avoid amplification of noise {Default: 40}
            tileGridSize {(int, int)} -- size of tiles on which histogram is computed {Default: (8, 8)}
            color {string} -- color on which to apply the clahe algorithm. if image is BW this value is ignored {Default: 'rgb'}

        Keyword Arguments:
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        # create CLAHE filter
        clahe = cv2.createCLAHE(clipLimit, tileGridSize)
        # BW case
        if self.bw:
            if inplace:
                self.img = clahe.apply(self.img)
            else:
                imgcopy = copy.deepcopy(self.img)
                return spimage(clahe.apply(imgcopy))
        # Color case
        else:
            color = color.lower()
            cols = list(color)
            if inplace:
                b, g, r = cv2.split(self.img)
            else:
                b, g, r = cv2.split(copy.deepcopy(self.img))
            if 'b' in cols:
                b = clahe.apply(b)
            if 'g' in cols:
                g = clahe.apply(g)
            if 'r' in cols:
                r = clahe.apply(r)
            if inplace:
                self.img = cv2.merge((b, g, r))
            else:
                return spimage(cv2.merge((b, g, r)))

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

    def dilate(self, kernel, reps=1, forceBW=False, inplace=False):
        """Applies CV2 dilate function

        Arguments:
            kernel {[int] or (int, int)} -- kernel size for dilation. If only one number, the kernel is assumed squared (Must be ODD)

        Keyword Arguments:
            reps {int}: number of iteration of the algorithm {Default: 1}
            forceBW {Boolean}: if true the image is transformed to BW regardless of its previous state, otherwise it expectes a BW image {Default: False}
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if not self.bw and not forceBW:
            raise spe.NonBWImageException("dilate can only be applied to BW images")
        if isinstance(kernel, int):
            print("is int")
            kernel = (kernel, kernel)
        if inplace:
            if not self.bw:
                self.toBW(inplace=True)
            self.img = cv2.dilate(self.img, kernel, iterations=reps)
        else:
            cp = self.toBW(inplace=False)
            cp.dilate(kernel, reps=reps, inplace=True)
            return cp

    def colorAreas(self, areas, color, inplace=False):
        """colors given areas of provided color

        Arguments:
            areas {[histoArea]} -- List of areas to color:
            color {str} -- color of coloration

        Keyword Arguments:
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        c = spc.hex2bgr(color)
        if self.bw:
            c = (c[0] + c[1] + c[2]) / 3
        for a in areas:
            self.img[a.anchor[0]:a.anchor[0] + a.size[0], a.anchor[1]:a.anchor[1] + a.size[1]] = c

    def erode(self, kernel, reps=1, forceBW=False, inplace=False):
        """Applies CV2 erode function

        Arguments:
            kernel {[int] or (int, int)} -- kernel size for erosion. If only one number, the kernel is assumed squared (Must be ODD)

        Keyword Arguments:
            reps {int}: number of iteration of the algorithm {Default: 1}
            forceBW {Boolean}: if true the image is transformed to BW regardless of its previous state, otherwise it expectes a BW image {Default: False}
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if not self.bw and not forceBW:
            raise spe.NonBWImageException("erode can only be applied to BW images")
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if inplace:
            if not self.bw:
                self.toBW(inplace=True)
            self.img = cv2.erode(self.img, kernel, iterations=reps)
        else:
            cp = self.toBW(inplace=False)
            cp.erode(kernel, reps=reps, inplace=True)
            return cp

    def clearNoise(self, kernel, noisecolor='w', forceBW=False, inplace=False):
        """Clear noise dots

        Arguments:
            kernel {[int] or (int, int)} -- kernel size for erosion. If only one number, the kernel is assumed squared (Must be ODD)

        Keyword Arguments:
            noisecolor {str}: color of noise. Either black ('b') or white ('w') {Default: 'w'}
            forceBW {Boolean}: if true the image is transformed to BW regardless of its previous state, otherwise it expectes a BW image {Default: False}
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if not self.bw and not forceBW:
            raise spe.NonBWImageException("clearNoise can only be applied to BW images")
        # define tranformation type
        nc = noisecolor.lower()[0]
        if 'b' in nc:
            morph = cv2.MORPH_CLOSE
        else:
            morph = cv2.MORPH_OPEN
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if inplace:
            if not self.bw:
                self.toBW(inplace=True)
            self.img = cv2.morphologyEx(self.img, morph, kernel)
        else:
            cp = self.toBW(inplace=False)
            cp.clearNoise(kernel, noisecolor=nc, inplace=True)
            return cp

    def applyMorphTrans(self, transform, kernel, forceBW=False, inplace=False):
        """Applies CV2 morphological transform

        Arguments:
            transform {cv2 enum} -- transform to apply (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT, cv2.MORPH_GRADIENT)
            kernel {[int] or (int, int)} -- kernel size for erosion. If only one number, the kernel is assumed squared (Must be ODD)

        Keyword Arguments:
            forceBW {Boolean}: if true the image is transformed to BW regardless of its previous state, otherwise it expectes a BW image {Default: False}
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if not self.bw and not forceBW:
            raise spe.NonBWImageException("applyMorphTrans can only be applied to BW images")
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if inplace:
            if not self.bw:
                self.toBW(inplace=True)
            self.img = cv2.morphologyEx(self.img, transform, kernel)
        else:
            cp = self.toBW(inplace=False)
            cp.applyMorphTrans(kernel, transform=transform, inplace=True)
            return cp

    def simpleThreshold(self, threshvalue, threshstyle, ceilvalue=255, forceBW=False, inplace=False):
        """Thresholds image

        Arguments:
            threshvalue {int} -- value used for thresholding comparison
            threshstyle {cv2 enum} -- Style of thrsholding (cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV, cv2.THRESH_BINARY,
                                                            cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC)

        Keyword Arguments:
            ceilvalue {int} -- value used as maximum value for ceiling thresholding in THRESH_BINARY and THRESH_BINARY_INVERTED styles {Default: 255}
            forceBW {Boolean}: if true the image is transformed to BW regardless of its previous state, otherwise it expectes a BW image {Default: False}
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if not self.bw and not forceBW:
            raise spe.NonBWImageException("threshold can only be applied to BW images")
        bwimg = self.toBW(inplace=False)
        if inplace:
            self.img = cv2.threshold(bwimg.img, threshvalue, ceilvalue, threshstyle)[1]
            self.bw = True
        else:
            return spimage(cv2.threshold(bwimg.img, threshvalue, ceilvalue, threshstyle)[1])

    def adaptiveThreshold(self, method, threshstyle, kernel, const, ceilvalue=255, forceBW=False, inplace=False):
        """Applyes adaptiveThreshold to image

        Arguments:
            method {str} -- adaptive method ('m', 'mean' <=> cv2.ADAPTIVE_THRESH_MEAN_C), ('g', 'gaussian' <=> cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
            threshstyle {cv2 enum} -- Style of thrsholding (cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV, cv2.THRESH_BINARY,
                                                            cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC)
            kernel {int} -- size of kernel for mean or gaussian computation (Must be ODD)
            const {int} -- constant subtracted by the computed mean

        Keyword Arguments:
            ceilvalue {int} -- value used as maximum value for ceiling thresholding in THRESH_BINARY and THRESH_BINARY_INVERTED styles {Default: 255}
            forceBW {Boolean}: if true the image is transformed to BW regardless of its previous state, otherwise it expectes a BW image {Default: False}
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if not self.bw and not forceBW:
            raise spe.NonBWImageException("threshold can only be applied to BW images")
        method = method.lower()[0]
        if 'm' in method:
            met = cv2.ADAPTIVE_THRESH_MEAN_C
        elif 'g' in method:
            met = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            raise spe.InvalidParameterException('value "' + method + '" is invalid for "method" parameter')
        bwimg = self.toBW(inplace=False)
        if inplace:
            self.img = cv2.adaptiveThreshold(bwimg.img, ceilvalue, met, threshstyle, kernel, const)
            self.bw = True
        else:
            return spimage(cv2.adaptiveThreshold(bwimg.img, ceilvalue, met, threshstyle, kernel, const))

    def getOtsuThreshold(self, ceilvalue=255, immediate=False, inplace=False, forceBW=False):
        """Gets the best thresholding parameter for BIMODAL images using OTSU's method

        Keyword Arguments:
            ceilvalue {int} -- value used as maximum value for ceiling thresholding in THRESH_BINARY and THRESH_BINARY_INVERTED styles {Default: 255}
            immediate {Boolean} -- if true the thresholding is also immediately applied on the image (or on a copy depending on inplace), else the value is returned
            forceBW {Boolean}: if true the image is transformed to BW regardless of its previous state, otherwise it expectes a BW image {Default: False}
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}

        Return:
            val {int} -- if NOT immediate, is the suggested threshold computed by the OTSU algorithm
        """
        if not self.bw and not forceBW:
            raise spe.NonBWImageException("Otsu can only be computed on BW images")
        val, thresh = cv2.threshold(self.img, 0, ceilvalue, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if immediate and inplace:
            self.img = thresh
            self.bw = True
        elif immediate and not inplace:
            return spimage(thresh)
        else:
            return val

    def blur(self, method, param, inplace=False):
        """Blurs image according to given method
            No checks on parameter validity!

        Arguments:
            method {str} -- blurring method ('average', 'a'), ('gaussian', 'g'), ('median', 'm'), ('bilateral', 'b')
            param {} -- parameter specific for each method
                a -> kernelsize {int} or {(int, int)}
                g -> (kernelsize, sigmaX, sigmaY) {(int, float, float)} (sigmaY may be omitted and will be considered equal to sigmaX)
                m -> kernelsize {int}
                b -> (kernelsize, sigmaColor, sigmaSpace) {(int, float, float)}

        Keyword Arguments:
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        method = method.lower()[0]
        # average
        if 'a' in method:
            if isinstance(param, int):
                param = (param, param)
            if inplace:
                self.img = cv2.blur(self.img, param)
            else:
                return spimage(cv2.blur(self.img, param))

        # gaussian
        if 'g' in method:
            if isinstance(param[0], int):
                ksize = (param[0], param[0])
            else:
                ksize = param[0]
            sigmaX = param[1]
            if param[2] is None:
                sigmaY = sigmaX
            else:
                sigmaY = param[2]
            if inplace:
                self.img = cv2.GaussianBlur(self.img, ksize, sigmaX=sigmaX, sigmaY=sigmaY)
            else:
                return spimage(cv2.GaussianBlur(self.img, ksize, sigmaX=sigmaX, sigmaY=sigmaY))

        # median
        if 'm' in method:
            if inplace:
                self.img = cv2.medianBlur(self.img, param)
            else:
                return spimage(cv2.medianBlur(self.img, param))

        # bilateral
        if 'b' in method:
            ksize = param[0]
            sColor = param[1]
            sSpace = param[2]
            if inplace:
                self.img = cv2.bilateralFilter(self.img, ksize, sigmaColor=sColor, sigmaSpace=sSpace)
            else:
                return spimage(cv2.bilateralFilter(self.img, ksize, sigmaColor=sColor, sigmaSpace=sSpace))

    def laplacian(self, ksize, format=cv2.CV_8U, inplace=False):
        """Computes laplacian transform

        Arguments:
            ksize {int} -- kernel size. Must be ODD.

        Keyword Arguments:
            format {cv2 enum}: color data format {Default: cv2.CV_8U}
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if inplace:
            self.img = cv2.Laplacian(self.img, format, ksize=ksize)
            self.bw = True
        else:
            return spimage(cv2.Laplacian(self.img, format, ksize=ksize))

    def getHoughLines(self, rhoacc, thetacc, minLen):
        """retrieve lines from image

        Arguments:
            rhoacc {float} -- Accuracy on rho computation
            thetacc {float} -- accuracy on theta computation
            minLen {float} -- minimum length for a line to be recognized as such

        Returns:
            lines {[(float, float)]} -- list of lines described by rho and theta
        """
        if not self.bw:
            raise spe.NonBWImageException("getHoughLines can only be performed on BW images")
        return list(map(lambda x: x[0], cv2.HoughLines(self.img, rhoacc, thetacc, minLen)))

    def getContours(self, retrmode, approxmode, filterfunc=None):
        """finds contours in the image

        Arguments:
            retrmode {cv2 enum} -- retrieval mode, representing the resulting hierarchy structure
            approxmode {cv2 enum} -- approximation mode

        Keyword Arguments:
            filterfunc {lambda function} -- filtering function {Default: None}

        Returns:
            contours {[contours], [hierarchy]}
        """
        if not self.bw:
            raise spe.NonBWImageException("getContours can only be performed on BW images")
        im2, contours, hierarchy = cv2.findContours(self.img, retrmode, approxmode)
        if filterfunc is not None:
            contours = list(filter(filterfunc, contours))
        return contours, hierarchy

    def fastMeanDenoising(self, strength, winsize=21, blocksize=7, inplace=False):
        """Performs fast mean-based denoising. Works best if noise is gaussian

        Arguments:
            strength {float} -- strength of the filter

        Keyword Arguments:
            winsize {int}: size of window for weighted mean computation     {Default: 21}
            blocksize {int}: size of block to compute weights   {Default: 7}
            inplace {Boolean}: if true the transformation is made in place, else the transformed image is returned as a new spimage class {Default: False}
        """
        if inplace:
            if self.bw:
                self.img = cv2.fastNlMeansDenoising(self.img, None, strength, winsize, blocksize)
            else:
                self.img = cv2.fastNlMeansDenoising(self.img, None, strength, 10, winsize, blocksize)
        else:
            cp = copy.deepcopy(self)
            cp.fastMeanDenoising(strength, winsize, blocksize, inplace=True)
            return cp


    def histogram(self, channel, view=True):
        """Computes histogram of the image

        Arguments:
            channel {string} -- channel on which to compute histogram. If image is BW this value is ignored (R,r,Red,red)(G,g,Green,green)(B,b,Blue,blue)

        Keyword Arguments:
            view {boolean} -- whether to display or not the histogram {Default: True}

        Result:
            hist {hist} -- computed histogram
        """
        # CHANNEL DEFINITION
        channel = channel.lower()[0]
        if self.bw:             # BW
            ch = [0]
        elif 'b' in channel:    # Blue
            ch = [0]
        elif 'g' in channel:    # Green
            ch = [1]
        elif 'r' in channel:    # Red
            ch = [2]
        else:
            ch = [0]            # Default
        hist = cv2.calcHist([self.img], ch, None, [256], [0, 256])
        if view:
            plt.plot(hist)
            plt.xlim([0, 256])
            plt.show()
        return hist

    def getAreaHist(self, area, channel):
        """Computes histogram on given area of image

        Arguments:
            area {histoArea} -- area on which to compute histogram
            channel {string} -- channel on which to compute histogram. If image is BW this value is ignored (R,r,Red,red)(G,g,Green,green)(B,b,Blue,blue)

        Result:
            hist {(int)} -- histogram as flattened list
        """
        # CHANNEL DEFINITION
        channel = channel.lower()[0]
        if self.bw:             # BW
            ch = [0]
        elif 'b' in channel:    # Blue
            ch = [0]
        elif 'g' in channel:    # Green
            ch = [1]
        elif 'r' in channel:    # Red
            ch = [2]
        else:
            ch = [0]            # Default

        # COMPUTE HISTOGRAMS
        mask = np.zeros(self.img.shape[:2], np.uint8)
        mask[area.anchor[0]: area.anchor[0] + area.size[0], area.anchor[1]: area.anchor[1] + area.size[1]] = 255
        hist = cv2.calcHist([self.img], ch, mask, [256], [0, 256])
        area.hist = list(x for y in hist for x in y)
        return hist

    def histoClassification(self, channel, ksize, kstep, dst_metric, numCluster, showDendogram=True, view=True):
        """classify areas according to histogram

        Arguments:
            channel {string} -- channel on which to compute histogram. If image is BW this value is ignored (R,r,Red,red)(G,g,Green,green)(B,b,Blue,blue)
            ksize {[int, int]} -- size of areas to be classified
            kstep {[int, int]} -- step of area shift. kstep[0] is horizontal shift, kstep[1] is vertical shift
            dst_metric {str} -- metric to compute distance
            numCluster {int} -- number of cluster

        Keyword Arguments:
            showDendogram {boolean} -- whether to prepare a dendogram for display {Default: True}
            view {boolean} -- whether to display or not the dendogram (ignored if showDendogram=False) {Default: True}

        returns:
            areas {[histoArea]} -- list of areas with location info and classification
            hists {[histograms]} -- list of histograms
        """
        # CHANNEL DEFINITION
        channel = channel.lower()[0]
        if self.bw:             # BW
            ch = [0]
        elif 'b' in channel:    # Blue
            ch = [0]
        elif 'g' in channel:    # Green
            ch = [1]
        elif 'r' in channel:    # Red
            ch = [2]
        else:
            ch = [0]            # Default

        # COMPUTE HISTOGRAMS
        start = [-kstep[0], -kstep[1]]
        hists = []
        areas = []
        while start[1] + ksize[1] <= self.img.shape[1]:
            start[1] += kstep[1]
            start[0] = -kstep[0]
            while start[0] + ksize[0] <= self.img.shape[0]:
                start[0] += kstep[0]
                mask = np.zeros(self.img.shape[:2], np.uint8)
                mask[start[0]: start[0] + ksize[0], start[1]: start[1] + ksize[1]] = 255
                hist = cv2.calcHist([self.img], ch, mask, [256], [0, 256])
                hist = list(x for y in hist for x in y)
                # std = np.std(hist)
                # avg = np.average(hist)
                hists.append(list([hist[0]]))
                # hists.append(list([std]))
                areas.append(histoArea((start[0], start[1]), (ksize[0], ksize[1]), hist))
        np.reshape(hists, (-1, 1))

        # CLASSIFICATION
        dists = dst.pdist(hists, metric=dst_metric)        # Compute distances
        links = cl_hier.linkage(dists)  # Compute linkages

        if showDendogram:
            currId = self.id
            for i in range(0, self.showcount):
                currId += self.id * 10
            self.showcount += 1
            plt.figure(currId, figsize=(7, 5))
            plt.title('Metric: ' + str(dst_metric))
            plt.xlabel('sample index')
            plt.ylabel('distance')
            cl_hier.dendrogram(links, leaf_rotation=90., leaf_font_size=8)  # font size for the x axis labels
            if view:
                plt.show()

        clusts = cl_hier.fcluster(links, numCluster, criterion='maxclust')
        for i, a in enumerate(areas):
            a.addClassification(clusts[i])

        return areas, hists


class histoArea:
    """Contains information regarding one area of histogram analysis
    """

    def __init__(self, anch, size, hist):
        self.anchor = anch
        self.size = size
        self.hist = hist

    def __eq__(self, area):
        if self.anchor == area.anchor and self.size == area.size:
            return True
        return False

    def addClassification(self, cl):
        self.cl = cl


class CONTENT(Enum):
    """Enum representing the result of the histogram classification
    """
    BACKGROUND = 0
    FOREGROUND = 1
    CONTOUR = 2
