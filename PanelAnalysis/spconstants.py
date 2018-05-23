""" Costants of project
"""
import os
import numpy as np

# PATHS
HERE = os.path.dirname(__file__)
PATH_IMAGES_NICC = HERE + "\Photos\FromNiccolai\\"
PATH_IMAGES = HERE + "\Photos\\"

# TEST
TEST_IMAGE = "test-picture.jpg"
TEST_IMAGE_2 = "test-picture_2.png"

# FROM NICCOLAI
FN_IMAGE = "ph000245.jpeg"
CF_IMAGE = 'ph000153.jpeg'
# CF_IMAGE = 'test-picture.jpg'

# DISTANCE METRICS
dst_met = {'braycurtis', 'canberra', 'chebyshev',
           'cityblock', 'correlation', 'cosine',
           'dice', 'euclidean', 'hamming',
           'jaccard', 'kulsinski', 'mahalanobis',
           'matching', 'minkowski', 'rogerstanimoto',
           'russellrao', 'seuclidean', 'sokalmichener',
           'sokalsneath', 'sqeuclidean', 'yule'}


# COLOR HANDLING
def hex2rgb(hex):
    hex = hex.split('#')[1].lower()
    r, g, b = bytes.fromhex(hex)
    return (r, g, b)


def hex2bgr(hex):
    hex = hex.split('#')[1].lower()
    r, g, b = bytes.fromhex(hex)
    return (b, g, r)


def rgb2hex(rgb):
    return "#{0:02x}{1:02x}{2:02x}".format(__clamp(rgb[0]), __clamp(rgb[1]), __clamp(rgb[2]))


# GEOMETRIC TOOLS
def rad2deg(rad):
    return rad * 180 / np.pi


def deg2rad(deg):
    return deg * np.pi / 180


# PRIVATE
def __clamp(x):
    return max(0, min(x, 255))
