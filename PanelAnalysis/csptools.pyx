cpdef unsigned char[:, :, :] threshold(unsigned char[:, :, :] img, unsigned char thr_value, unsigned char col, unsigned char below, unsigned char[:] subcolor):
   # thresholds on a particular color (R,G,B) on the image

   # Arguments:
       # img {nparray 3dim} -- image to filter
       # color {string} -- color to filter (R,r,Red,red)(G,g,Green,green)(B,b,Blue,blue)
       # value {float or (float, float, float)} -- color filtering value
       # below {boolean} -- filter if pixel value below (True) or above (False) value
       # subcolor {(float, float, float)} -- color substituted to filtered ones
   ###
    # set the variable extension types
    cdef int x, y, w, h

    # grab the image dimensions
    h = int(img.shape[0])
    w = int(img.shape[1])
    if below:
        for y in range(0, h):
            for x in range(0, w):
                if img[y, x, col] < thr_value:
                    img[y, x, 0] = subcolor[0]
                    img[y, x, 1] = subcolor[1]
                    img[y, x, 2] = subcolor[2]
    else:
        for y in range(0, h):
            for x in range(0, w):
                if img[y, x, col] > thr_value:
                    img[y, x, 0] = subcolor[0]
                    img[y, x, 1] = subcolor[1]
                    img[y, x, 2] = subcolor[2]
    return img
