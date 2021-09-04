# -*- coding: utf-8 -*-
import tifffile
import cv2
import numpy as np
from copy import deepcopy


def main():
    path = '/Users/rhc/WorkPlace/tmp/OMA389_028_025_LEFT_DSP.tif'
    img = np.array(tifffile.imread(path))
    labels = deepcopy(img)
    labels[:] = 255
    labels[img == -999] = 0
    cv2.imshow('1', labels)
    cv2.waitKey()


if __name__ == '__main__':
    main()
