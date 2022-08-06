#!/usr/bin/env python3

"""
Tansu Alpcan
v1 2022-3-14

Contents:
* Example usage of common image processing functions (uses included screenshot
  "./screen.png" for demonstration)
"""

import sys
import cv2 as cv
from PIL import Image


def display_and_wait(title, frame):
    """
    Opens a new window with the given title to display the image data in frame.
    """
    print("Displaying \"%s\" in new window, press any key to continue" % title)
    cv.imshow(title, frame)
    cv.waitKey(0)


def main():
    # read the screenshot (assumed to be in the same folder)
    im = cv.imread('screen.png')
    print("Original shape:", im.shape)
    display_and_wait("Original screenshot", im)

    # turn into gray scale
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    print("Grayscale shape:", im_gray.shape)
    display_and_wait("Grayscale", im_gray)

    # do threshold based segmentation (level given by cv constants as shown below)
    ret, thresh = cv.threshold(im_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    print("Threshold value:", ret)
    print("Thresholded shape:", thresh.shape)
    display_and_wait("Thresholded", thresh)


if __name__ == "__main__":
    main()
