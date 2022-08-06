#!/usr/bin/env python3

"""
Tansu Alpcan
v1 2022-3-14

Contents:
* Functions to process and extract lines from an image
* A demonstration of each image processing function when applied to the image
  "./screen.png"

Suggestions:
* This type of approach is good for highway line finding, for example try the
straight_road environment.
* Follow some of these links for lane detection:
https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132
https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
https://github.com/MaybeShewill-CV/lanenet-lane-detection
"""

import sys
import cv2 as cv
import numpy as np


def display_and_wait(title, frame):
    """
    Opens a new window with the given title to display the image data in frame.
    """
    print("Displaying \"%s\" in new window, press any key to continue" % title)
    cv.imshow(title, frame)
    cv.waitKey(0)


def do_canny(frame):
    """
    Returns a modified version of frame, where only "edges" within the image
    are retained, all other pixels are set to 0.
    """
    # Optionally convert the image to grayscale to make further processing less CPU intensive
    # gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    # Applies a blur to the image (not mandatory since Canny will do this for us)
    blur = cv.GaussianBlur(frame, (5, 5), 0)

    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    return cv.Canny(blur, 50, 150)


def do_segment(frame):
    """
    Returns a masked version of the given frame that only contains data in a
    region that is likely to contain road lane markers.
    This region is a combination of rectangle and triangle occupying the lower
    portion of the given frame.
    """
    # Since an image is a multi-dimensional array containing the relative
    # intensities of each pixel in the image, we can use frame.shape to return a
    # tuple: [number of rows, number of columns, number of channels] of the
    # dimensions of the frame
    # Since height begins from 0 at the top, the y-coordinate of the bottom of
    # the frame is its height
    height = frame.shape[0]
    width = frame.shape[1]

    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)

    mask_points = np.array([[
        (0, height),
        (width, height),
        (width, int(height*0.6)),
        (int(width/2), int(height/3)),
        (0, int(height*0.6))
    ]])

    # Creates a mask that will be used to keep only the regions of a frame that
    # contain lines that we care about
    cv.fillPoly(mask, mask_points, 255)
    # Optionally display the mask
    # display_and_wait("Segmentation Mask", mask)

    # A bitwise AND operation between the mask and frame keeps only the triangular area of the frame
    return cv.bitwise_and(frame, mask)


def visualize_lines(frame, lines_arr):
    """
    Returns a new frame displaying only straight lines as defined by lines_arr
    Each item in lines_arr must be a 4-tuple (start x, start y, end x, end y)
    """
    # Creates an image filled with zero intensities with the same dimensions as the frame
    line_frame = np.zeros_like(frame)
    color = (255, 255, 255)  # BGR, note: if the frame is grayscale, only the first entry is used
    thickness = 5  # px

    if lines_arr is not None:
        for line in lines_arr:
            for x1, y1, x2, y2 in line:
                # Draw lines between two coordinates
                cv.line(line_frame, (x1, y1), (x2, y2), color, thickness)
    return line_frame


def main():
    im = cv.imread('screen.png')

    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    display_and_wait("Gray", im_gray)

    print("Original shape:", im_gray.shape)

    # Optionally, threshold each image
    # ret, thresh = cv.threshold(im_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    canny = do_canny(im_gray)
    display_and_wait("canny", canny)
    print("Canny shape:", canny.shape)

    segment = do_segment(canny)
    display_and_wait("segment", segment)

    hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 50, maxLineGap = 60)
    print("Hough line description:\n {}".format(hough))

    lines_visualize = visualize_lines(segment, hough)
    display_and_wait("hough", lines_visualize)
    lines_visualize.save("line.png")
    print(hough)
    print(lines_visualize.shape)
    print(lines_visualize)


if __name__  == '__main__':
    main()
