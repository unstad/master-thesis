#!/usr/bin/env python

import glob
import numpy as np
import sys

from cv2 import cv2


# Find calibration matrix from given images. Chessboard used must be asymmetric.
def find_calibration_matrix(images):
    # Prepare real world points
    rwp = np.zeros((7 * 9, 3), np.float32)
    rwp[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)

    # Arrays to store 3D points in the real world and 2D points in image plane
    rw_points = []
    image_points = []

    for image in images:
        img = cv2.imread(image)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Take in a 8-bit grayscale image and the number of internal corners of chessboard and flags.
        # CLAIB_CV_FAST_CHECK = check quickly for corners and end if none to save time
        found, corners = cv2.findChessboardCorners(
            img_gray, (7, 9), flags=cv2.CALIB_CB_FAST_CHECK
        )

        if found:
            rw_points.append(rwp)
            # Needs accuracy below pixel level
            corners = cv2.cornerSubPix(
                img_gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            image_points.append(corners)

            # Draws the corners on picture
            # cv2.drawChessboardCorners(img, (7, 9), cornersnew, retval)
            # cv2.imshow('img', img)
            # cv2.waitKey()
        else:
            print("no corners found in " + image, file=sys.stderr)

    cv2.destroyAllWindows()

    _, cameramatrix, distortioncoefficients, _, _ = cv2.calibrateCamera(
        rw_points, image_points, img_gray.shape[::-1], None, None
    )
    return cameramatrix, distortioncoefficients


def fail(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        fail("usage: {} <image> ...".format(sys.argv[0]))
    images = sys.argv[1:]
    calibration_matrix, distortion_coefficients = find_calibration_matrix(images)
    print(
        "calibration matrix {}\ndistortion coefficients {}".format(
            calibration_matrix, distortion_coefficients
        )
    )


# Example run: ./camera_calibration.py img/chessboard/*.jpeg
if __name__ == "__main__":
    main()
