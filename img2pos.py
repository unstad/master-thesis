#!/usr/bin/env python

import math
import numpy as np
import numpy.linalg as mlin

from cv2 import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint


SEMI_MAJOR_AXIS = 6378137.0
SEMI_MINOR_AXIS = 6356752.3142


# Returns true if the value v is considered an outlier in the given list of values
def is_outlier(v, values):
    sigma = np.std(values)
    mean = np.mean(values)
    return v > (2 * sigma) + mean or v < mean - (2 * sigma)


# Remove any match where the distance is higher than the average
def clean_matches(matches):
    distances = [m.distance for m in matches]
    avg = np.average(distances)
    return [m for m in matches if m.distance < avg]


# Extract coordinates using matches and key points
def extract_coordinates(matches, key_points, indexField):
    points = []
    for m in matches:
        index = getattr(m, indexField)
        points.append(key_points[index].pt)
    return np.array(points)


# Group 3D points by each axis. Three lists are returned, one for each axis.
def points_by_axis(points):
    points_x = []
    points_y = []
    points_z = []
    for coord in points:
        points_x.append(coord[0][0])
        points_y.append(coord[0][1])
        points_z.append(coord[0][2])
    return points_x, points_y, points_z


def find_3d_points(image1_path, image2_path):
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)  # trainImage

    # Initial calibration matrix from camera
    init_calibration_matrix = np.array(
        [
            [2.78228443e03, 0.00000000e00, 1.65670819e03],
            [0.00000000e00, 2.77797243e03, 1.19855894e03],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    distortion_coefficients = np.array(
        [0.07874525, -0.07184864, -0.00619498, 0.00252332, -0.09900985]
    )

    # Undistort images. getOptimalNewCameraMatrix: 1 tells us that we want to see the "black hills" after undistorting. Exchanging for 0 removes them.
    height, width = img1.shape[:2]
    calibration_matrix, roi = cv2.getOptimalNewCameraMatrix(
        init_calibration_matrix,
        distortion_coefficients,
        (width, height),
        1,
        (width, height),
    )
    img1_distorted = cv2.undistort(
        img1, init_calibration_matrix, distortion_coefficients, None, calibration_matrix
    )
    img2_distorted = cv2.undistort(
        img2, init_calibration_matrix, distortion_coefficients, None, calibration_matrix
    )

    # Crop images
    x, y, w, h = roi
    img1_distorted = img1_distorted[y : y + h, x : x + w]
    img2_distorted = img2_distorted[y : y + h, x : x + w]

    # To display the undistorted images:
    # plt.imshow(img1_distorted), plt.show()
    # plt.imshow(img2_distorted), plt.show()

    # Create an ORB object
    orb = cv2.ORB_create()

    # Detect keypoints
    kp1 = orb.detect(img1_distorted, None)
    kp2 = orb.detect(img2_distorted, None)

    # Find descriptors
    kp1, des1 = orb.compute(img1_distorted, kp1)
    kp2, des2 = orb.compute(img2_distorted, kp2)

    # To draw the keypoints:
    #img1kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0) #flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # img2kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)
    #plt.imshow(img1kp), plt.show()
    # plt.imshow(img2kp), plt.show()

    # Brute-force matcher object. crossCheck=True means that it has to match both ways
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Matching descriptors
    matches = brute_force.match(des1, des2)

    # Clean the matches by distance
    matches = clean_matches(matches)

    # Sort matches in order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    # To draw the first 20 matches:
    #img_matches = cv2.drawMatches(img1_distorted, kp1, img2_distorted, kp2, matches[:], None, flags = 2)
    #plt.imshow(img_matches), plt.show()

    # Extract coordinates
    points1 = extract_coordinates(matches, kp1, "queryIdx")
    points2 = extract_coordinates(matches, kp2, "trainIdx")

    # Find essential Matrix
    essential_matrix, _ = cv2.findEssentialMat(
        points1, points2, calibration_matrix, method=cv2.RANSAC, prob=0.999, threshold=3
    )
    determinant = mlin.det(essential_matrix)
    eps = 1e-10
    if determinant > eps:
        raise Exception(
            "expected determinant to be close to zero, but is {}".format(determinant)
        )

    # Find camera2 position relative to camera1 (t is only in unit)
    _, R, t, _ = cv2.recoverPose(essential_matrix, points1, points2, calibration_matrix)

    # Create camera matrices
    M1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    M2 = np.hstack((R, t))
    camera_matrix1 = np.dot(calibration_matrix, M1)
    camera_matrix2 = np.dot(calibration_matrix, M2)

    # Compute 3D points
    points_3d = []
    for c1, c2 in zip(points1, points2):
        point = cv2.triangulatePoints(camera_matrix1, camera_matrix2, c1, c2)
        points_3d.append(point)
    points_3d = cv2.convertPointsFromHomogeneous(np.array(points_3d))

    return points_3d, t


def scale_factor_from_geodetic(geodetic_coordinate1, geodetic_coordinate2, t):
    ecef1 = convert_geodetic_to_ecef(
        geodetic_coordinate1["latitude"],
        geodetic_coordinate1["longitude"],
        geodetic_coordinate1["height"],
    )
    ecef2 = convert_geodetic_to_ecef(
        geodetic_coordinate2["latitude"],
        geodetic_coordinate2["longitude"],
        geodetic_coordinate2["height"],
    )

    distance = distance_between_two_gps_positions(ecef1, ecef2)
    print("distance=%f" % distance)
    length_t = length_of_translation_vector(t)

    return distance / length_t


def show_figure(points_3d, t):
    # Extract each X, Y and Z value into separate lists for the purpose of eliminating outlier values
    points_x, points_y, points_z = points_by_axis(points_3d)

    # Show points in a figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for coord in points_3d:
        x = coord[0][0]
        y = coord[0][1]
        z = coord[0][2]
        if is_outlier(x, points_x):
            continue
        if is_outlier(y, points_y):
            continue
        if is_outlier(z, points_z):
            continue
        ax.scatter(x, y, z, marker=".")
    ax.scatter(0, 0, 0, marker="s")
    ax.scatter(t[0], t[1], t[2], marker="s")
    plt.show(fig)


def ellipsoid_of_revolution():
    e2 = (math.pow(SEMI_MAJOR_AXIS, 2) - math.pow(SEMI_MINOR_AXIS, 2)) / math.pow(
        SEMI_MAJOR_AXIS, 2
    )
    return e2


def distance_between_two_gps_positions(ECEF1, ECEF2):  # Euclidian distance
    x = ECEF2[0] - ECEF1[0]
    y = ECEF2[1] - ECEF1[1]
    z = ECEF2[2] - ECEF1[2]
    distance = math.sqrt(x * x + y * y + z * z)
    return distance


def length_of_translation_vector(t):
    length_t = math.sqrt(math.pow(t[0], 2) + math.pow(t[1], 2) + math.pow(t[2], 2))
    return length_t


def convert_geodetic_to_ecef(lat, lon, height):  # lat long in radians
    lat = math.radians(lat)
    lon = math.radians(lon)
    e2 = ellipsoid_of_revolution()
    n = SEMI_MAJOR_AXIS / math.sqrt(1 - e2 * math.pow(math.sin(lat), 2))
    ecef = [
        (n + height) * math.cos(lat) * math.cos(lon),
        (n + height) * math.cos(lat) * math.sin(lon),
        (n * (1 - e2) + height) * math.sin(lat),
    ]
    return ecef


# Implemented as code in appendix in Olson, D. K. (1996).
# "Converting earth-Centered, Earth-Fixed Coordinates to Geodetic Coordinates"
def convert_ECEF_to_Geodetic(ECEF):
    e2 = ellipsoid_of_revolution()
    a1 = SEMI_MAJOR_AXIS * e2
    a2 = a1 * a1
    a3 = a1 * e2 / 2
    a4 = (5 / 2) * a2
    a5 = a1 + a3
    a6 = 1 - e2

    z = ECEF[2]
    z_abs = abs(z)
    w2 = ECEF[0] * ECEF[0] + ECEF[1] * ECEF[1]
    w = math.sqrt(w2)
    z2 = ECEF[2] * ECEF[2]
    r2 = w2 + z2
    r = math.sqrt(r2)

    geo = [0, 0, 0]
    if r < 100000:
        geo[0] = 0
        geo[1] = 0
        geo[2] = -1.0e7
        return geo

    geo[1] = np.arctan2(ECEF[1], ECEF[0])  # longitude

    s2 = z2 / r2
    c2 = w2 / r2
    u = a2 / r
    v = a3 - a4 / r

    if c2 > 0.3:
        s = (z_abs / r) * (1.0 + c2 * (a1 + u + s2 * v) / r)
        geo[0] = np.arcsin(s)  # Latitude
        ss = s * s
        c = math.sqrt(1.0 - ss)
    else:
        c = (w / r) * (1.0 - s2 * (a5 - u - c2 * v) / r)
        geo[0] = np.arccos(c)  # Latitude
        ss = 1.0 - c * c
        s = math.sqrt(ss)

    g = 1.0 - e2 * ss
    rg = SEMI_MAJOR_AXIS / math.sqrt(g)
    rf = a6 * rg
    u = w - rg * c
    v = z_abs - rf * s
    f = c * u + s * v
    m = c * v - s * u
    p = m / (rf / g + f)

    geo[0] = geo[0] + p  # Latitude
    geo[2] = f + m * p / 2.0  # Height

    if z < 0.0:
        geo[0] *= -1.0  # Latitude
    return geo
