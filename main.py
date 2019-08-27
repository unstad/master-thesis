#!/usr/bin/env python

import exif
import img2pos
import sys

from pprint import pprint
import math


def fail(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        fail("usage: {} <image1> <image2>".format(sys.argv[0]))
    image1_path, image2_path = sys.argv[1], sys.argv[2]

    print("--> Image 1")
    geodetic_coord1 = exif.get_geodetic_coordinate(image1_path)
    pprint(geodetic_coord1)

    print("\n--> Image 2")
    geodetic_coord2 = exif.get_geodetic_coordinate(image2_path)
    pprint(geodetic_coord2)
    print()

    points_3d, t = img2pos.find_3d_points(image1_path, image2_path)
    scale_factor = img2pos.scale_factor_from_geodetic(
        geodetic_coord1, geodetic_coord2, t
    )
    points_3d = points_3d * scale_factor
    img2pos.show_figure(points_3d, t)

    print("\n--> Image 1: Geodetic -> ECEF")
    ecef1 = img2pos.convert_geodetic_to_ecef(
        geodetic_coord1["latitude"],
        geodetic_coord1["longitude"],
        geodetic_coord1["height"],
    )
    print("--> ECEF")
    pprint(ecef1)
    geodetic1 = img2pos.convert_ECEF_to_Geodetic(ecef1)
    geodetic1[0] = math.degrees(geodetic1[0])
    geodetic1[1] = math.degrees(geodetic1[1])
    print("--> Geodetic")
    pprint(geodetic1)


if __name__ == "__main__":
    main()
