#!/usr/bin/env python

import sys

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pprint import pprint

# PIL documentation: https://pillow.readthedocs.io/en/stable/handbook/index.html


# Extracts exif data and makes it understandable
def get_exif(image):
    exif_data = {}
    data = image.getexif()
    for tag, value in data.items():
        field_name = TAGS.get(tag, tag)
        if field_name == "GPSInfo":
            gps_data = {}
            for gps_tag, gps_value in value.items():
                gps_field_name = GPSTAGS.get(gps_tag, gps_tag)
                gps_data[gps_field_name] = gps_value
            exif_data[field_name] = gps_data
        else:
            exif_data[field_name] = value
    return exif_data


# Lat and long are given in rational64u. Returns lat long in degrees.
# rational64u gives lat long in degrees, minutes and seconds in tuples with denominator or and nominator
# Converts to degrees
def lat_long_degrees(value):
    deg = value[0][0] / value[0][1]
    minute = value[1][0] / value[1][1]
    second = value[2][0] / value[2][1]
    return deg + (minute / 60) + (second / 3600)


def get_lat_long(exif_data):
    latitude = None
    longitude = None
    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]
        latitude = lat_long_degrees(gps_info["GPSLatitude"])
        latitude_ref = gps_info["GPSLatitudeRef"]
        longitude = lat_long_degrees(gps_info["GPSLongitude"])
        longitude_ref = gps_info["GPSLongitudeRef"]

        if latitude_ref != "N":
            latitude = 0 - latitude

        if longitude_ref != "E":
            longitude = 0 - longitude
    return latitude, longitude


def get_direction(exif_data):
    direction = None
    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]
        direction_ref = gps_info["GPSImgDirectionRef"]
        direction = gps_info["GPSImgDirection"]
        direction = direction[0] / direction[1]
    return direction, direction_ref


def get_height(exif_data):
    height = None
    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]
        height_ref = gps_info["GPSAltitudeRef"]
        height = gps_info["GPSAltitude"]
        height = height[0] / height[1]
        if height_ref == b"\x00":
            height = height
        else:
            height = -height
    return height


def get_geodetic_coordinate(image_path):
    image = Image.open(image_path)
    exif_data = get_exif(image)
    height = get_height(exif_data)
    direction = get_direction(exif_data)
    lat, lon = get_lat_long(exif_data)
    return {"height": height, "direction": direction, "latitude": lat, "longitude": lon}
