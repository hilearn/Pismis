from osgeo import ogr
from osgeo import gdal
import numpy as np
from pyproj import Proj, transform
import json
import logging
logging.debug('import utils')


def coordinates_from_geojson(geojson):
    """
    Get coordinates from geojson
    :param geojson: string
        path o geojson file
    :return: vertices of the polygon
    """

    f = json.load(open(geojson, 'r'))
    return np.array(f['features'][0]['geometry']['coordinates'][0][:-1])


def poly_from_list(poly_list):
    """
    Creates Polygon object.
    :param poly_list: list of vertices.
    :return: Polygon object
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for point in poly_list:
        ring.AddPoint_2D(point[0], point[1])
    ring.AddPoint_2D(poly_list[0][0], poly_list[0][1])
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly


def contains(poly_contains, poly):
    """
    Returns True if <poly_contains> contains <poly>
    :param poly_contains: list of lists or nx2 array
        Coordinates of vertices of the polygon (n is number of vertices)
    :param poly: list of lists or nx2 array
        Coordinates of vertices of the polygon (n is number of vertices)
    :return: bool
    """
    poly_contains = poly_from_list(poly_contains)
    poly = poly_from_list(poly)
    return poly_contains.Contains(poly)


def transform_coordinates(coordinates, reverse=False):
    """
    Transform coordinates from Geojson like to Gdal like
    :param coordinates: coordinates in system epsg:4326.
        Common lat.long
    :return: coordinates in system epsg:32638
    """
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:32638')
    if reverse is True:
        inProj, outProj = outProj, inProj

    new_coordinates = []
    for coordinate in coordinates:
        new_coordinate = transform(inProj, outProj, coordinate[0],
                                   coordinate[1])
        new_coordinates.append(new_coordinate)
    return np.array(new_coordinates)


def get_corner_coordinates(image_name):
    """
    Get coordinates of image

    :param image_name: string like "image.jp2"
    :return: array like [[300, 460], [300, 449], [409, 449], [409, 460]]
    """

    def _get_extend(gt, cols, rows):
        ext = []
        xarr = [0, cols]
        yarr = [0, rows]

        for px in xarr:
            for py in yarr:
                x = gt[0] + (px * gt[1]) + (py * gt[2])
                y = gt[3] + (px * gt[4]) + (py * gt[5])
                ext.append([x, y])
            yarr.reverse()
        return ext

    ds = gdal.Open(image_name)

    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    ext = _get_extend(gt, cols, rows)

    return np.array(ext)
