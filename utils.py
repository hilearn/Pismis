from osgeo import gdal, ogr, osr
import numpy as np
from pyproj import Proj, transform
import json
import logging
from datetime import datetime
import time
from glob import glob
import os
import enum

logging.debug('import utils')


@enum.unique
class Bands(enum.Enum):
    """
    Sentinel 2 bands.
    """
    RED = 'B04'
    GREEN = 'B03'
    BLUE = 'B02'
    NIR = 'B08'  # Near infrared
    TCI = 'TCI'  # Colored image
    TCI1 = 'TIC1'


def timestamp_to_datetime(timestamp):
    """
    Converts 13 digit timestamp to datetime object
    :param timestamp: int
    :return: datetime
    """
    return datetime.fromtimestamp(time.mktime(time.gmtime(timestamp / 1000.)))


def band_name(directory, band, extension='.tiff'):
    """
    Get band name we need from given <directory>

    :param directory: The directory containing bands (...B04.tiff,
                      ...B03.tiff, ... )
    :param band: Bands enum
    :param extension: str, extension of the file, (e.g. ".tiff", ".jp2")
    :return: Full band name (including directory)
    """
    names = glob("{}/*{}".format(os.path.normpath(directory), extension))
    for name in names:
        if name.endswith(band.value + extension):
            return name
    return None


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


def transform_coordinates(coordinates, in_system='epsg:4326',
                          out_system='epsg:32638'):
    """
    Transform coordinates from Geojson like to Gdal like
    :param coordinates: coordinates in input system.
        Common lat.long
    :param in_system: input coordinate system, default epsg:4326
    :param out_system: output coordinate system, default epsg:32638
    :return: coordinates in output system
    """
    inProj = Proj(init=in_system)
    outProj = Proj(init=out_system)

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


def change_datatype(input_file, output_file=None, processor=lambda x: x,
                    output_type=gdal.GDT_Byte):
    if output_file is None:
        output_file = input_file

    dataset = gdal.Open(input_file)
    transform = dataset.GetGeoTransform()

    band_list = []
    for i in range(dataset.RasterCount):
        band = dataset.GetRasterBand(i + 1)  # 1-based index
        data = processor(band.ReadAsArray())
        band_list.append(data)

    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_file, dataset.RasterXSize,
                           dataset.RasterYSize,
                           len(band_list), output_type)

    # Writing output raster
    for j in range(len(band_list)):
        dst_ds.GetRasterBand(j + 1).WriteArray(band_list[j])

    # Setting extension of output raster
    dst_ds.SetGeoTransform(transform)
    wkt = dataset.GetProjection()
    # Setting spatial reference of output raster
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dst_ds.SetProjection(srs.ExportToWkt())
    # Close output raster dataset
    dataset = None
    dst_ds = None
