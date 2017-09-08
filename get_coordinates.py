"""
Use like:

    from get_coordinates import get_corner_coordinates
    my_coordinates = get_corner_coordinates('T38TLL_B01.jp2')
"""
import numpy as np
from osgeo import gdal


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


def get_corner_coordinates(image_name):
    """
    Get coordinates of image

    :param image_name: string like "image.jp2"
    :return: array like [[300, 460], [300, 449], [409, 449], [409, 460]]
    """
    ds = gdal.Open(image_name)

    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    ext = _get_extend(gt, cols, rows)

    return np.array(ext)
