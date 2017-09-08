import numpy as np
from osgeo import gdal
from pyproj import Proj, transform


def _transform_coordinates(coordinates, reverse=False):
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
