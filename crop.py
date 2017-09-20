"""
Crop image my coordinates. Use like:
    from crop import crop
    a = crop("ijevan.geojson","T38TML_20170830T074611_TCI.jp2")

'a' is the array of croped image
"""
from osgeo import gdal, osr
import numpy as np
import logging


def crop(selection, image_name, output_file='o.tiff'):
    """
    Crop image by coordinate masks and save geotiff file.

    :param selection: array
        contains coordinates for cropping
    :param image_name: string
        image file to crop
    :param output_file: string or None
        output file name
    :return: ndarray
        if output_file is None returns array object
    """
    logging.debug('cropping ' + image_name)

    min_x, max_x, min_y, max_y = (selection[:, 0].min(),
                                  selection[:, 0].max(),
                                  selection[:, 1].min(),
                                  selection[:, 1].max())

    dataset = gdal.Open(image_name)
    if dataset is None:
        raise Exception('Could not open ' + image_name)
    # Getting image dimensions
    bands = dataset.RasterCount

    # Getting georeference info
    transform = dataset.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    # Computing Point1(i1,j1), Point2(i2,j2)
    i1 = int(round((min_x - xOrigin) / pixelWidth))
    j1 = int(round((yOrigin - max_y) / pixelHeight))
    i2 = int(round((max_x - xOrigin) / pixelWidth))
    j2 = int(round((yOrigin - min_y) / pixelHeight))

    # cropped image size
    new_cols = (i2 - i1 + 1)
    new_rows = (j2 - j1 + 1)
    band_list = []
    # Read in bands and store all the data in bandList
    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)  # 1-based index
        data = band.ReadAsArray(i1, j1, new_cols, new_rows)
        band_list.append(data)

    new_x = xOrigin + i1 * pixelWidth
    new_y = yOrigin - j1 * pixelHeight
    new_transform = (new_x, transform[1], transform[2],
                     new_y, transform[4], transform[5])

    if output_file is None:
        return np.stack(band_list, axis=-1)

    # Create gtif file
    logging.debug('Creating ' + output_file)
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_file, new_cols, new_rows,
                           bands, band.DataType)

    # Writing output raster
    for j in range(bands):
        data = band_list[j]
        dst_ds.GetRasterBand(j + 1).WriteArray(data)

    # Setting extension of output raster
    dst_ds.SetGeoTransform(new_transform)
    wkt = dataset.GetProjection()
    # Setting spatial reference of output raster
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dst_ds.SetProjection(srs.ExportToWkt())
    # Close output raster dataset
    dataset = None
    dst_ds = None
