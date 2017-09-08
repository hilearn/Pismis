"""
Crop image my coordinates. Use like:
    from crop import crop
    a = crop("ijevan.geojson","T38TML_20170830T074611_TCI.jp2")

'a' is the array of croped image
"""
import json
from osgeo import gdal
from utils import get_corner_coordinates
from utils import _transform_coordinates


def crop(selection_geojson, image_name):
    """
    Crop image by coordinate masks

    :param selection_geojson: geojson file name
    :param image: string like "image.jp2"
    :return:
    """

    selection = json.load(open(selection_geojson))
    new_coordinates_lat_long = \
        selection['features'][0]['geometry']['coordinates'][0][:4]
    new_coordinates = _transform_coordinates(new_coordinates_lat_long)

    origin_coordinates = get_corner_coordinates(image_name)
    image_array = gdal.Open(image_name).ReadAsArray()

    if len(image_array.shape) == 3:  # Check is it 3D colored
                                        # image or 2D channel
        image_array = image_array.transpose(1, 2, 0)

    # Get origin image shapes
    bottom_origin = origin_coordinates[:, 1].min()
    top_origin = origin_coordinates[:, 1].max()

    left_origin = origin_coordinates[:, 0].min()
    right_origin = origin_coordinates[:, 0].max()

    height_origin = abs(top_origin - bottom_origin)
    width_origin = abs(right_origin - left_origin)

    # Get new image shapes
    bottom_new = new_coordinates[:, 1].min()
    top_new = new_coordinates[:, 1].max()

    left_new = new_coordinates[:, 0].min()
    right_new = new_coordinates[:, 0].max()

    x_1 = (image_array.shape[1] * (left_new - left_origin)
           / width_origin).astype(int)
    x_2 = (image_array.shape[1] * (right_new - left_origin)
           / width_origin).astype(int)

    y_1 = (image_array.shape[0] * (top_origin - top_new)
           / height_origin).astype(int)
    y_2 = (image_array.shape[0] * (top_origin - bottom_new)
           / height_origin).astype(int)

    cropped_image_array = image_array[y_1:y_2, x_1:x_2]
    return cropped_image_array
