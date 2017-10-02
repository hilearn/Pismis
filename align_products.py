import numpy as np
import cv2
from utils import band_name, Bands
import os
from osgeo import gdal, osr
import shutil
from argparse import ArgumentParser
import json
from distutils.dir_util import copy_tree
from glob import glob


def parse_arguments():
    parser = ArgumentParser(description='Align all products by given '
                                        'base product',
                            epilog='python align_products.py ./data'
                                   ' --base ./base_product')
    parser.add_argument('data', help='directory of preprocessed products.')
    parser.add_argument('output', help='directory for aligned products.')

    return parser.parse_args()


def create_init_align_json(data_path):
    """
    Creates align_info.json file containing warp_matrices and .geojson data. If
    that file exists, add warp_matrices with None value for new products (if
    found) and doesn't touch existing data

    :param data_path: Directory containing products.
        Be shure that there is .geojson file here
    """
    data_path = os.path.normpath(data_path)

    json_file_name = os.path.join(data_path, "align_info.json")
    if os.path.exists(json_file_name):
        with open(json_file_name, "r") as f:
            align_info = json.load(f)
    else:
        geojson_file_name = "{}.geojson".format(data_path)
        with open(os.path.join(data_path, geojson_file_name), "r") as f:
            crop_geojson = json.load(f)
        align_info = {"crop_geojson": crop_geojson}
        align_info["warp_matrices"] = {}

    product_paths = glob("{}/*/".format(data_path))
    for path in product_paths:
        with open(os.path.join(path, "info.json")) as f:
            product_info = json.load(f)
            product_title = product_info["title"]

        product_warp = align_info["warp_matrices"].get(product_title)
        if product_warp is None:
            align_info["warp_matrices"][product_title] = None

    with open(json_file_name, "w") as f:
        json.dump(align_info, f, indent=4)


def warp_image(image, warp_matrix, output_image, size=None):
    """
    Warp tiff file with warp matrix and areate new tiff file.
    :param image: str, imput tiff file
    :param warp_matrix: 2x3 array, warp matrix for warping/aligning image.
    :param output_image: aligned image name
    :param size: aligned image, if not given, uses input image size
    :return: aligned image array
    """
    dataset = gdal.Open(image)
    array = dataset.ReadAsArray()
    if size is None:
        size = array.shape[-2:]

    band_list = []
    for i in range(dataset.RasterCount):
        band = dataset.GetRasterBand(i + 1)  # 1-based index
        warped_band = cv2.warpAffine(band.ReadAsArray(), warp_matrix,
                                     (size[1], size[0]),
                                     flags=(cv2.INTER_LINEAR +
                                            cv2.WARP_INVERSE_MAP))
        band_list.append(warped_band)

    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_image, warped_band.shape[1],
                           warped_band.shape[0],
                           len(band_list), band.DataType)
    # Writing output raster
    for j in range(len(band_list)):
        dst_ds.GetRasterBand(j + 1).WriteArray(band_list[j])

    # Setting extension of output raster
    transform = dataset.GetGeoTransform()
    dst_ds.SetGeoTransform(transform)
    wkt = dataset.GetProjection()
    # Setting spatial reference of output raster
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dst_ds.SetProjection(srs.ExportToWkt())
    # Close output raster dataset
    dataset = None
    dst_ds = None
    return np.stack(band_list, axis=-1)


def align_product(prouct_path, warp_matrix, aligned_product_path=None,
                  size=None):
    """
    Align all images in product.
    :param prouct_path: str, path to product
    :param warp_matrix: 2x3 array, warp matrix for warping/aligning images.
    :param aligned_product_path: str, directory for aligned product,
        By default it creates "aligned" folder inside product directory.
    :param size: tuple, size of aligned images
    :return: str, aligned product path
    """
    if aligned_product_path is None:
        aligned_product_path = os.path.join(prouct_path, 'aligned')
    os.makedirs(aligned_product_path, exist_ok=True)

    for file in os.listdir(prouct_path):
        path_to_file = os.path.join(prouct_path, file)
        if file.endswith('.tiff'):
            warp_image(path_to_file, warp_matrix,
                       os.path.join(aligned_product_path, file), size)
        elif file.startswith('info.'):
            shutil.copy(path_to_file, aligned_product_path)
    return aligned_product_path


def _get_warp_matrix(base_image, image):  # Allready unused
    """
    Align image to base image and return warp matrix.
    :param base_image: RGB array, base image
    :param image: RGB array, image to align
    :return: 2x3 array, warp matrix for warping/aligning images.
    """
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float64)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)

    # do greyscale
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cc, warp_matrix = cv2.findTransformECC(base_image, image, warp_matrix,
                                           warp_mode, criteria)
    return warp_matrix


def align_data(data_path, aligned_data_path,
               align_info_file="align_info.json"):
    """
    Aligns all products in data directory by given base product.
    e.g. align_data("Ijevan","Ijevan/product","Ijevan_aligned")

    :param data_path: str, path to preprocessed data / Ijevan
    :param base_product: str, base broduct
    :param aligned_data_path: str, directory for aligned products
    """
    with open(os.path.join(data_path, align_info_file), "r") as f:
        align_info = json.load(f)

    for product in os.listdir(data_path):
        path = os.path.join(data_path, product)
        if os.path.exists(os.path.join(path, 'info.json')) is False:
            continue
        with open(os.path.join(path, 'info.json'), "r") as f:
            product_info = json.load(f)
        warp_matrix = align_info["warp_matrices"].get(product_info["title"])
        if warp_matrix is not None:
            print('Aligning {}...'.format(path))
            warp_matrix = np.array(warp_matrix, dtype=np.float32)
            align_product(path, warp_matrix,
                          os.path.join(aligned_data_path, product))
        else:
            copy_tree(path, os.path.join(aligned_data_path, product))

        shutil.copy(band_name(os.path.join(aligned_data_path, product),
                              Bands.TCI),
                    os.path.join(aligned_data_path, product + '.tiff'))


if __name__ == '__main__':
    args = parse_arguments()
    align_data(args.data, args.output)
