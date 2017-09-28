from osgeo import gdal
import numpy as np
import cv2
from utils import band_name
from glob import glob
import os


def align_image(base_image_directory, image_to_align_directory):
    """
    Align all bands from <image_to_align_directory> directory to bands from
    <base_image_directory>

    :param base_image_directory: Directory of
    :param image_to_align_directory:
    :return:
    """

    base_image_directory = os.path.normpath(base_image_directory)
    image_to_align_directory = os.path.normpath(image_to_align_directory)

    # Read an image to be aligned
    colored_image_to_align_name = band_name(image_to_align_directory, "TCI1")
    image_to_align_array = gdal.Open(colored_image_to_align_name).ReadAsArray()
    image_to_align_array = image_to_align_array.transpose(1, 2, 0)

    # Read base image
    colored_base_image_name = band_name(base_image_directory, "TCI1")
    base_image_array = gdal.Open(colored_base_image_name).ReadAsArray()
    base_image_array = base_image_array.transpose(1, 2, 0)

    size = base_image_array.shape

    # Convert images to grayscale
    im_base_gray = cv2.cvtColor(image_to_align_array, cv2.COLOR_BGR2GRAY)
    im_to_align_gray = cv2.cvtColor(base_image_array, cv2.COLOR_BGR2GRAY)

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC(im_base_gray, im_to_align_gray,
                                             warp_matrix, warp_mode, criteria)

    bands_names = glob("{}/*.tiff".format(
        os.path.normpath(image_to_align_directory)))
    for name in bands_names:
        band = gdal.Open(name).ReadAsArray()
        if "TCI" in name:
            band = band.transpose(1, 2, 0)
        aligned_band = cv2.warpAffine(band, warp_matrix, (size[1], size[0]),
                                      flags=(cv2.INTER_LINEAR +
                                             cv2.WARP_INVERSE_MAP))

        os.makedirs("{}/aligned".format(
            os.path.normpath(image_to_align_directory)), exist_ok=True)
        cv2.imwrite("{}/aligned/{}.tiff".format(image_to_align_directory,
                                                name.split("/")[-1][:-5]),
                    aligned_band)
