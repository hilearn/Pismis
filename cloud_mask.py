"""
Get cloud and clowd shadows mask. Use it like

    ```
    from cloud_mask import get_cloud_mask
        cloud_and_shadows_mask = get_cloud_mask("bands_directory/")
    ```

You can also save "cloud_mask.pkl" file in the same directory alongside bands.
"""
from utils import band_name
from utils import Bands
from utils import read_array
import pickle as pkl


def cloud_mask(bands_directory, cloud_threshold=1500,
               shadow_threshold=20, save=False):
    """
    Get cloud mask from bands of the image in <directory>
    according to thresholding images by red, green and blue channels

    :param directory: The directory of images
    :param threhold: If red, green and blue values of the pixel
                        are greater than <threshold> to be considered
                        as cloud pixel
    :return: 2D array with 1s and 0s, where 0s corespond to cloudy pixels
            1s - to cloud-free pixels.
    """
    red_band_name = band_name(bands_directory, Bands.RED)
    green_band_name = band_name(bands_directory, Bands.GREEN)
    blue_band_name = band_name(bands_directory, Bands.BLUE)

    red_band = read_array(red_band_name)
    green_band = read_array(green_band_name)
    blue_band = read_array(blue_band_name)

    cloud_mask = ((red_band > cloud_threshold) * (green_band > cloud_threshold)
                  * (blue_band > cloud_threshold))
    shadow_mask = ((red_band < shadow_threshold)
                   * (green_band < shadow_threshold)
                   * (blue_band < shadow_threshold))

    mask = ~(cloud_mask + shadow_mask)
    mask = mask.astype('int')
    if save:
        with open('{}/cloud_mask.pkl'.format(bands_directory), 'wb') as f:
            pkl.dump(mask, f)

    return mask


def is_useful(bands_directory_name, useful_part_we_need=0.1):
    """
    Resolve is given image useful or not. If clouds and black areas
    are greater than (1-<useful_part_we_need>),
    the image will be considered as unuseful.

    :param bands_directory_name: The directory containing image bands
    :param useful_part_we_need: The part of useful pixels (0-1), necessary to
                                consider the image as useful
    :return: return True if image is useful, False elsewise
    """

    # This is 2D array of 0s and 1s where 1s indicate useful pixels
    useful_part_ones = cloud_mask(bands_directory_name)
    enough_clear_area = (1. * useful_part_ones.sum() / useful_part_ones.size >
                         useful_part_we_need)
    return bool(enough_clear_area)
