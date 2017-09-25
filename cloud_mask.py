"""
Get cloud and clowd shadows mask. Use it like

    ```
    from cloud_mask import get_cloud_mask
        cloud_and_shadows_mask = get_cloud_mask("bands_directory/")
    ```

You can also save "cloud_mask.pkl" file in the same directory alongside bands.
"""
from osgeo import gdal
from glob import glob
import os
import pickle as pkl


def get_cloud_mask_path(directory, cloud_threshold=1500,
                        shadow_threshold=400, save=False):
    """
    Get cloud mask from bands of the image in <directory>
    according to thresholding images by red, green and blue channels

    :param directory: The directory of images ends by "/"
    :param threhold: If red, green and blue values of the pixel
                        are greater than <threshold> to be considered
                        as cloud pixel
    :return: 2D array with 1s and 0s, where 0s corespond to cloudy pixels
            1s - to cloud-free pixels.
    """

    band_names = glob("{}/*.tiff".format(os.path.normpath(directory)))

    for name in band_names:
        if "_B04" in name:
            red_band_name = name
        elif "_B03" in name:
            green_band_name = name
        elif "_B02" in name:
            blue_band_name = name

    red_band = gdal.Open(red_band_name).ReadAsArray()
    green_band = gdal.Open(green_band_name).ReadAsArray()
    blue_band = gdal.Open(blue_band_name).ReadAsArray()
    return get_cloud_mask(red_band, green_band, blue_band, cloud_threshold,
                          shadow_threshold, save)


def get_cloud_mask(red_band, green_band, blue_band, cloud_threshold=1500,
                   shadow_threshold=400, save=False):
    """
    Get cloud mask from provided bands
    according to thresholding images by red, green and blue channels

    :param red_band: array, red color band
    :param green_band: array, green color band
    :param blue_band: array, blue color band
    :param threhold: If red, green and blue values of the pixel
                        are greater than <threshold> to be considered
                        as cloud pixel
    :return: 2D array with 1s and 0s, where 0s corespond to cloudy pixels
            1s - to cloud-free pixels.
    """

    cloud_mask = ((red_band > cloud_threshold) * (green_band > cloud_threshold)
                  * (blue_band > cloud_threshold))
    shadow_mask = ((red_band < shadow_threshold)
                   * (green_band < shadow_threshold)
                   * (blue_band < shadow_threshold))

    mask = ~(cloud_mask + shadow_mask)
    mask = mask.astype('int')
    if save:
        with open('cloud_mask.pkl', 'wb') as f:
            pkl.dump(mask, f)

    return mask
