"""
Get colored image from given directory.
If exist colored image named like '...TCI.jp2', get it,
else create colored image from RGB channels
named like '...B04.jp2','...B03.jp2','...B02.jp2'
You can Print or Save image.
"""
from glob import glob
from osgeo import gdal, osr
from argparse import ArgumentParser
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from PIL import Image


def brightness_limitization(channel, bright_limit=3500):
    channel = np.array(channel)
    channel[channel > bright_limit] = bright_limit
    channel = channel * 255.0 / bright_limit
    return channel


def resize_band(image, height, width):
    """
    Returns a resized copy of image.
    :param image: 2-dimensional array
    :param height: height in pixels
    :param width: width in pixels
    :return: 2-dimensional array
    """
    if image.shape == (height, width):
        print('A')
        return image
    img = Image.fromarray(image)
    img = img.resize((width, height))
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])


def mix(red_channel, green_channel, blue_channel, bright_limit=3500):
    """
    get colored image from RGB channels

    :param red_channel: 2D array
    :param green_channel:
    :param blue_channel:
    :param bright_limit: Supremum of chanel brightness.
        Each value in cannel array greater than bright_limit
        to be assigned bright_limit
    :return: 3D array of (height, width, 3) shape
    """

    red_channel = brightness_limitization(channel=red_channel,
                                          bright_limit=bright_limit)
    green_channel = brightness_limitization(channel=green_channel,
                                            bright_limit=bright_limit)
    blue_channel = brightness_limitization(channel=blue_channel,
                                           bright_limit=bright_limit)

    image_to_transpose = np.array([red_channel, green_channel, blue_channel])
    image = image_to_transpose.transpose(1, 2, 0).astype("uint8")

    return image


def color_image(directory, r_band, g_band, b_band, suffix='TCI1',
                bright_limit=3500):
    """
    Creates color image from given bands
    :param directory: str, directory, where are located band files
    :param r_band: str, suffix of red band
    :param g_band: str, suffix of green band
    :param b_band: str, suffix of blue band
    :param suffix: str, suffix for output file
    :param bright_limit: Supremum of chanel brightness.
        Each value in cannel array greater than bright_limit
        to be assigned bright_limit
    """
    red_channel, green_channel, blue_channel = None, None, None
    dataset = None
    for file in os.listdir(directory):
        file_name = os.path.splitext(file)[0]
        n = file_name[-3:]
        if n == r_band:
            print('\tRed band file:   ' + file)
            dataset = gdal.Open(os.path.join(directory, file))
            red_channel = dataset.ReadAsArray()
        if n == g_band:
            print('\tGreen band file: ' + file)
            green_channel = gdal.Open(
                os.path.join(directory, file)).ReadAsArray()
        if n == b_band:
            print('\tBlue band file:  ' + file)
            blue_channel = gdal.Open(
                os.path.join(directory, file)).ReadAsArray()

    if type(None) in map(type, (red_channel, green_channel, blue_channel)):
        raise Exception('Not all bands found.' +
                        str([y for x, y in
                            zip((red_channel, green_channel, blue_channel),
                             (r_band, g_band, b_band)) if x is None]))

    if not (red_channel.shape == green_channel.shape == blue_channel.shape):
        print('Bands have different resolution.' +
              str((red_channel.shape, green_channel.shape,
                   blue_channel.shape)))
        resolution = max(red_channel.shape, green_channel.shape,
                         blue_channel.shape)
        red_channel = resize_band(red_channel,
                                  resolution[0], resolution[1])
        green_channel = resize_band(green_channel,
                                    resolution[0], resolution[1])
        blue_channel = resize_band(blue_channel,
                                   resolution[0], resolution[1])

    image = mix(red_channel, green_channel, blue_channel, bright_limit)
    output_file = os.path.join(directory, file_name[:-3] + suffix + '.tiff')

    # Create gtif file
    logging.debug('Creating ' + output_file)
    print('\tColor file:      ' + os.path.split(output_file)[-1])
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_file, image.shape[1], image.shape[0],
                           image.shape[2], gdal.GDT_Byte)

    # Writing output raster
    for j in range(image.shape[2]):
        dst_ds.GetRasterBand(j + 1).WriteArray(image[..., j])

    # Setting extension of output raster
    dst_ds.SetGeoTransform(dataset.GetGeoTransform())
    wkt = dataset.GetProjection()
    # Setting spatial reference of output raster
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dst_ds.SetProjection(srs.ExportToWkt())
    # Close output raster dataset
    dataset = None
    dst_ds = None


def main(args):
    image_names = glob("{}/*.jp2".format(args.file))
    there_is_colored_file = False
    print("Search colored image...")
    for name in image_names:
        if "TCI.jp2" in name:  # colored image named "...TCI.jp2"
            colored_image_name = name
            there_is_colored_file = True

    if there_is_colored_file:
        print("There is colored image")
        image_to_transpose = gdal.Open(colored_image_name).ReadAsArray()
        image = np.array(image_to_transpose).transpose(1, 2, 0)

    else:
        print("There isn't colored image. Try mix...\n")
        print("Get red channel...")

        for name in image_names:
            if "B04.jp2" in name:  # red channel image named "...B04.jp2"
                red_channel = gdal.Open(name).ReadAsArray()

        print("Get green channel...")
        for name in image_names:
            if "B03.jp2" in name:  # green channel image named "...B03.jp2"
                green_channel = gdal.Open(name).ReadAsArray()

        print("Get blue channel...")
        for name in image_names:
            if "B02.jp2" in name:  # blue channel image named "...B02.jp2"
                blue_channel = gdal.Open(name).ReadAsArray()

        print("\nMix...")
        image = mix(red_channel=red_channel, green_channel=green_channel,
                    blue_channel=blue_channel, bright_limit=args.bright_limit)

    if args.plot or args.save is not None:
        print("\nPrepare plot...")
        plt.figure(figsize=args.size)
        plt.imshow(image)

        if args.plot:
            print("\nPlot...")
            plt.show()

        if args.save is not None:
            print("\nSave in {}...".format(args.save))
            plt.savefig(args.save)

    return image


def parse_args():
    parser = ArgumentParser(epilog="mix.py ")
    parser.add_argument('-p', '--plot', help='plot image', action='store_true')
    parser.add_argument('--size', type=int, nargs=2, help='plot image',
                        default=[7, 7])
    parser.add_argument('-s', '--save',
                        help='save PNG image in the given directory')
    parser.add_argument('--bright_limit', type=float,
                        help='save PNG image in the given directory',
                        default='3500')

    parser.add_argument('file', nargs='?',
                        help='Directory containing channels')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_args())
