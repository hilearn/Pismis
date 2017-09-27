"""
Get colored image from given directory.
If exist colored image named like '...TCI.jp2', get it,
else create colored image from RGB channels
named like '...B04.jp2','...B03.jp2','...B02.jp2'
You can Print or Save image.
"""
from osgeo import gdal, osr
from argparse import ArgumentParser
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from PIL import Image
from utils import Bands
from utils import band_name


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

    parser.add_argument('product', nargs='?',
                        help='Directory containing channels')
    parser.add_argument('-o', '--output',
                        help='Output file path, by default is saves '
                             'into same folder with "TCI1" suffix',
                        default=None)
    args = parser.parse_args()

    return args


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


def save_color_image(directory, r_band, g_band, b_band, suffix='TCI1',
                     bright_limit=3500, output_file=None):
    """
    Creates color image from given bands
    :param directory: str, directory, where are located band files
    :param r_band: Bands enum,  red band
    :param g_band: Bands enum, suffix of green band
    :param b_band: Bands enum, suffix of blue band
    :param suffix: str, suffix for output file
    :param bright_limit: Supremum of chanel brightness.
        Each value in cannel array greater than bright_limit
        to be assigned bright_limit
    :param output_file: str
        output file name. By default it saves into same folder
    :return array, mixed image array
    """
    channel_names = [r_band, g_band, b_band]
    channels = [None, None, None]
    for c in range(len(channel_names)):
        file = band_name(directory, channel_names[c])
        if file is None:
            raise Exception('"{}" band not found in {}.'.
                            format(channel_names[c].value, directory))
        channels[c] = gdal.Open(file).ReadAsArray()

    red_channel, green_channel, blue_channel = channels
    dataset = gdal.Open(file)
    if output_file is None:
        output_file = os.path.splitext(file)[0][:-3] + suffix + '.tiff'

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

    # Create gtif file
    logging.debug('Creating ' + output_file)
    print('Color file:      ' + os.path.split(output_file)[-1])
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_file, image.shape[1], image.shape[0],
                           image.shape[2], gdal.GDT_Byte)
    print(output_file)
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
    return image


def main(args):
    print("Search colored image...")
    colored_image_name = band_name(args.product, Bands.TCI, extension='.jp2')
    if colored_image_name is not None:
        print("There is colored image")
        image_to_transpose = gdal.Open(colored_image_name).ReadAsArray()
        image = np.array(image_to_transpose).transpose(1, 2, 0)

    else:
        print("There isn't colored image. Try mix...\n")
        image = save_color_image(args.product, Bands.RED,
                                 Bands.GREEN, Bands.BLUE,
                                 bright_limit=args.bright_limit,
                                 output_file=args.output)

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


if __name__ == '__main__':
    main(parse_args())
