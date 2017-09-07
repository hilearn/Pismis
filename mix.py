"""
Get colored image from given directory.
If exist colored image named like '...TCI.jp2', get it,
else create colored image from RGB channels
named like '...B04.jp2','...B03.jp2','...B02.jp2'
You can Print or Save image.
"""
from glob import glob
from osgeo import gdal
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt


def brightness_limitization(channel, bright_limit=3500):
    channel = np.array(channel)
    channel[channel > bright_limit] = bright_limit
    channel = channel * 255.0 / bright_limit
    return channel


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
        plt.figure(figsize=eval(args.size))
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
