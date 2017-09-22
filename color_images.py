from mix import color_image, brightness_limitization
import os
import shutil
from argparse import ArgumentParser
from datetime import datetime
import time
import json
from utils import change_datatype


def parse_arguments():
    parser = ArgumentParser(description='Create colored images and collect'
                                        'into folder.',
                            epilog='python color_images.py ./downloads')
    parser.add_argument('directory', help='directory for images.')

    parser.add_argument('-c', '--collect', help='directory to collect images.',
                        default=None)
    parser.add_argument('--collect-only', help="collect only",
                        action='store_true')

    parser.add_argument('-b', '--bright-limit', type=int,
                        help='Supremum of chanel brightness.',
                        default=3500)
    return parser.parse_args()


def color_images(directory, bright_limit=3500):
    """
    Search tail folder in <directory> and create colored image
    :param directory: str, directory, where to look
    :param bright_limit: int, Supremum of chanel brightness.
    """
    for root, dirs, files in os.walk(directory):
        if len(dirs) == 0:
            try:
                product_dir = os.path.split(os.path.normpath(root))[0]

                # open information about product
                info = json.load(open(os.path.join(product_dir,
                                                   'info.json'), 'r'))
                sentinel = info['Satellite']
                if sentinel == 'Sentinel-2':
                    print('Coloring ' + root + '...')
                    color_image(root, 'B04', 'B03', 'B02', 'TCI1',
                                bright_limit)
                elif sentinel == 'Sentinel-1':
                    print('Changing DType to uint8 ' + root + '...')
                    for file in files:
                        if 'uint8' in file:
                            continue

                        new_file = os.path.splitext(file)[0] + '_uint8' + \
                            os.path.splitext(file)[1]
                        change_datatype(os.path.join(root, file),
                                        os.path.join(root, new_file),
                                        processor=lambda
                                        x: brightness_limitization(x, 255))
                        print('\tuint8 file:  ' + new_file)
                else:
                    print('Unknown satellite')

            except Exception as e:
                print('Error: ' + 'Path: ' + root + '\n' + str(e))


def from_timestamp(timestamp):
    """
    Converts 13 digit timestamp to datetime object
    :param timestamp: int
    :return: datetime
    """
    return datetime.fromtimestamp(time.mktime(time.gmtime(timestamp / 1000.)))


def collect_images(search_directory, target='./colored'):
    """
    Search colored images in <search_directory> and copy them
    into target directory
    :param search_directory: str, directory to search imaegs
    :param target: str, directory to copy images
    """
    for root, dirs, files in os.walk(search_directory):
        for file in files:
            if 'TCI1' in file or 'uint8' in file:
                file_hint = ' '.join([os.path.splitext(file)[0]] +
                                     os.path.normpath(root).split(os.sep)[-2:])
                product_dir = os.path.split(os.path.normpath(root))[0]

                # open information about product
                info = json.load(open(os.path.join(product_dir,
                                                   'info.json'), 'r'))

                sensing_start = from_timestamp(info['Sensing start'])

                new_file = info['Satellite'] + \
                    ' {:%Y-%m-%d %H:%M} '.format(sensing_start) + \
                    file_hint + '.tiff'

                shutil.copy(os.path.join(root, file),
                            os.path.join(target, new_file))


if __name__ == '__main__':
    args = parse_arguments()
    if args.collect_only is False:
        print('Coloring images in ' + args.directory)
        color_images(args.directory, args.bright_limit)

    if args.collect is not None:
        print('Collecting files into ' + args.collect)
        if os.path.isdir(args.collect) is False:
            os.mkdir(args.collect)
        collect_images(args.directory, args.collect)
