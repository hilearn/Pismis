from mix import color_image
import os
import shutil
from argparse import ArgumentParser


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
    for d in os.walk(directory):
        if len(d[1]) == 0:
            print('Coloring ' + d[0] + '...')
            try:
                color_image(d[0], 'B04', 'B03', 'B02', 'TCI1', bright_limit)
            except Exception as e:
                print('Error: ' + str(e))


def collect_images(search_directory, target='./colored'):
    """
    Search colored images in <search_directory> and copy them
    into target directory
    :param search_directory: str, directory to search imaegs
    :param target: str, directory to copy images
    """
    for root, dirs, files in os.walk(search_directory):
        for file in files:
            if 'TCI' in file:
                new_file = '.'.join(
                    os.path.normpath(root).split(os.sep)[-2:] + ['tiff'])

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
