import os
import json
from mix import save_color_image
import shutil
from cloud_mask import is_useful
from argparse import ArgumentParser
from crop import crop
from utils import coordinates_from_geojson
from utils import transform_coordinates
from utils import timestamp_to_datetime
from utils import band_name
from utils import Bands


def parse_arguments():
    parser = ArgumentParser(description='Crop, align images and keep '
                                        'only clean ones',
                            epilog='pre_process.py narrow_area_of_interest.ge'
                                   'ojson --input-dir cropped_raw_data_dir --'
                                   'output-dir croppped_cleaned_data')
    parser.add_argument('--geojson', help='crop images with this geojson'
                                          ' if provided', default=None)
    parser.add_argument('input_dir', help='directory for images.')
    parser.add_argument('-o', '--output-dir',
                        help='Directory to save cropped, cleaned data.',
                        default='./data')
    return parser.parse_args()


def copy_and_format_names(origin, destination):
    """
    Remove duplicate images.
    :param origin: str, path to original data.
    :param destination: str, directory to copy and format files.
    """
    # From products with the same date-time keep only one.
    product_paths = []
    dates = set()
    for product in os.listdir(origin):
        info_file = os.path.join(origin, product, 'info.json')
        if os.path.exists(info_file) is False:
            continue
        info = json.load(open(info_file, 'r'))
        date = timestamp_to_datetime(info['Sensing start'])
        if date in dates:
            continue
        dates.add(date)
        product_paths.append(os.path.join(origin, product))

    os.makedirs(destination, exist_ok=True)
    for path in product_paths:
        info = json.load(open(os.path.join(path, 'info.json'), 'r'))
        if info['Satellite'] != 'Sentinel-2':
            continue
        date_str = 'product ' + '{:%Y-%m-%d %H:%M}'.format(
            timestamp_to_datetime(info['Sensing start']))
        tail = None
        for name in os.listdir(path):
            if name.startswith('tail.'):
                tail = os.path.join(path, name)
                break
        if tail is None:
            continue
        # useful bands
        bands = [Bands.RED, Bands.GREEN, Bands.BLUE, Bands.NIR]
        os.makedirs(os.path.join(destination, date_str), exist_ok=True)
        for band in bands:
            shutil.copy(band_name(tail, band),
                        os.path.join(destination, date_str,
                                     band.value + '.tiff'))
        # create colored image
        save_color_image(tail, Bands.RED, Bands.GREEN, Bands.BLUE,
                         output_file=os.path.join(
                             destination, date_str, 'TCI.tiff'))

        # copy info files
        shutil.copy(os.path.join(path, 'info.json'),
                    os.path.join(destination, date_str))
        shutil.copy(os.path.join(path, 'info.txt'),
                    os.path.join(destination, date_str))


def crop_images(directory, selection):
    """
    Crop all tiff files in provided directory.
    :param directory: str, directory containing tiff files.
    :param selection: array, selection for cropping
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            if os.path.splitext(file)[1] == '.tiff':
                crop(selection, path, path)


def remove_unactionable_images(data):
    """
    Keep useful products in data directory.
    :param data: str, path to products
    """
    os.makedirs(os.path.join(data, 'removed'), exist_ok=True)
    for product in os.listdir(data):
        if product.startswith('product') is False:
            continue
        path = os.path.join(data, product)
        if os.path.isdir(path) is False:
            continue
        if is_useful(path, 0.1) is False:
            print('\tRemoving ' + path)
            shutil.copy(os.path.join(path, 'TCI.tiff'),
                        os.path.join(data, 'removed', product + '.tiff'))
            shutil.rmtree(path)
        else:
            shutil.copy(os.path.join(path, 'TCI.tiff'),
                        os.path.join(data, product + '.tiff'))


if __name__ == '__main__':
    args = parse_arguments()
    print(dir(args))

    copy_and_format_names(args.input_dir, args.output_dir)

    if args.geojson is not None:
        crop_images(args.output_dir, transform_coordinates(
            coordinates_from_geojson(args.geojson)))

    remove_unactionable_images(args.output_dir)
