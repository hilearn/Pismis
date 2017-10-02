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
from datetime import datetime


def parse_arguments():
    parser = ArgumentParser(description='Crop, align images and keep '
                                        'only clean ones',
                            epilog='pre_process.py narrow_area_of_interest.ge'
                                   'ojson --input-dir cropped_raw_data_dir --'
                                   'output-dir croppped_cleaned_data')
    parser.add_argument('--geojson', help='crop images with this geojson'
                                          ' if provided', default=None)
    parser.add_argument('--input-dir', help='directory for images.',
                        required=True)
    parser.add_argument('-o', '--output-dir',
                        help='Directory to save cropped, cleaned data.',
                        default='./data')
    return parser.parse_args()


def copy_and_format_names(origin, destination, selection=None):
    """
    Remove duplicate images.
    :param origin: str, path to original data.
    :param destination: str, directory to copy and format files.
    :param selection: array, coordinates to crop images
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
        bands = [Bands.RED, Bands.GREEN, Bands.BLUE, Bands.NIR, Bands.SWIR]
        os.makedirs(os.path.join(destination, date_str), exist_ok=True)
        for band in bands:
            if selection is None:
                shutil.copy(band_name(tail, band),
                            os.path.join(destination, date_str,
                                         band.value + '.tiff'))
            else:
                crop(selection, band_name(tail, band),
                     os.path.join(destination, date_str,
                                  band.value + '.tiff'))
        # create colored image
        save_color_image(os.path.join(destination, date_str),
                         Bands.RED, Bands.GREEN, Bands.BLUE,
                         output_file=os.path.join(
                             destination, date_str, Bands.TCI.value + '.tiff'))

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
        if is_useful(path, 0.5) is False:
            print('\tRemoving ' + path)
            shutil.copy(os.path.join(path, 'TCI.tiff'),
                        os.path.join(data, 'removed', product + '.tiff'))
            shutil.rmtree(path)
        else:
            shutil.copy(os.path.join(path, 'TCI.tiff'),
                        os.path.join(data, product + '.tiff'))


def seasonal(path, date_inf="15-05", date_sup="15-10"):
    """
    Resolve if given image (<path>) of season or not according to <date_inf>
    and <date_sup>
    :param path: Directory of the product
    :param date_inf: Day of year in "dd-mm" format
    :param date_sup: Day of year in "dd-mm" format
    :return: Bull
    """
    with open(os.path.join(path, "info.json"), "r") as f:
        info = json.load(f)

    date_inf = datetime.strptime(date_inf, "%d-%m").timetuple().tm_yday
    date_sup = datetime.strptime(date_sup, "%d-%m").timetuple().tm_yday
    day_of_year = timestamp_to_datetime(
        info['Sensing start']).timetuple().tm_yday

    return (day_of_year > date_inf) and (day_of_year < date_sup)


def remove_unseasonal_images(data, date_inf="15-05", date_sup="15-10"):
    """
    Disposs of images below May 15 and over Oct 15
    :param data: str, path to products
    :param date_inf: str, infimum date to consider product as seasonal
    :param date_sup: str, supremum date to consider product as seasonal

    """
    os.makedirs(os.path.join(data, 'removed'), exist_ok=True)
    for product in os.listdir(data):
        if product.startswith('product') is False:
            continue
        path = os.path.join(data, product)
        if os.path.isdir(path) is False:
            continue
        if seasonal(path, date_inf, date_sup) is False:
            print('\tRemoving ' + path)
            shutil.copy(os.path.join(path, 'TCI.tiff'),
                        os.path.join(data, 'removed', product + '.tiff'))
            shutil.rmtree(path)


if __name__ == '__main__':
    args = parse_arguments()
    selection = None
    if args.geojson is not None:
        selection = transform_coordinates(
            coordinates_from_geojson(args.geojson))
    copy_and_format_names(args.input_dir, args.output_dir, selection)
    # remove_unseasonal_images(args.output_dir)
    remove_unactionable_images(args.output_dir)
