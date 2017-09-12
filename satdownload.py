"""
Download and extract products (images) from sentinal sattilites with
given product ids
"""

from argparse import ArgumentParser
from sentinelsat import SentinelAPI
import os
import csv
from config import USERNAME, PASSWORD
import zipfile
import shutil
import pandas
from crop import crop
from utils import contains
from utils import get_corner_coordinates
from utils import coordinates_from_geojson
from utils import transform_coordinates
import logging


def parse_arguments():
    parser = ArgumentParser(description='Download products (images) from'
                            'sentinal sattilites with given product ids',
                            epilog='python satdownload.py o.csv')
    parser.add_argument('ids', help='csv file, txt file with ids on each line '
                                    'or a single id')
    parser.add_argument('-d', '--directory', help='directory for downloads',
                        default='./downloads')
    parser.add_argument('--download',
                        help="Download only (Do not extract)",
                        action='store_true')

    return parser.parse_args()


def satdownload_zip(product_id, directory='./', api=None):
    if api is None:
        api = SentinelAPI(USERNAME, PASSWORD,
                          'https://scihub.copernicus.eu/dhus')
    try:
        print('Downloading {}...'.format(product_id))
        api.download(product_id, directory)
    except Exception as inst:
        print('Wrong product id.')


def process_tail(tile_dir, geojson, remove_trash=False):
    """
    Crops images in tile_dir folder.
    :param tile_dir: str
        path to directory
    :param geojson: str
        cropping area
    :param remove_trash: Bool
    """
    for image_name in os.listdir(tile_dir):
        image_path = os.path.join(tile_dir, image_name)
        if contains(get_corner_coordinates(image_path),
                    transform_coordinates(
                        coordinates_from_geojson(geojson))) is True:

            output_file_name = os.path.splitext(image_name)[0] + '.tiff'
            # print('\t' + output_file_name)
            crop(geojson, image_path, os.path.join(tile_dir, output_file_name))
        else:
            print('\t' + 'does not contain your selection')
            logging.info(tile_dir + ' does not contain selection')
            if remove_trash is True:
                logging.info('removing ' + tile_dir)
                shutil.rmtree(tile_dir)
            break
        if remove_trash is True:
            logging.info('removing ' + image_path)
            os.remove(image_path)


def satdownload(product_id, geojson, download_path='./downloads/',
                remove_trash=False, api=None, download_only=False):
    """
    Downloads, extracts and crops products.
    Args:
        product_id: str
            Example: "e3fea737-a83b-4fec-8a5a-68ed8d647c71"
        geojson: str
            Path to geojson file.
        download_path: str, optional
            location to download products.
        remove_trash: bool, default Fasle
            remove unnecessary files after downloading.
        download_only: bool, default False
            Download only (Do not extract).
        api: SentinelAPI api object
    """

    print('Satdownload for ' + product_id)
    logging.debug('satdownload: ' + product_id)
    # create downloads folder
    if os.path.isdir(download_path) is False:
        os.mkdir(download_path)

    if api is None:
        api = SentinelAPI(USERNAME, PASSWORD,
                          'https://scihub.copernicus.eu/dhus')

    # query product information
    product_info = api.get_product_odata(product_id, full=True)

    # directory for images only
    target_directory = os.path.join(download_path, product_info['title'])

    if os.path.isdir(target_directory):
        print('Product is already processed, skipping product...')
        return

    # download
    if os.path.isfile(os.path.join(
            download_path, product_info['title'] + '.zip')) is True:
        print(product_info['title'] + '.zip' + ' exist.')
    else:
        satdownload_zip(product_info['id'], download_path, api=api)
    # skip extraction part
    if download_only is True:
        return

    # extract zip file
    zipfile_path = os.path.join(download_path, product_info['title'] + '.zip')
    zip_ref = zipfile.ZipFile(zipfile_path, 'r')
    zip_ref.extractall(download_path)
    zip_ref.close()

    if os.path.isdir(
            os.path.join(download_path, product_info['Filename'])) is False:
        raise Exception('Directory not found after unzipping.')

    # clearing target directory
    if os.path.isdir(target_directory) is True:
        shutil.rmtree(target_directory)
    os.mkdir(target_directory)

    # product can contain many tails (located in ./GRANULE/)
    granule = os.path.join(download_path, product_info['Filename'], 'GRANULE')
    for i, tail_name in enumerate(os.listdir(granule)):
        print('\ttail name: ' + tail_name)
        tail_folder_name = 'tail.{}'.format(i)
        os.mkdir(os.path.join(target_directory, tail_folder_name))

        # image directories are different for different product types
        image_dir = os.path.join(granule, tail_name, 'IMG_DATA')
        if product_info['Product type'] == 'S2MSI2Ap':
            image_dir = os.path.join(image_dir, 'R10m')

        # move bands into target directory
        for image in os.listdir(image_dir):
            image_prime = image
            if product_info['Product type'] == 'S2MSI2Ap':
                image_prime = image_prime[4:-8] + '.jp2'

            os.rename(os.path.join(image_dir, image),
                      os.path.join(target_directory,
                                   tail_folder_name, image_prime))

    # save info file
    product_info_series = pandas.Series(product_info)
    with open(os.path.join(target_directory, 'info.txt'), 'w') as f:
        f.write(product_info_series.to_string())
    with open(os.path.join(target_directory, 'info.json'), 'w') as f:
        product_info_series.to_json(f)

    # remove unnecessary files
    if remove_trash is True:
        os.remove(zipfile_path)
        shutil.rmtree(os.path.join(download_path, product_info['Filename']))

    # cropping images
    print(target_directory)
    for tail_name in os.listdir(target_directory):
        if os.path.isdir(os.path.join(target_directory, tail_name)) is False:
            continue
        print('\tprocessing ' + tail_name + ' ...')
        process_tail(os.path.join(target_directory, tail_name), geojson,
                     remove_trash=remove_trash)
    print('\n\n')


if __name__ == '__main__':
    api = SentinelAPI(USERNAME, PASSWORD, 'https://scihub.copernicus.eu/dhus')
    args = parse_arguments()
    ids = args.ids
    download_path = os.path.abspath(os.path.join('./', args.directory))

    if os.path.splitext(ids)[1] == '.csv':
        with open(ids, 'r') as csvfile:
            csvfile.readline()
            for line in csv.reader(csvfile, delimiter=','):
                product_id = line[0]
                satdownload(product_id, download_path, api=api,
                            download_only=args.download)

    elif os.path.splitext(ids)[1] == '.txt':
        with open(ids, 'r') as txtfile:
            for line in txtfile.read().splitlines():
                product_id = line
                satdownload(product_id, download_path, api=api,
                            download_only=args.download)
    else:
        product_id = ids
        satdownload(product_id, download_path, api=api,
                    download_only=args.download)
