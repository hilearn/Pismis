from argparse import ArgumentParser
import os
import logging
import csv
from config import ACCOUNTS
from satdownload import satdownload
from sentinelsat import SentinelAPI

DOWNLOAD_PATH = './downloads/'

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='./myapp.log',
                    filemode='w')


def parse_arguments():
    parser = ArgumentParser(description='Download all products for '
                                        'given selection',
                            epilog='example: python download_script.py '
                                   'file.geojson ids.txt')
    parser.add_argument('geojson', help='geojson file for fotprint')
    parser.add_argument('ids', help='txt or csv file for product ids')
    parser.add_argument('--username', default=None)
    parser.add_argument('--password', default=None)
    parser.add_argument('--user', type=int,
                        help='user index from config file', default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    username, password = ACCOUNTS[args.user
                                  if args.user in range(len(ACCOUNTS))
                                  else 0]
    if args.username is not None and args.password is not None:
        username, password = args.username, args.password

    api = SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')
    print('Logging in as:')
    print('  username: ' + username)
    print('  password: ' + password)

    product_ids = []
    if os.path.splitext(args.ids)[1] == '.csv':
        with open(args.ids, 'r') as csvfile:
            csvfile.readline()
            for line in csv.reader(csvfile, delimiter=','):
                product_ids.append(line[0])

    elif os.path.splitext(args.ids)[1] == '.txt':
        with open(args.ids, 'r') as txtfile:
            for line in txtfile.read().splitlines():
                product_ids.append(line)

    print('all products...')
    print('\n'.join(product_ids))

    for product_id in product_ids:
        try:
            logging.debug('Processing product-id: ' + product_id)
            satdownload(product_id, args.geojson, DOWNLOAD_PATH,
                        remove_trash=True, api=api)
        except Exception as e:
            print('!!! Problem with product({})'.format(product_id))
            print(e)
            logging.error('Problem with product({})'.format(product_id))
            logging.error(e)
