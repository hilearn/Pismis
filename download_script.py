from argparse import ArgumentParser
import os
import logging
import csv
from satdownload import satdownload

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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    os.path.abspath(os.path.join('./', args.geojson))

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
                        remove_trash=True)
        except Exception as e:
            print('!!! Problem with product({})'.format(product_id))
            print(e.message)
            logging.error('Problem with product({})'.format(product_id))
            logging.error(e.message)
