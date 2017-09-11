from argparse import ArgumentParser
from datetime import datetime
import os
import logging
from satquery import satquery, parse_date
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
                                   'file.geojson -f 01-08-2017')
    parser.add_argument('geojson', help='geojson file for fotprint')
    parser.add_argument('-f', '--from', help='query products starting from'
                        'this date, date format: DD-MM-YYYY', default=None)
    parser.add_argument('-t', '--to', help='query products until with this'
                        'date, date format: DD-MM-YYYY', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    os.path.abspath(os.path.join('./', args.geojson))

    if getattr(args, 'from') is None:
        date_from = datetime(1900, 1, 1)
    else:
        date_from = parse_date(getattr(args, 'from'))

    if getattr(args, 'to') is None:
        date_to = datetime(2900, 1, 1)
    else:
        date_to = parse_date(getattr(args, 'to'))

    df = satquery(args.geojson, date_from, date_to)

    size = 0
    for x in df['size']:
        val = float(x[:-3])
        if x[-2:] == 'GB':
            size += val
        elif x[-2:] == 'MB':
            size += val / 1024
    # print(df[['beginposition', 'size']])
    print('Total size: %0.2f' % size + ' GB')

    product_ids = df.index.tolist()

    with open('./ids.txt', 'r') as f:
        product_ids = f.read().splitlines()

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
