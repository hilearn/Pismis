"""
Query products for given footprint and time filter'
"""
import numpy as np
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import datetime
import os
from argparse import ArgumentParser
from config import USERNAME, PASSWORD


def parse_arguments():
    parser = ArgumentParser(description='Query products for given footprint '
                                        'and time filter',
                            epilog='example: python satquery.py ijevan.geojson'
                                   ' --output q --platform Sentinel-1'
                                   ' --shuffle --split 6 -from 01-08-2017')
    parser.add_argument('geojson', help='geojson file for fotprint')
    parser.add_argument('-f', '--from', help='query products starting from'
                        'this date, date format: DD-MM-YYYY', default=None)
    parser.add_argument('-t', '--to', help='query products until with this'
                        'date, date format: DD-MM-YYYY', default=None)
    parser.add_argument('-o', '--output', help='output into *.csv file',
                        default=None)
    parser.add_argument('-s', '--split', type=int,
                        help='split ids into files.', default=0)
    parser.add_argument('-p', '--platform', help="Platform name for query"
                                                 " (default 'Sentinel-2')",
                        choices=['Sentinel-1', 'Sentinel-2'],
                        default='Sentinel-2')
    parser.add_argument('--shuffle', help='Shuffle product ids.',
                        action='store_true')
    parser.add_argument('--skip-cloudy-percentage', type=int,
                        help='Query Copernicus not to include images that '
                             'have very high cloud cover', default=95)

    return parser.parse_args()


def parse_date(date):
    return datetime.strptime(date, "%d-%m-%Y")


def satquery(geojson, date_from=None, date_to=None, platform='Sentinel-2',
             cloud_cover_percentage=95):
    """
    Query products with given properties.
    :param geojson: str
        The geojson file path for footprint.
    :param date_from: datetime, optional
    :param date_to: datetime, optional
        The time interval filter based on the
        Sensing Date of the products
    :param platform: string
        'Sentinel-1' or 'Sentinel-2'
    :param cloud_cover_percentage, Maximum cloud coverage percentage. Hundred
        percent cloud cover means no clear sky is visible. Default is 95%
    :return: Pandas DataFrame, Return the products from a query response as
        a Pandas DataFrame with the values in their appropriate Python types.
    """

    api = SentinelAPI(USERNAME, PASSWORD, 'https://scihub.copernicus.eu/dhus')

    footprint = geojson_to_wkt(read_geojson(geojson), decimals=6)
    kwargs = dict()
    kwargs['platformname'] = platform
    if platform == 'Sentinel-1':
        # Level-1 Ground Range Detected (GRD) products
        kwargs['producttype'] = 'GRD'
    elif platform == 'Sentinel-2':
        kwargs['cloudcoverpercentage'] = (0, cloud_cover_percentage)

    products = api.query(footprint, date=(date_from, date_to),
                         area_relation='Contains', **kwargs)
    df = api.to_dataframe(products)
    return df.sort_values(by='beginposition')


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

    df = satquery(args.geojson, date_from, date_to, platform=args.platform,
                  cloud_cover_percentage=args.skip_cloudy_percentage)

    # Estimate download size
    size = 0
    for x in df['size']:
        val = float(x[:-3])
        if x[-2:] == 'GB':
            size += val
        elif x[-2:] == 'MB':
            size += val / 1024

    if args.shuffle is True:
        print('Shuffle product ids...')
        df = df.iloc[np.random.permutation(len(df))]

    print(df[['beginposition', 'size']])
    print('Maximum cloud percentage: {}%'.format(args.skip_cloudy_percentage))
    print('Total products: {}'.format(len(df)))
    print('Total size: {:.2f}GB'.format(size))
    if args.output is not None:
        if args.split == 0:
            df.to_csv(args.output)
        else:
            os.makedirs(args.output, exist_ok=True)
            files = [None] * args.split
            for i, product_id in enumerate(df.index):
                i_file = i % args.split
                if files[i_file] is None:
                    files[i_file] = open(os.path.join(
                        args.output, str(i_file+1) + '.txt'), 'w')
                f = files[i_file]
                f.write(product_id + '\n')
            for file in files:
                if file is not None:
                    file.close()
