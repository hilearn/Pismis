"""
Query products for given footprint and time filter'
"""

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import datetime
import os
from argparse import ArgumentParser
from config import USERNAME, PASSWORD


def parse_arguments():
    parser = ArgumentParser(description='Query products for given footprint '
                                        'and time filter',
                            epilog='example: python satquery.py '
                                   'example.geojson -f 01-08-2017 -o o.csv')
    parser.add_argument('geojson', help='geojson file for fotprint')
    parser.add_argument('-f', '--from', help='query products starting from'
                        'this date, date format: DD-MM-YYYY', default=None)
    parser.add_argument('-t', '--to', help='query products until with this'
                        'date, date format: DD-MM-YYYY', default=None)
    parser.add_argument('-o', '--output', help='output into *.csv file',
                        default=None)
    return parser.parse_args()


def parse_date(date):
    return datetime.strptime(date, "%d-%m-%Y")


def satquery(geojson, date_from=None, date_to=None):
    """
    Args:
        geojson: str
            The geojson file path for footprint.
        date_from: datetime, optional
        date_to: datetime, optional
            The time interval filter based on the
            Sensing Date of the products

    Returns:
        Return the products from a query response as a Pandas DataFrame
        with the values in their appropriate Python types.
    """

    api = SentinelAPI(USERNAME, PASSWORD, 'https://scihub.copernicus.eu/dhus')

    footprint = geojson_to_wkt(read_geojson(geojson), decimals=6)
    products = api.query(footprint, date=(date_from, date_to),
                         area_relation='Contains', platformname='Sentinel-2')
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

    df = satquery(args.geojson, date_from, date_to)

    size = 0
    for x in df['size']:
        val = float(x[:-3])
        if x[-2:] == 'GB':
            size += val
        elif x[-2:] == 'MB':
            size += val / 1024
    print(df[['beginposition', 'size']])
    print('Total size: %0.2f' % size + ' GB')
    if args.output is not None:
        df.to_csv(args.output)
