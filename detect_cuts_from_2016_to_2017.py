import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import ast
from argparse import ArgumentParser
from osgeo import gdal


def parse_arguments():
    parser = ArgumentParser(description='Show deforestation points on image.')
    parser.add_argument('csv', help='csv file of porest probabilities.')
    parser.add_argument('--image', help='sample image')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    df = pd.read_csv(args.csv, index_col=0)
    df.columns = [ast.literal_eval(x) for x in df.columns]
    df.index = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df.index]
    start2016, end2016 = datetime(2016, 1, 1), datetime(2016, 12, 31)
    start2017, end2017 = datetime(2017, 1, 1), datetime(2017, 12, 31)

    df2016 = df[(df.index > start2016) & (df.index < end2016)].mean(axis=0)
    df2017 = df[(df.index > start2017) & (df.index < end2017)].mean(axis=0)

    deforestation_points = []
    for point in df.columns:
        x, y = df2016[point], df2017[point]
        if x > 0.5 and y < 0.5:
            deforestation_points.append(point)
    plt.figure(figsize=(15, 15))
    if args.image is not None:
        image = gdal.Open(args.image).ReadAsArray().transpose(1, 2, 0)
        plt.imshow(image)
    plt.scatter([y for x, y in deforestation_points],
                [x for x, y in deforestation_points], s=5, alpha=0.6)
    plt.show()
