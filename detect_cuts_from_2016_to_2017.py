import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import ast
from argparse import ArgumentParser
from osgeo import gdal
import os


def parse_arguments():
    parser = ArgumentParser(description='Show deforestation points on image.')
    parser.add_argument('probabilities', help='csv or pickle file of porest'
                                              ' probabilities.')
    parser.add_argument('--image', help='sample image', required=True)
    return parser.parse_args()


def plot_deforestation_ponts(df, image, plot=True):
    start2016, end2016 = datetime(2016, 1, 1), datetime(2016, 12, 31)
    start2017, end2017 = datetime(2017, 1, 1), datetime(2017, 12, 31)

    df2016 = df[(df.index > start2016) & (df.index < end2016)].mean(axis=0)
    df2017 = df[(df.index > start2017) & (df.index < end2017)].mean(axis=0)
    deforestation_points = df2016.index[
        (df2016 - df2017 > 1 - 0.55)]

    print('Points: {}'.format(len(deforestation_points)))

    image = gdal.Open(image).ReadAsArray().transpose(1, 2, 0)
    if plot:
        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.scatter([y for x, y in deforestation_points],
                    [x for x, y in deforestation_points], s=1, alpha=0.6)

        plt.figure(figsize=(6, 6))
        plt.imshow(
            (df2016 - df2017).values.reshape(image.shape[0], image.shape[1]))
        plt.colorbar()

        plt.figure(figsize=(6, 6))
        plt.scatter(df2016, df2017, s=0.1, c='gray')
        plt.scatter(df2016[deforestation_points], df2017[deforestation_points],
                    s=0.1, c='orange')
    return df2016, df2017


if __name__ == '__main__':
    args = parse_arguments()
    if os.path.exists(args.probabilities) is False:
        raise Exception('Probabilities file does not exist.')
    if args.probabilities.endswith('.csv'):
        df = pd.read_csv(args.probabilities, index_col=0)
        df.columns = [ast.literal_eval(x) for x in df.columns]
        df.index = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in
                    df.index]
    else:
        df = pd.read_pickle(args.probabilities)
    plot_deforestation_ponts(df, args.image)
    plt.show()
