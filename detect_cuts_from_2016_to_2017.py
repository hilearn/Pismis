import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import ast
from argparse import ArgumentParser
from osgeo import gdal
import os
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib

matplotlib.rcParams.update({'font.size': 30})


def parse_arguments():
    parser = ArgumentParser(description='Show deforestation points on image.')
    parser.add_argument('probabilities', help='csv or pickle file of porest'
                                              ' probabilities.')
    parser.add_argument('--image', help='sample image', required=True)
    return parser.parse_args()


def plot_deforestation_ponts(df, image, save_dir=None, show=False, c=0.6,
                             plot=True):
    """
    Plot probability map, high probability points on real image, scatter and 2D
    histogram of pixel-probabilities

    :param df: Dataframe. You can get it by df = forest_probabilities_TS(path)
    :param image: Colored image to visually check our result
    :param save_dir: Where you'd like to save plots
    :param show: Do you want to show plots or not
    :param c: Threshold of probabilities
    :param plot: Plot or not
    :return: Dataframe of probabilities
    """
    start2016, end2016 = datetime(2016, 1, 1), datetime(2016, 12, 31)
    start2017, end2017 = datetime(2017, 1, 1), datetime(2017, 12, 31)

    s2016 = df[(df.index > start2016) & (df.index < end2016)].mean(axis=0)
    s2017 = df[(df.index > start2017) & (df.index < end2017)].mean(axis=0)
    ss = pd.DataFrame([s2016, s2017], index=[2016, 2017]).T.dropna()
    prob_mask = ss[2016] - ss[2016] * ss[2017]

    deforestation_points = ss.index[
        (prob_mask > c)]

    print('Points: {}'.format(len(deforestation_points)))

    image = gdal.Open(image).ReadAsArray().transpose(1, 2, 0)

    if plot:
        pred = (prob_mask).values.reshape(image.shape[0], image.shape[1])

        # PLOT PROBABILITY MASK************************************************
        plt.figure(figsize=(30, 10), dpi=200)
        plt.subplot(122)
        plt.axis("off")

        plt.imshow(image)
        plt.scatter([y for x, y in deforestation_points],
                    [x for x, y in deforestation_points], s=1, c="Orange")
        plt.suptitle("Deforestation Probability")

        plt.subplot(121)
        plt.axis("off")
        plt.imshow(pred, cmap="hot_r")
        plt.colorbar(fraction=0.037, pad=0.04)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "deforestatin.png"),
                        bbox_inches='tight')

        # PLOT FOREST PROBABILITIES********************************************
        plt.figure(figsize=(20, 10), dpi=200)
        y = np.linspace(0, 1 - c, 100)
        x = c / (1 - y)
        plt.subplot(121)
        plt.margins(0)
        plt.plot(x, y, color="red")
        plt.scatter(ss[2016], ss[2017], s=0.1)
        plt.suptitle("Forest Probabilities of pixels")

        plt.subplot(122)
        plt.axis("off")
        plt.hist2d(ss[2016], ss[2017], norm=LogNorm(), bins=12)
        plt.colorbar(fraction=0.037, pad=0.04)

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "scatter.png"),
                        bbox_inches='tight')

        if show:
            plt.show()
    return ss


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
