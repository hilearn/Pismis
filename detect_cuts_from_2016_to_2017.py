from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from argparse import ArgumentParser
from datetime import datetime
from utils import read_array
import pandas as pd
import numpy as np
import ast
import os


def parse_arguments():
    parser = ArgumentParser(description='Show deforestation points on image.')
    parser.add_argument('probabilities', help='csv or pickle file of porest'
                                              ' probabilities.')
    parser.add_argument('--image', help='sample image', required=True)
    parser.add_argument('--save-plots', default=None,
                        help='directory to save plotted images')
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
    ss = pd.DataFrame([s2016, s2017], index=[2016, 2017]).T
    prob_mask = ss[2016] * (1 - ss[2017])

    deforestation_points = ss.dropna().index[(prob_mask.dropna() > c)]

    print('Points: {}'.format(len(deforestation_points)))

    image = read_array(image).transpose(1, 2, 0)

    if plot:
        pred = prob_mask.values.reshape(image.shape[0], image.shape[1])

        # plot probability mask
        plt.figure(figsize=(10, 4.1))
        plt.subplot(122)
        plt.axis("off")

        plt.imshow(image)
        plt.scatter([y for x, y in deforestation_points],
                    [x for x, y in deforestation_points], s=0.5, c="Orange",
                    lw=0)
        plt.suptitle("Deforestation Probability", fontsize=20)

        plt.subplot(121)
        plt.axis("off")
        plt.imshow(pred, cmap="hot_r")
        plt.colorbar(fraction=0.037, pad=0.04)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "deforestatin.png"),
                        bbox_inches='tight', dpi=300)

        # plot forest probabilities
        plt.figure(figsize=(10, 4.5))
        y = np.linspace(0, 1 - c, 100)
        x = c / (1 - y)
        plt.subplot(121)
        plt.margins(0)
        plt.plot(x, y, color="red")
        plt.scatter(ss[2016], ss[2017], s=1.5, lw=0, alpha=0.8)
        plt.suptitle("Forest Probabilities of pixels", fontsize=20)

        plt.subplot(122)
        plt.axis("off")
        plt.hist2d(ss.dropna()[2016], ss.dropna()[2017], norm=LogNorm(),
                   bins=12)
        plt.colorbar(fraction=0.037, pad=0.04)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "scatter.png"),
                        bbox_inches='tight', dpi=300)

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

    plot_deforestation_ponts(df, args.image, save_dir=args.save_plots)
    plt.show()
