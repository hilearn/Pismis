from osgeo import gdal
from argparse import ArgumentParser
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from cloud_mask import cloud_mask
from sklearn import mixture
import pandas as pd
import json
from utils import timestamp_to_datetime
from utils import band_name
from utils import Bands
from scipy.signal import convolve2d
from mix import resize_band


def parse_arguments():
    parser = ArgumentParser(description='Save Forest Probabilities Time Series'
                                        ' for each pixel to csv file')
    parser.add_argument('path', help='path to preprocessed products')
    parser.add_argument('output', help='Output file name')
    parser.add_argument('--float-precision', type=int, default=3,
                        help='Number of digits after the decimal point')
    parser.add_argument('--mean-convolve', action='store_true',
                        help='Do mean convolution on NDVI')
    return parser.parse_args()


def multi_normal_pdf(x, mean, covariance):
    """
    Evaluates Multivariate Gaussian Distribution density function
    :param x: location where to evaluate the density function
    :param mean: Center of the Gaussian Distribution
    :param covariance: Covariance of the Gaussian Distribution
    :return: density function evaluated at point x
    """
    var = multivariate_normal(mean=mean, cov=covariance)
    return var.pdf(x)


def forest_probability(ndvi, mask=None, predict_ndvi=None,
                       missing_values=np.NaN):
    """
    Estimates forest probability for each pixel using Gaussian Mixture model.
    :param ndvi: array, ndvi values for each pixel
    :param mask: array of ones and zeros.
        uses pixel information, if mask is one for that pixel.
    :param missing_values: float. set this value for pixels with zero mask.
    :param predict_ndvi: array, If given predict probabilities for this array.
    :return: (array, dict),
        probability of being forest for avery pixel and some information about
        modeled distribution (i.e. means, weights, variances)
    """
    if mask is None:
        mask = np.ones_like(ndvi)

    if mask.shape != ndvi.shape:
        raise Exception('Wrong mask dimension')

    if predict_ndvi is None:
        predict_ndvi = ndvi

    gmm = mixture.GaussianMixture(n_components=2)
    gmm.fit(ndvi[mask == 1].reshape(-1, 1))
    covariances = gmm.covariances_[:]
    pixel_cluster_proba = gmm.predict_proba(
        predict_ndvi.reshape(-1, 1)).reshape(ndvi.shape[0], ndvi.shape[1], 2)
    weights = gmm.weights_.copy()
    means = gmm.means_.copy()

    P = pixel_cluster_proba[..., 1]
    if means[0] > means[1]:
        P = pixel_cluster_proba[..., 0]

    P[mask == 0] = missing_values
    return P, {
        'probalilities': pixel_cluster_proba,
        'weights': weights,
        'means': means,
        'covariances': covariances
    }


def NDVI(path):
    """
    Calculates NDVI for product
    :param path: str, path to product.
    :return: array, NDVI
    """
    NIR = gdal.Open(band_name(path, Bands.NIR)).ReadAsArray()
    RED = gdal.Open(band_name(path, Bands.RED)).ReadAsArray()
    size = max(NIR.shape, RED.shape)
    NIR = resize_band(NIR, size[0], size[1])
    RED = resize_band(RED, size[0], size[1])
    return (NIR.astype('float') - RED.astype('float')) / (
        NIR.astype('float') + RED.astype('float'))


def forest_probabilities_TS(data_path, convolve=False):
    """
    Return time series for every pixel.
    :param data_path: str, path to processed data.
    :return: Pandas Dataframe, TS for every pixel.

    Example. (Column names are pixel indices)
                               (0, 0)        (0, 1)        (0, 2) ...
    2015-12-29 08:03:50  1.951377e-05  2.943721e-06  2.627551e-05 ...
    2016-01-18 08:06:16  9.143812e-05           NaN           NaN ...
    2016-03-08 07:53:03  2.009606e-01  6.967517e-02  1.160830e-01 ...
    2016-04-07 07:54:33           NaN           NaN           NaN ...
    """
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size))/kernel_size**2
    df_list = [], []
    shape = None
    for product in os.listdir(data_path):
        if product.startswith('product') is False:
            continue
        path = os.path.join(data_path, product)
        if os.path.isdir(path) is False:
            continue
        info = json.load(open(os.path.join(path, 'info.json')))
        ndvi = NDVI(path)
        ndvi_convolved = ndvi
        if convolve:
            ndvi_convolved = convolve2d(ndvi, kernel,
                                        mode='full', boundary='symm')

        mask = cloud_mask(path)
        P, _ = forest_probability(ndvi, mask, ndvi_convolved,
                                  missing_values=np.NaN)
        if shape is None:
            shape = P.shape
        if shape != P.shape:
            raise Exception('Products have different resolution')

        df_list[0].append(P.flatten())
        df_list[1].append(timestamp_to_datetime(info['Sensing start']))

    df = pd.DataFrame(df_list[0], index=df_list[1],
                      columns=list(np.ndindex(shape))).sort_index()
    return df


def debug_forest_probability(path):
    """
    Shows image, cloud mask, probabilities and NDVI distribution.
    :param path: str, path to product
    """

    TCI = gdal.Open(band_name(
        path, Bands.TCI)).ReadAsArray().transpose(1, 2, 0)
    ndvi = NDVI(path)
    mask = cloud_mask(path + '/')
    P, D = forest_probability(ndvi, mask, missing_values=0.5)

    X = np.linspace(np.min(ndvi), np.max(ndvi), 100)
    Y = (multi_normal_pdf(X, D['means'][0], D['covariances'][0]) *
         D['weights'][0] +
         multi_normal_pdf(X, D['means'][1], D['covariances'][1]) *
         D['weights'][1])

    plt.figure(figsize=(18, 5))

    plt.subplot(141)
    plt.imshow(TCI)

    plt.subplot(142)
    plt.imshow(mask)

    plt.subplot(143)
    plt.imshow(P, cmap='gray')

    plt.subplot(144)
    plt.hist(ndvi[mask == 1].flatten(), bins=100, normed=True)
    plt.plot(X, Y)
    plt.show()


if __name__ == '__main__':
    args = parse_arguments()
    df = forest_probabilities_TS(args.path)
    output = args.output
    if output.endswith('.csv') is True:
        df.to_csv(output, float_format='%0.{}f'.format(args.float_precision))
    else:
        df.to_pickle(output)
