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


def forest_probability(ndvi, mask=None, missing_values=np.NaN):
    """
    Estimates forest probability for each pixel using Gaussian Mixture model.
    :param ndvi: array
    :param mask: array of ones and zeros.
        uses pixel information, if mask is one for that pixel.
    :param missing_values: float. set this value for pixels with zero mask.
    :return: (array, dict),
        probability of being forest for avery pixel and some information about
        modeled distribution (i.e. means, weights, variances)
    """
    if mask is None:
        mask = np.ones_like(ndvi)
    if mask.shape != ndvi.shape:
        raise Exception('Wrong mask dimension')

    gmm = mixture.GaussianMixture(n_components=2)
    gmm.fit(ndvi[mask == 1].reshape(-1, 1))
    covariances = gmm.covariances_[:]
    pixel_cluster_proba = gmm.predict_proba(ndvi.reshape(-1, 1)).reshape(
        ndvi.shape[0], ndvi.shape[1], 2)
    weights = gmm.weights_.copy()
    means = gmm.means_.copy()

    P_NF, P_F = weights[0], weights[1]
    P_NFcNF, P_FcF = pixel_cluster_proba[..., 0], pixel_cluster_proba[..., 1]

    if means[0] > means[1]:
        P_NF, P_F = P_F, P_NF
        P_NFcNF, P_FcF = P_FcF, P_NFcNF

    P = P_FcF * P_F + (1 - P_NFcNF) * P_NF
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
    return (NIR.astype('float') - RED.astype('float')) / (
        NIR.astype('float') + RED.astype('float'))


def forest_probabilities_TS(data):
    """
    Return time series for every pixel.
    :param data: str, path to processed data.
    :return: Pandas Dataframe, TS for every pixel.

    Example. (Column names are pixel indices)
                               (0, 0)        (0, 1)        (0, 2) ...
    2015-12-29 08:03:50  1.951377e-05  2.943721e-06  2.627551e-05 ...
    2016-01-18 08:06:16  9.143812e-05           NaN           NaN ...
    2016-03-08 07:53:03  2.009606e-01  6.967517e-02  1.160830e-01 ...
    2016-04-07 07:54:33           NaN           NaN           NaN ...
    """
    df_list = [], []
    shape = None
    for product in os.listdir(data):
        if product.startswith('product') is False:
            continue
        path = os.path.join(data, product)
        if os.path.isdir(path) is False:
            continue
        info = json.load(open(os.path.join(path, 'info.json')))
        ndvi = NDVI(path)
        mask = cloud_mask(path)
        P, _ = forest_probability(ndvi, mask, missing_values=np.NaN)
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


def parse_arguments():
    parser = ArgumentParser(description='Shows image, cloud mask, '
                                        'probabilities and NDVI distribution,')
    parser.add_argument('path', help='path to product')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    debug_forest_probability(args.path)