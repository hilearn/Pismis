from osgeo import gdal
from argparse import ArgumentParser
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from cloud_mask import get_cloud_mask_path
from sklearn import mixture


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
    f_NIR = os.path.join(path,
                         [x for x in os.listdir(path) if x[-8:-5] == 'B08'][0])
    f_RED = os.path.join(path,
                         [x for x in os.listdir(path) if x[-8:-5] == 'B04'][0])
    NIR = gdal.Open(f_NIR).ReadAsArray()
    RED = gdal.Open(f_RED).ReadAsArray()
    return (NIR.astype('float') - RED.astype('float')) / (
        NIR.astype('float') + RED.astype('float'))


def debug_forest_probability(path):
    """
    Shows image, cloud mask, probabilities and NDVI distribution.
    :param path: str, path to product
    """
    f_TCI1 = os.path.join(path, [x for x in os.listdir(path)
                                 if x[-8:-5] == 'CI1'][0])
    TCI1 = gdal.Open(f_TCI1).ReadAsArray().transpose(1, 2, 0)
    ndvi = NDVI(path)
    mask = get_cloud_mask_path(path + '/')
    P, D = forest_probability(ndvi, mask, missing_values=0.5)

    X = np.linspace(np.min(ndvi), np.max(ndvi), 100)
    Y = (multi_normal_pdf(X, D['means'][0], D['covariances'][0]) *
         D['weights'][0] +
         multi_normal_pdf(X, D['means'][1], D['covariances'][1]) *
         D['weights'][1])

    plt.figure(figsize=(18, 5))

    plt.subplot(141)
    plt.imshow(TCI1)

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
