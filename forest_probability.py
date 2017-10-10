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
from utils import resize_band
from utils import read_array
from scipy.signal import convolve as convolution


def parse_arguments():
    parser = ArgumentParser(description='Save Forest Probabilities Time Series'
                                        ' for each pixel to csv file')
    parser.add_argument('path', help='path to preprocessed products')
    parser.add_argument('--output', help='Output file name', required=True)
    parser.add_argument('--float-precision', type=int, default=3,
                        help='Number of digits after the decimal point')
    parser.add_argument('--mean-convolve', action='store_true',
                        help='Do mean convolution on indices')
    parser.add_argument('--gmm-components', type=int, choices=[2, 3],
                        default=3, help='Number of gaussians to fit.')
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


def NDVI(path, size=None):
    """
    Calculates NDVI for product
    :param path: str, path to product.
    :param size: (height, width), output size.
        If not given, use size of bands.
    :return: array, NDVI
    """
    NIR = read_array(band_name(path, Bands.NIR))
    RED = read_array(band_name(path, Bands.RED))
    if size is None:
        size = max(NIR.shape, RED.shape)
    NIR = resize_band(NIR, size)
    RED = resize_band(RED, size)
    return (NIR.astype('float') - RED.astype('float')) / (
        NIR.astype('float') + RED.astype('float'))


def index_tensor(path, size=None):
    """
    Calculates 13 different deforestation indices
    :param path: str, path to product.
    :param size: (height, width), output size.
        If not given, use maximal size of bands.
    :return: array (size, 13)
    """
    B02 = read_array(band_name(path, Bands.B02))
    if size is None:
        size = B02.shape
    B02 = resize_band(B02, size).astype('float32')
    B03 = resize_band(read_array(band_name(path, Bands.B03)),
                      size).astype('float32')
    B04 = resize_band(read_array(band_name(path, Bands.B04)),
                      size).astype('float32')
    B05 = resize_band(read_array(band_name(path, Bands.B05)),
                      size).astype('float32')
    B06 = resize_band(read_array(band_name(path, Bands.B06)),
                      size).astype('float32')
    B08 = resize_band(read_array(band_name(path, Bands.B08)),
                      size).astype('float32')
    B11 = resize_band(read_array(band_name(path, Bands.B11)),
                      size).astype('float32')
    # B12 = resize_band(read_array(band_name(path, Bands.B12)),
    #                   size).astype('float32')
    # All indices take two classes corresponding to ground and forest
    # all indices are written such that obtained class distribution with
    # greater mean value corresponds to forest
    index = [
        ('NDVI', (B08 - B04) / (B08 + B04)),
        # ('SR', B08 / B04),
        ('MSI', -(B11 / B08)),
        ('NDVI705', (B06 - B05) / (B06 + B05)),
        # ('SAVI', 1.5 * (B08 - B04) / (B08 + B04 + 0.5)),
        # ('PSRI-NIR', (B04 - B02) / B08),
        ('PSRI', -(B04 - B02) / B05),
        # ('NBR-RAW', (B08 - B12) / (B08 + B12)),
        # ('MSAVI2', (B08 + 1) - 0.5 * np.sqrt((2 * B08 - 1) ** 2 + 8 * B04)),
        # ('LAI-SAVI',
        #  -np.log(0.371 + 1.5 * (B08 - B04) / (B08 + B04 + 0.5)) / 2.4),
        ('GRVI1', -(B04 - B03) / (B04 + B03)),
        # ('GNDVI', (B08 - B03) / (B08 + B03)),
        # ('EVI2', 2.5 * (B08 - B04) / (B08 + 2.4 * B04 + 1)),
        # ('NDMI', (B08 - B11) / (B11 + B08))
    ]
    return np.stack([x[1] for x in index], axis=-1)


def forest_probability(index, mask=None, predict_index=None,
                       missing_values=np.NaN, n_components=2):
    """
    Estimates forest probability for each pixel using Gaussian Mixture model.
    :param index: array, ndvi values for each pixel
    :param mask: array of ones and zeros.
        uses pixel information, if mask is one for that pixel.
    :param missing_values: float. set this value for pixels with zero mask.
    :param predict_index: array, If given predict probabilities for this array.
    :param n_components: int, number of components for mixture model
    :return: (array, dict),
        probability of being forest for avery pixel and some information about
        modeled distribution (i.e. means, weights, variances)
    """
    if mask is None:
        mask = np.ones(index.shape[:2])
    if mask.shape != index.shape[:2]:
        raise Exception('Wrong mask dimension')

    if predict_index is None:
        predict_index = index

    gmm = mixture.GaussianMixture(n_components=n_components)
    gmm.fit(index[mask == 1].reshape(-1, index.shape[-1]))
    pixel_cluster_proba = gmm.predict_proba(predict_index.reshape(
        -1, index.shape[-1])).reshape(*index.shape[:2], n_components)

    covariances = gmm.covariances_[:]
    weights = gmm.weights_.copy()
    means = gmm.means_.copy()

    P = pixel_cluster_proba[..., weights.argmax()]

    P[mask == 0] = missing_values
    return P, {
        'probalilities': pixel_cluster_proba,
        'weights': weights,
        'means': means,
        'covariances': covariances
    }


def forest_probabilities_TS(data_path, convolve=False, n_components=3):
    """
    Return time series for every pixel.
    :param data_path: str, path to processed data.
    :param convolve: boolean, Do mean convolution on indices
    :param n_components: int, number of components for mixture model
    :return: Pandas Dataframe, TS for every pixel.

    Example. (Column names are pixel indices)
                               (0, 0)        (0, 1)        (0, 2) ...
    2015-12-29 08:03:50  1.951377e-05  2.943721e-06  2.627551e-05 ...
    2016-01-18 08:06:16  9.143812e-05           NaN           NaN ...
    2016-03-08 07:53:03  2.009606e-01  6.967517e-02  1.160830e-01 ...
    2016-04-07 07:54:33           NaN           NaN           NaN ...
    """
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
    df_list = [], []
    shape = None
    for product in os.listdir(data_path):
        if product.startswith('product') is False:
            continue
        path = os.path.join(data_path, product)
        if os.path.isdir(path) is False:
            continue
        print('\t' + product)
        info = json.load(open(os.path.join(path, 'info.json')))
        # index = NDVI(path)[..., None]
        index = index_tensor(path)

        index_convolved = index
        if convolve:
            index_convolved = convolution(index, kernel[..., None],
                                          mode='full', boundary='symm')
        mask = cloud_mask(path)
        P, _ = forest_probability(index, mask, index_convolved,
                                  missing_values=np.NaN,
                                  n_components=n_components)
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
    TCI = read_array(band_name(
        path, Bands.TCI)).transpose(1, 2, 0)
    ndvi = NDVI(path)[..., None]
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
    df = forest_probabilities_TS(args.path, convolve=args.mean_convolve,
                                 n_components=args.gmm_components)
    output = args.output
    if output.endswith('.csv') is True:
        df.to_csv(output, float_format='%0.{}f'.format(args.float_precision))
    else:
        df.to_pickle(output)
