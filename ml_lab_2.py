import argparse

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='This script uses EM algorithm '
                    'for defining Gaussian clusters '
                    'and show plot for visualization'
    )

    parser.add_argument('-s', '--size', type=int, default='500', help='An integer for sample size')

    parser.add_argument(
        '-c', '--centers', type=int, default='3', help='An integer for number of blobs points sample centers'
    )

    return parser


def generate_blobs_points_sample(sample_size, blobs_centers_number):
    X, y_true = make_blobs(n_samples=sample_size, centers=blobs_centers_number,
                           cluster_std=.5, random_state=0)
    X = X[:, ::-1]

    return X


def get_gaussian_EM(X, clusters_num):
    gmm = GaussianMixture(n_components=clusters_num, covariance_type='full').fit(X)
    labels = gmm.predict(X)

    return labels, gmm


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def show_plot(X, labels, gmm):
    ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

    plt.show()


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    blobs_points_sample = generate_blobs_points_sample(args.size, args.centers)
    predicted_labels, gmm = get_gaussian_EM(blobs_points_sample, args.centers)

    show_plot(blobs_points_sample, predicted_labels, gmm)


if __name__ == "__main__":
    main()
