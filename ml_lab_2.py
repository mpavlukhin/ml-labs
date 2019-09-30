import argparse

import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='ML Lab 2'
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


def get_gaussian_EM(X):
    gmm = GaussianMixture(n_components=3).fit(X)
    labels = gmm.predict(X)

    return labels


def show_plot(X, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')

    plt.show()


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    blobs_points_sample = generate_blobs_points_sample(args.size, args.centers)
    predicted_labels = get_gaussian_EM(blobs_points_sample)

    show_plot(blobs_points_sample, predicted_labels)


if __name__ == "__main__":
    main()
