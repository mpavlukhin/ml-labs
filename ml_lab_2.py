import argparse

from sklearn.datasets.samples_generator import make_blobs


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


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    blobs_points_sample = generate_blobs_points_sample(args.size, args.centers)


if __name__ == "__main__":
    main()
