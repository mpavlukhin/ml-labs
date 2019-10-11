import argparse

import matplotlib.pyplot as plt
import numpy as np

from math import sqrt, pi, exp
from scipy import optimize, integrate


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='This script uses uniform distribution '
                    'for sampling complex distribution '
                    '(1 / (2 * sqrt(2 * pi)) * (exp(-(x - mu) ** 2 / 2) + exp(-(x + mu) ** 2 / 2)) '
                    'and show plot for visualization'
    )

    parser.add_argument(
        '-a', '--accuracy', type=int, default='1000', help='An integer for sampling accuracy'
    )
    parser.add_argument(
        '-s', '--start', type=int, default='-10', help='An integer for start point of sampling'
    )
    parser.add_argument(
        '-e', '--end', type=int, default='10', help='An integer for end point of sampling'
    )
    parser.add_argument(
        '-i', '--interval', type=float, default='0.1', help='An integer for step width of sampling'
    )

    return parser


def uniform_pdf(count, left_bound=0, right_bound=1):
    return np.random.uniform(left_bound, right_bound, int(count))


def density_for_test(x, mu=5):
    return 1 / (2 * sqrt(2 * pi)) * (exp(-(x - mu) ** 2 / 2) + exp(-(x + mu) ** 2 / 2))


def sample_distribution(accuracy, start_point, end_point):
    uniform_points = uniform_pdf(accuracy)

    return list(
        map(
            lambda uniform_point: sample_distribution_point(
                uniform_point, density_for_test, start_point, end_point
            ),
            uniform_points
        )
    )


def sample_distribution_point(point, density, start_point, end_point):
    x = optimize.bisect(
        lambda x: integrate.quad(density, -np.inf, x)[0] - point, start_point, end_point
    )
    return x


def show_plot(x, y, step):
    plt.plot(x, np.array(list(map(density_for_test, x))), 'go-', label="sampled")
    plt.hist(y, bins=int(step * 1000), normed=True)

    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.grid(True)
    plt.legend(loc=0, fontsize=20)

    plt.show()


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    x = np.arange(args.start, args.end, args.interval)
    y = sample_distribution(args.accuracy, args.start, args.end)

    show_plot(x, y, args.interval)


if __name__ == "__main__":
    main()
