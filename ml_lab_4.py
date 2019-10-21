import numpy as np
import matplotlib.pyplot as plt


DESCRIPTION = '''
This script uses rejection sampling for sampling complex distribution
np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2)) / (2 * np.sqrt(2 * np.pi) * sigma1) +
np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))
/ (2 * np.sqrt(2 * np.pi) * sigma2)
and show plot for visualization
'''

MU = -0.5
SIGMA = 6


def get_complex_distr_func():
    mu1, sigma1, mu2, sigma2 = 1, 0.5, -5, 1

    return (
        lambda x: (
                np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2)) / (2 * np.sqrt(2 * np.pi) * sigma1) +
                np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))
                / (2 * np.sqrt(2 * np.pi) * sigma2)
        )
    )


def get_normal_distr_for_sampling():
    mu, sigma = MU, SIGMA
    const = 6.5

    return (
        lambda x: const * (np.exp(-(x - mu) ** 2 / sigma ** 2) / (np.sqrt(2 * np.pi) * sigma))
    )


def get_sampling_batch(distr_to_sample, sampling_distr):
    batch = np.random.normal(loc=MU, scale=SIGMA, size=10000)

    new_batch = []
    for i in batch:
        if np.random.uniform(high=sampling_distr(i)) <= distr_to_sample(i):
            new_batch.append(i)

    return new_batch


def show_plot(*funcs):
    for fun in funcs:
        plt.plot([x for x in range(-1000, 1000)], [fun(x / 100) for x in range(-1000, 1000)])

    plt.show()


def show_hist(batch):
    plt.hist(batch, bins=100)
    plt.show()


def main():
    print(DESCRIPTION)

    distr_to_sample = get_complex_distr_func()
    sampling_normal_distr = get_normal_distr_for_sampling()
    show_plot(distr_to_sample, sampling_normal_distr)

    new_batch = get_sampling_batch(distr_to_sample, sampling_normal_distr)
    show_hist(new_batch)


if __name__ == "__main__":
    main()
