import matplotlib.pyplot as plt
import numpy as np


DESCRIPTION = '''
This script uses Metropolis Hastings for sampling complex distribution
np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2)) / (2 * np.sqrt(2 * np.pi) * sigma1) +
np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))
/ (2 * np.sqrt(2 * np.pi) * sigma2)
and show plot for visualization
'''


def complex_distr_func(x):
    mu1, sigma1, mu2, sigma2 = 1, 0.5, -5, 1

    return (
        np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2)) / (2 * np.sqrt(2 * np.pi) * sigma1) +
        np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))
        / (2 * np.sqrt(2 * np.pi) * sigma2)
    )


def metropolis_hastings(iters=10000):
    x = 0
    s = 10

    p = complex_distr_func(x)
    q = np.random.normal

    samples = []
    for i in range(0, iters):
        xn = x + q(size=1)
        pn = complex_distr_func(xn)
        if pn >= p:
            p = pn
            x = xn
        else:
            u = np.random.rand()
            if u < pn/p:
                p = pn
                x = xn
        if i % s == 0:
            samples.append(x)

    return np.array(samples)


def show_plot(x_array, y_array, samples):
    plt.title('Metropolis Hastings visualization')
    plt.scatter(samples, np.zeros_like(samples), s=10)
    plt.grid()

    plt.plot(x_array, y_array)
    plt.hist(samples, bins=50, normed=True)

    plt.show()


def main():
    print(DESCRIPTION)

    x_array = np.linspace(-15.0, 15.0, 100)
    y_array = np.asarray([complex_distr_func(x) for x in x_array])
    samples = metropolis_hastings()

    show_plot(x_array, y_array, samples)


if __name__ == "__main__":
    main()