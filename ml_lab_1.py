import argparse

import numpy
import pylab

import matplotlib.animation as animation


ROOT_TWO_PI = (2 * numpy.pi) ** 0.5
PARZEN_EST = lambda x, X, h, sigma: numpy.exp(-(X - x) ** 2 / (2 * h ** 2 * sigma ** 2)).sum() / (
        ROOT_TWO_PI * h * sigma * X.size)

GAUSSIAN_PDF = lambda x: (1 / (2 * numpy.pi) ** .5) * numpy.exp(-x ** 2 / 2)


def uniform_pdf(x):
    p = numpy.zeros(x.size)
    p[(-2 < x) & (x < 2)] = 0.25

    return p


UNIFORM_PDF = uniform_pdf


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='This script uses Parzen window density estimation '
                    'for building probability distributions '
                    'and saves animation to file'
    )

    parser.add_argument('-s', '--size', type=int, default='200', help='An integer for N (sample size)')
    parser.add_argument('-d', '--distr', type=str, default='gaussian', help='Distribution type (uniform, gaussian)')

    return parser


def init_coordinate_plane():
    fig = pylab.figure(figsize=(3, 6))
    ax = pylab.axes(xlim=(-5, 5), ylim=(-0.5, 0.7))

    return fig, ax


def get_distr(distr, sample_size):
    if distr == 'uniform':
        r_start = r_start=numpy.random.rand(sample_size)*4-2
        return r_start, UNIFORM_PDF

    elif distr == 'gaussian':
        r_start = numpy.random.randn(sample_size)
        return r_start, GAUSSIAN_PDF

    raise ValueError('You must provide distribution name (gaussian, uniform)')


def animate(i, r, ax, distr):
    this_r = r[:i + 2]

    ax.cla()

    h = 1.06 * this_r.std() * this_r.size ** (-.2)
    lim = [-5, 5]
    bins = 51
    x = numpy.linspace(lim[0], lim[1], num=100)

    pylab.text(-.4, -.025, 'n=' + str(this_r.size))

    pylab.hist(
        this_r, bins=bins, range=lim, normed=True, edgecolor=(.9, .9, .9), color=(.8, .8, .8), histtype='stepfilled'
    )

    pylab.plot(x, distr(x), color=(.1, .1, 1.0), lw=5)
    pylab.plot(x, [PARZEN_EST(xx, this_r, h, this_r.std()) for xx in x], 'k', lw=3)

    pylab.setp(ax, xlim=(-5, 5), ylim=(-0.05, 0.7))


def save_anim(fig, r_start, ax, distr, sample_size, filename):
    anim = animation.FuncAnimation(
        fig, animate, fargs=(r_start, ax, distr), frames=sample_size-2, interval=2, repeat=False
    )
    anim.save(filename + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    distr_name = args.distr

    r_start, distr = get_distr(distr_name, args.size)
    fig, ax = init_coordinate_plane()
    save_anim(fig, r_start, ax, distr, args.size, distr_name)


if __name__ == "__main__":
    main()
