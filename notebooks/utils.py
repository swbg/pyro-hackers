import os
import sys
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Ellipse


def plot_samples(samples, figsize=(10, 4)):
    """
    Plot MCMC samples.
    """
    fig, ax = plt.subplots(ncols=2, figsize=figsize)

    ax[0].hist(samples, bins=30, color=sns.color_palette()[0])
    ax[0].set_ylabel("Number of samples")
    ax[0].set_xlabel("Value")
    ax[0].set_title("Posterior distribution")

    ax[1].plot(np.arange(len(samples)), samples, color=sns.color_palette()[1])
    ax[1].set_ylabel("Value")
    ax[1].set_xlabel("Iteration")
    ax[1].set_title("Sample trace")

    plt.tight_layout()
    plt.show()


def plot_galaxies(galaxies, size=50, halos=None, true_halo=None):
    """
    Plot galaxies from the Dark Worlds competition, adapted from
    https://github.com/CamDavidsonPilon/
    Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    for x, y, e1, e2 in galaxies:
        d = np.sqrt(e1 ** 2 + e2 ** 2)
        a = 1.0 / (1 - d)
        b = 1.0 / (1 + d)
        theta = np.degrees(np.arctan2(e2, e1) * 0.5)

        ax.add_patch(
            Ellipse(
                xy=(x, y),
                width=size * a,
                height=size * b,
                angle=theta,
                color=sns.color_palette()[0],
            )
        )

    ax.autoscale_view(tight=True)

    if true_halo is not None:
        ax.scatter(
            [true_halo[0]],
            [true_halo[1]],
            color=sns.color_palette()[2],
            zorder=999
        )

    if halos is not None:
        for halo in halos:
            ax.scatter(
                [halo[0]],
                [halo[1]],
                color=sns.color_palette()[1],
                alpha=0.1
            )


@contextmanager
def stdout_disabled():
    """
    Temporarily disable all output, see http://thesmithfam.org/blog/2012/10/25/
    temporarily-suppress-console-output-in-python/
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
