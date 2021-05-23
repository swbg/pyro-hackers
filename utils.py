import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_samples(samples, figsize=(10, 4)):
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
