import matplotlib.ticker as ticker
import numpy as np
from matplotlib import pyplot as plt


def get_data(shuffle=False):
    x_train = np.loadtxt("./data/1d_x.txt", delimiter=",").reshape(-1, 1)
    y_train = np.loadtxt("./data/1d_y.txt", delimiter=",").reshape(-1, 1)
    num_train = len(y_train)
    batch_size = num_train // 3
    # Move the first third of the data to the left by 1
    x_train[:batch_size, :] -= 1
    # Move the second third of the data to the right by 1
    x_train[2 * batch_size:3 * batch_size, :] += 1
    if shuffle:
        indices = np.random.permutation(x_train)
        x_train = x_train[indices, :]
        y_train = y_train[indices, :]
    x_test = np.linspace(-2.0, 12.0, 100)[:, None]
    return x_train, y_train, x_test, batch_size


def plot_result(
    ax,
    x_train,
    y_train,
    x_test,
    mean,
    var,
    noise_variance,
    x_pseudo=None,
    f_pseudo=None,
    x_old=None,
    y_old=None,
    plot_legend=False,
):
    x_test = x_test.ravel()
    mean = mean.ravel()
    std = np.sqrt(var.ravel())
    noise_std = np.sqrt(noise_variance)

    ax.plot(x_train, y_train, "kx", mew=1, alpha=0.8, label="Training")
    if x_old is not None and y_old is not None:
        ax.plot(x_old, y_old, "kx", mew=1, alpha=0.2, label="Old")
    ax.plot(x_test, mean, "b", lw=2, label="Mean")
    ax.fill_between(
        x_test.ravel(),
        mean - 2 * std,
        mean + 2 * std,
        color="b",
        alpha=0.3,
        label="±2σ(f)",
    )
    ax.fill_between(
        x_test.ravel(),
        mean - 2 * std - 2 * noise_std,
        mean + 2 * std + 2 * noise_std,
        color="c",
        alpha=0.2,
        label="±2σ(y)",
    )
    if x_pseudo is not None and f_pseudo is not None:
        ax.plot(x_pseudo, f_pseudo, "ro", mew=1, alpha=0.8, label="Pseudo")
    ax.set_ylim([-3.0, 3.0])
    ax.set_xlim([np.min(x_test), np.max(x_test)])
    plt.subplots_adjust(hspace=0.08)
    ax.set_ylabel("y")
    ax.yaxis.set_ticks(np.arange(-2, 3, 1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
    if plot_legend:
