import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



"""
Generate barplot of tau values for each iteration
In axis x, number of iterations
In axis y, tau_x and tau_y values
Specify the absolute path
"""
def barplot_tau(tau_x, tau_y, num_iterations, path=".", filename="barplot_tau", ylabel="tau"):
    
    axis_x = np.arange(num_iterations)

    axis_y = {
        ylabel+"_x": tau_x,
        ylabel+"_y": tau_y
    }

    x = np.copy(axis_x)  # the label locations
    
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    width = 0.25
    multiplier = 0

    for attribute, measurement in axis_y.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        #ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel(ylabel)
    ax.set_xlabel('iterations')
    ax.set_xticks(x + width, axis_x)
    ax.legend(loc='upper right')
    
    plt.savefig(str(path + f"/{filename}.png"), transparent=False, facecolor='white', dpi=100)
    #plt.show()


# line plot
def plot_tau(tau_x, tau_y, num_iterations, save=True, path=".", filename="plot_tau"):
    axis_x = np.arange(num_iterations)
    plt.plot(axis_x, tau_x, label="tau_x")
    plt.plot(axis_x, tau_y, label="tau_y")
    plt.title("Co-clustering quality")
    plt.xlabel("iteration")
    plt.ylabel("tau values")
    plt.legend()
    if save:
        if os.path.exists(path):
            plt.savefig(str(path + "/" + filename + ".png"), transparent=False, facecolor='white', dpi=300)
        else:
            raise ValueError(f"No such file or directory: {path}")
    else:
        plt.show()


def plot_coclus(X, label_row, label_col, save=True, path=".", filename="plot_coclus", title="Co-clustering"):
    fig = plt.figure()
    row_indices = np.argsort(label_row)
    col_indices = np.argsort(label_col)
    X_reorg = X[row_indices, :]
    X_reorg = X_reorg[:, col_indices]
    #plt.spy(X_reorg, aspect="auto")
    #plt.spy(X_reorg, precision=precision, markersize=markersize, aspect=aspect, gapcolor=gapcolor, color=color)
    plt.imshow(X_reorg, aspect="auto")
    plt.xlabel("cols")
    plt.ylabel("rows")
    plt.title(title)
    # plt.imshow(X_reorg, interpolation="nearest")
    plot_coclus_remove_ticks()
    if save:
        plt.savefig(str(path + "/" + filename + ".png"), format='png', dpi=300)
    else:
        plt.show()


def plot_coclus_remove_ticks():
    plt.tick_params(axis='both', which='both', bottom='off', top='off',right='off', left='off', labelsize=12)


def create_combined_plot(n_rows=2, n_cols=5, fig_size=(15, 6), image_files=None, save=True, path=".", filename="combined_plot"):
    # Create a 2x5 grid of subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)

    # Loop through each subplot and load the corresponding image
    for ax, img_file in zip(axes.flatten(), image_files):
        img = mpimg.imread(img_file)  # Read the image file
        ax.imshow(img)  # Display the image
        ax.axis('off')  # Hide axes

    # Adjust layout
    plt.tight_layout()

    if save:
        plt.savefig(f"{path}/{filename}.png", bbox_inches='tight', dpi=300)
    else:
        plt.show()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", xlabel_txt="xlabel", ylabel_txt="ylabel", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels)
    # ,rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    ax.set_xlabel(xlabel_txt)
    ax.set_ylabel(ylabel_txt)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts