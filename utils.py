import time
import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt


def set_size(width='thesis', fraction=1, subplots=(1, 1), height_add=0):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = height_add + fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


def setup_plt():
    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        # "font.family": "times",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 12,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    }
    plt.rcParams.update(tex_fonts)


def get_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S",
                         datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-7))).timetuple())


def count_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def pad_leading_zeros(segment, length):
    assert len(segment) <= length

    if len(segment) < length:
        padding_length = length - len(segment)
        if isinstance(segment, np.ndarray):
            padding = np.zeros(padding_length)
            segment = np.concatenate((padding, segment))
        elif isinstance(segment, torch.Tensor):
            padding = torch.zeros(padding_length)
            segment = torch.concatenate((padding, segment))

    return segment


if __name__ == '__main__':
    ...
