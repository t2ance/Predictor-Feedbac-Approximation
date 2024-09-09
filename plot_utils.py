from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from dynamic_systems import ConstantDelay
from utils import SimulationResult

colors = ['red', 'green', 'blue', 'orange', 'black', 'cyan', 'magenta', 'white', 'pink', 'yellow', 'gray', 'lightblue',
          'lightgreen', 'purple', 'brown', 'teal', 'olive', 'navy', 'lime', 'coral', 'salmon', 'aqua', 'wheat']
styles = ['-', '--', '-.', ':']
legend_loc = 'best'
# legend_loc = 'lower right'
display_threshold = 1
fig_width = 469.75502
n_ticks = 5


def plot_switched_control(ts, result: SimulationResult, n_point_delay, ylim=None, ax=None, comment=True):
    if ax is None:
        ax = plt.figure(figsize=set_size(width=fig_width)).gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
    U, switching_indicator = result.U, result.switching_indicator
    ts = ts[n_point_delay:]
    U = U[n_point_delay:]
    n_input = U.shape[-1]
    switching_indicator = switching_indicator[n_point_delay:]
    marked_indices = np.where(np.logical_xor(switching_indicator[:-1], switching_indicator[1:]))[0]
    marked_indices = np.insert(marked_indices, 0, 0)
    color_labels = ['$U_{NO}$', '$U_{Numerical}$', 'Switch Point']
    styles = ['-', ':']
    for j in range(U.shape[-1]):
        u = U[:, j]
        # ignore the zero
        ax.scatter(ts[marked_indices][1:], u[marked_indices][1:], s=25, color=colors[j], facecolors='none',
                   linewidth=0.5)

        ax.plot(ts[:marked_indices[0] + 1], u[:marked_indices[0] + 1], linestyle=styles[1], label=color_labels[0],
                color=colors[j])
        for i in range(len(marked_indices) - 1):
            ax.plot(ts[marked_indices[i]:marked_indices[i + 1] + 1], u[marked_indices[i]:marked_indices[i + 1] + 1],
                    linestyle=styles[i % 2], label=color_labels[(i + 1) % 2] if i == 0 else "", color=colors[j])
        ax.plot(ts[marked_indices[-1]:], u[marked_indices[-1]:], linestyle=styles[(len(marked_indices) + 1) % 2],
                color=colors[j])
    if ylim is not None:
        try:
            ax.set_ylim(ylim)
        except:
            ...
    if comment:
        # ax.set_xlabel('Time t')
        ax.legend(loc=legend_loc)
        if n_input < display_threshold:
            ax.legend(loc=legend_loc)
        else:
            ax.legend(handles=[Line2D([0], [0], color='black', linestyle=styles[0]),
                               Line2D([0], [0], color='black', linestyle=styles[1]),
                               Line2D([0], [0], color='black', marker='o')
                               ],
                      labels=color_labels, loc=legend_loc)


def plot_control(ts, U, save_path, n_point_delay, ylim=None, ax=None, comment=True, figure=None, linestyle='-'):
    if ax is None:
        figure = plt.figure(figsize=set_size(width=fig_width))
        ax = figure.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

    assert U.ndim == 2
    n_point_start = n_point_delay(0)
    U = U.T
    for i, u in enumerate(U):
        ax.plot(ts[n_point_start:], u[n_point_start:], label=f'$U_{i + 1}(t)$', color=colors[i], linestyle=linestyle)
    if ylim is not None:
        try:
            ax.set_ylim(ylim)
        except:
            ...
    if comment:
        # ax.set_xlabel('Time t')
        ax.legend(loc=legend_loc)

    if figure is not None and save_path is not None:
        figure.savefig(save_path)
        figure.clear()
        plt.close(figure)


def plot_quantile(n_point_start, P_no_Ri, cp_alpha, ax, ylim=None, comment=False, legend_loc='best'):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
    Q = np.percentile(P_no_Ri[2 * n_point_start:], (1 - cp_alpha) * 100)
    ax.hist(P_no_Ri[2 * n_point_start:], bins=100, color='blue', alpha=0.7, label='$R$')
    ax.axvline(x=Q, color='red', linestyle='--', label=f'{(1 - cp_alpha) * 100}% quantile: {Q:.2f}')
    if comment:
        ax.legend(loc=legend_loc)
    if ylim is not None:
        try:
            ax.set_ylim(ylim)
        except:
            ...
    # ax.title(f'Distribution of $R$ and the ${1 - cp_alpha}$ quantile')
    # ax.set_xlabel('$R$')
    # ax.set_ylabel('frequency')


def plot_switch_system(train_config, dataset_config, result: SimulationResult, n_point_delay: int, img_save_path: str,
                       ax=None):
    if ax is None:
        ax = plt.figure(figsize=set_size(width=fig_width)).gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

    Q = np.percentile(result.P_no_Ri[2 * n_point_delay:], (1 - train_config.uq_alpha) * 100)
    ax.hist(result.P_no_Ri[2 * n_point_delay:], bins=100, color='blue', alpha=0.7, label='$R$')
    ax.axvline(x=Q, color='red', linestyle='--',
               label=f'{(1 - train_config.uq_alpha) * 100}% quantile: {Q:.2f}')
    ax.legend(loc=legend_loc)
    # ax.title(f'Distribution of $R$ and the ${1 - train_config.uq_alpha}$ quantile')
    ax.set_xlabel('$R$')
    ax.set_ylabel('frequency')
    # ax.savefig(f'{img_save_path}/quantile.png')
    # ax.close()

    ax.plot(dataset_config.ts[2 * n_point_delay:], result.q_ts[2 * n_point_delay:], label='$q_t$')
    ax.set_xlabel('Time t')
    ax.set_ylabel('$q_t$')
    ax.legend(loc=legend_loc)
    ax.savefig(f'{img_save_path}/q.png')
    ax.close()

    ax.plot(dataset_config.ts[2 * n_point_delay:], result.P_no_Ri[2 * n_point_delay:], label='$R_t$')
    ax.set_xlabel('Time t')
    ax.set_ylabel('$R_t$')
    ax.legend(loc=legend_loc)
    ax.savefig(f'{img_save_path}/quantile_time.png')
    ax.close()

    ax.plot(dataset_config.ts[2 * n_point_delay:], result.alpha_ts[2 * n_point_delay:], label='$\\alpha_t$')
    ax.set_xlabel('Time t')
    ax.set_ylabel('$\\alpha$')
    ax.legend(loc=legend_loc)
    ax.savefig(f'{img_save_path}/alpha.png')
    ax.close()

    ax.plot(dataset_config.ts[2 * n_point_delay:], result.e_ts[2 * n_point_delay:], label='$e_t$')
    ax.set_xlabel('Time t')
    ax.set_ylabel('$e_t$')
    ax.legend(loc=legend_loc)
    ax.savefig(f'{img_save_path}/e.png')
    ax.close()

    ax.plot(dataset_config.ts[2 * n_point_delay:], result.switching_indicator[2 * n_point_delay:],
            label='$\\mathbb{I}_t$')
    ax.set_xlabel('Time t')
    ax.set_ylabel('$\\mathbb{I}_t$')
    ax.legend(loc=legend_loc)
    ax.savefig(f'{img_save_path}/I.png')
    ax.close()

    plot_switched_control(dataset_config.ts, result, n_point_delay)


def plot_result(dataset_config, img_save_path, P_no, P_numerical, P_explicit, P_switching, Z, U,
                method: Literal['explicit', 'numerical', 'no', 'numerical_no', 'switching', 'scheduled_sampling']):
    if img_save_path is None:
        return
    ts = dataset_config.ts
    delay = dataset_config.delay
    n_point_delay = dataset_config.n_point_delay
    comparison_full = f'{img_save_path}/{method}_comp_fit.png'
    difference_full = f'{img_save_path}/{method}_diff_fit.png'
    comparison_zoom = f'{img_save_path}/{method}_comp.png'
    difference_zoom = f'{img_save_path}/{method}_diff.png'
    u_path = f'{img_save_path}/{method}_u.png'
    if method == 'explicit':
        plot_comparison(ts, [P_explicit], Z, delay, n_point_delay, comparison_full)
        plot_difference(ts, [P_explicit], Z, n_point_delay, difference_full)
        plot_comparison(ts, [P_explicit], Z, delay, n_point_delay, comparison_zoom, ylim=[-5, 5])
        plot_difference(ts, [P_explicit], Z, n_point_delay, difference_zoom, ylim=[-5, 5])
        plot_control(ts, U, u_path, n_point_delay)
    elif method == 'no' or method == 'scheduled_sampling':
        plot_comparison(ts, [P_no], Z, delay, n_point_delay, comparison_full)
        plot_difference(ts, [P_no], Z, n_point_delay, difference_full)
        plot_comparison(ts, [P_no], Z, delay, n_point_delay, comparison_zoom, ylim=[-5, 5])
        plot_difference(ts, [P_no], Z, n_point_delay, difference_zoom, ylim=[-5, 5])
        plot_control(ts, U, u_path, n_point_delay)
    elif method == 'numerical':
        plot_comparison(ts, [P_numerical], Z, delay, n_point_delay, comparison_full)
        plot_difference(ts, [P_numerical], Z, n_point_delay, difference_full)
        plot_comparison(ts, [P_numerical], Z, delay, n_point_delay, comparison_zoom, ylim=[-5, 5])
        plot_difference(ts, [P_numerical], Z, n_point_delay, difference_zoom, ylim=[-5, 5])
        plot_control(ts, U, u_path, n_point_delay)
    elif method == 'numerical_no':
        plot_comparison(ts, [P_numerical, P_no], Z, delay, n_point_delay, comparison_full,
                        Ps_labels=['numerical', 'no'])
        plot_difference(ts, [P_numerical, P_no], Z, n_point_delay, difference_full, Ps_labels=['numerical', 'no'])
        plot_comparison(ts, [P_numerical, P_no], Z, delay, n_point_delay, comparison_zoom,
                        Ps_labels=['numerical', 'no'], ylim=[-5, 5])
        plot_difference(ts, [P_numerical, P_no], Z, n_point_delay, difference_zoom, Ps_labels=['numerical', 'no'],
                        ylim=[-5, 5])
        plot_control(ts, U, u_path, n_point_delay)
    elif method == 'switching':
        plot_comparison(ts, [P_switching], Z, delay, n_point_delay, comparison_full)
        plot_difference(ts, [P_switching], Z, n_point_delay, difference_full)
        plot_comparison(ts, [P_switching], Z, delay, n_point_delay, comparison_zoom, ylim=[-5, 5])
        plot_difference(ts, [P_switching], Z, n_point_delay, difference_zoom, ylim=[-5, 5])
        plot_control(ts, U, u_path, n_point_delay)
    else:
        raise NotImplementedError()


def plot_q(ts, qs, q_des, save_path, ylim=None, ax=None, comment=True, figure=None):
    if ax is None:
        figure = plt.figure(figsize=set_size(width=fig_width))
        ax = figure.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
    n_state = q_des.shape[-1]

    for i in range(n_state):
        for j, q in enumerate(qs):
            ax.plot(ts[:], q[:, i], linestyle=styles[j], color=colors[i],
                    label=f'$q_{i + 1}(t)$')
        ax.plot(ts, q_des[:, i], label=f'$q_{{des,{i + 1}}}(t)$', linestyle='-.',
                color=colors[i])
    if ylim is not None:
        try:
            ax.set_ylim(ylim)
        except:
            ...
    if comment:
        # ax.set_xlabel('Time t')
        if n_state < display_threshold:
            ax.legend(loc=legend_loc)
        else:
            ax.legend(handles=[Line2D([0], [0], color='black', linestyle='-.'),
                               Line2D([0], [0], color='black', linestyle='-')],
                      labels=[f'$q(t)$', f'$q_{{des}}(t)$'], loc=legend_loc)
    if figure is not None and save_path is not None:
        figure.savefig(save_path)
        figure.clear()
        plt.close(figure)


def plot_comparison(ts, Ps, Z, delay, n_point_delay, save_path, ylim=None, Ps_labels=None, ax=None,
                    comment=True, figure=None):
    if ax is None:
        figure = plt.figure(figsize=set_size(width=fig_width))
        ax = figure.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

    n_state = Z.shape[-1]
    n_point_start = n_point_delay(0)
    if Ps_labels is None:
        Ps_labels = ['' for _ in range(len(Ps))]
    Ps_ = []
    for P in Ps:
        P_ = np.zeros_like(Z[n_point_start:])
        for ti, t in enumerate(ts[n_point_start:]):
            P_[ti] = P[ti + n_point_start - n_point_delay(t)]
        Ps_.append(P_)
    Ps = Ps_

    delay_label = str(delay(0)) if isinstance(delay, ConstantDelay) else 'D(t)'

    for i in range(n_state):
        for j, (P, label) in enumerate(zip(Ps, Ps_labels)):
            ax.plot(ts[2 * n_point_start:], P[n_point_start:, i], linestyle='--', color=colors[i],
                    label=f'$P^{{{label}}}_{i + 1}(t-{delay_label})$')
        ax.plot(ts[n_point_start:], Z[n_point_start:, i], label=f'$Z_{i + 1}(t)$', linestyle='-', color=colors[i])
    if ylim is not None:
        try:
            ax.set_ylim(ylim)
        except:
            ...

    if comment:
        # ax.set_xlabel('Time t')
        if n_state < display_threshold:
            ax.legend(loc=legend_loc)
        else:
            ax.legend(handles=[Line2D([0], [0], color='black', linestyle='--'),
                               Line2D([0], [0], color='black', linestyle='-')],
                      labels=[f'$P(t-{delay_label})$', f'$Z(t)$'], loc=legend_loc)
    if figure is not None and save_path is not None:
        figure.savefig(save_path)
        figure.clear()
        plt.close(figure)


def difference(Z, P, n_point_start, n_point_delay, ts):
    P_ = np.zeros_like(Z[n_point_start:])
    for ti, t in enumerate(ts[n_point_start:]):
        P_[ti] = P[ti + n_point_start - n_point_delay(t)]
    return P_ - Z[n_point_start:]


def plot_difference(ts, Ps, Z, n_point_delay, save_path, ylim=None, Ps_labels=None, ax=None,
                    comment=True, differences=None, figure=None):
    if ax is None:
        figure = plt.figure(figsize=set_size(width=fig_width))
        ax = figure.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
    n_state = Z.shape[-1]

    if Ps_labels is None:
        Ps_labels = ['' for _ in range(len(Ps))]
    n_point_start = n_point_delay(0)
    if differences is None:
        differences = []
        for P in Ps:
            differences.append(difference(Z, P, n_point_start, n_point_delay, ts))

    for i in range(n_state):
        for j, (d, label) in enumerate(zip(differences, Ps_labels)):
            ts_ = ts[n_point_start:-n_point_start] if n_point_start != 0 else ts[n_point_start:]
            ax.plot(ts_, abs(d[n_point_start:, i]), linestyle=styles[j], color=colors[i],
                    label=f'$\Delta P^{{{label}}}_{i + 1}(t)$')

    if ylim is not None:
        try:
            ax.set_yscale('log')
            # y_locator = LogLocator(base=10.0, numticks=5)
            # ax.yaxis.set_major_locator(y_locator)
            # ax.yaxis.set_major_formatter(LogFormatterMathtext())
            # ax.set_ylim([0, ylim[1]])
            ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
            # ax.set_ylim([0, 100])
        except:
            ...
    if comment:
        # ax.set_xlabel('Time t')
        if n_state < display_threshold:
            ax.legend(loc=legend_loc)
        else:
            ax.legend(handles=[Line2D([0], [0], color='black', linestyle='-')],
                      labels=[f'$\Delta P(t)$'], loc=legend_loc)
    if figure is not None and save_path is not None:
        figure.savefig(save_path)
        figure.clear()
        plt.close(figure)


def set_size(width=None, fraction=1, subplots=(1, 1), height_add=0.1):
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
    elif width is None:
        width_pt = fig_width
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
