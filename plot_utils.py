from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from config import DatasetConfig
from dynamic_systems import ConstantDelay
from utils import check_dir, SimulationResult

colors = ['red', 'green', 'blue', 'orange', 'black', 'cyan', 'magenta', 'white', 'pink', 'yellow', 'gray', 'lightblue',
          'lightgreen', 'purple', 'brown', 'teal', 'olive', 'navy', 'lime', 'coral', 'salmon', 'aqua', 'wheat']
styles = ['-', '--', '-.', ':']
legend_loc = 'best'
# legend_loc = 'lower right'
display_threshold = 1
fig_width = 469.75502
n_ticks = 5


def plot_distribution(samples, n_state: int, img_save_path: str = None):
    u_list = []
    z_list = []
    p_list = []
    p_z_ratio_list = []
    for feature, p in samples:
        feature = feature.cpu().numpy()
        z = feature[1:1 + n_state]
        u = feature[1 + n_state:]
        p = p.cpu().numpy()
        u_list += u.tolist()
        z_list.append(z)
        p_list.append(p)
        p_z_ratio_list.append(np.linalg.norm(p) / np.linalg.norm(z))
    bins = 100
    alpha = 0.5

    def draw_1d(t, title, file_name, xlabel='data', ylabel='density', xlim=None, ax=None):
        if xlim is None:
            xlim = [-5, 5]
        ax.hist(t, bins=bins, density=True, alpha=alpha)
        ax.title(title)
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if n_state < display_threshold:
            ax.legend(loc='upper left')
        if img_save_path is not None:
            ax.savefig(f'{img_save_path}/{file_name}')
            ax.clf()
        else:
            ...

    draw_1d(p_z_ratio_list, r'$\frac{||P||_2}{||Z||_2}$', 'p_z.png', xlim=[-2, 2])
    draw_1d(u_list, 'U', 'u.png')
    zs = np.array(z_list)
    ps = np.array(p_list)
    for i in range(n_state):
        draw_1d(zs[:, i], f'$Z_{i}$', f'z{i}.png')
        draw_1d(ps[:, i], f'$P_{i}$', f'p{i}.png', xlim=[-1, 1])


def plot_sample(feature, label, dataset_config: DatasetConfig, name: str = '1.png', ax=None):
    if ax is None:
        ax = plt.figure(figsize=set_size(width=fig_width)).gca()
    if isinstance(feature, torch.Tensor):
        feature = feature.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    n_state = dataset_config.system.n_state
    n_input = dataset_config.system.n_input
    t = feature[:1]
    z = feature[1:1 + n_state]
    u = feature[1 + n_state:].reshape(-1, n_input)
    p = label
    ts = np.linspace(t - dataset_config.delay, t, dataset_config.n_point_delay)
    ax.plot(ts, u, label='U')
    for i in range(n_state):
        ax.scatter(ts[-1], z[i], label=f'$Z_t({i})$', c=colors[i])
        ax.scatter(ts[-1], p[i], label=f'$P_t({i})$', c=colors[i], marker='^')
    if n_state < display_threshold:
        ax.legend(loc='upper left')
    out_dir = f'{dataset_config.dataset_base_path}/sample'
    check_dir(out_dir)
    ax.savefig(f'{out_dir}/{name}')
    ax.clf()


def plot_system(title, ts, Z, U, P, img_save_path, ax=None):
    if ax is None:
        ax = plt.figure(figsize=set_size(width=fig_width)).gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
    ax.title(title)

    ax.subplot(511)
    ax.set_ylim([-5, 5])
    ax.plot(ts, Z[:, 0], label='$Z_1(t)$')
    ax.set_ylabel('$Z_1(t)$')
    ax.grid(True)

    ax.subplot(512)
    ax.set_ylim([-5, 5])
    ax.plot(ts, Z[:, 1], label='$Z_2(t)$')
    ax.set_ylabel('$Z_2(t)$')
    ax.grid(True)

    ax.subplot(513)
    ax.set_ylim([-5, 5])
    ax.plot(ts, U, label='$U(t)$', color='black')
    ax.set_xlabel('time')
    ax.set_ylabel('$U(t)$')
    ax.grid(True)

    ax.subplot(514)
    ax.set_ylim([-5, 5])
    ax.plot(ts, P[:, 0], label='$P_1(t)$')
    ax.set_ylabel('$P_1(t)$')
    ax.grid(True)

    ax.subplot(515)
    ax.set_ylim([-5, 5])
    ax.plot(ts, P[:, 1], label='$P_2(t)$')
    ax.set_ylabel('$P_2(t)$')
    ax.grid(True)
    ax.tight_layout()
    ax.savefig(f'{img_save_path}/system.png')
    ax.clear()


def plot_switch_segments(ts, result: SimulationResult, n_point_delay, ylim=None, ax=None, comment=True):
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

        ax.plot(ts[:marked_indices[0] + 1], u[:marked_indices[0] + 1], linestyle=styles[0], label=color_labels[0],
                color=colors[j])
        for i in range(len(marked_indices) - 1):
            ax.plot(ts[marked_indices[i]:marked_indices[i + 1] + 1], u[marked_indices[i]:marked_indices[i + 1] + 1],
                    linestyle=styles[(i + 1) % 2], label=color_labels[(i + 1) % 2] if i == 0 else "", color=colors[j])
        ax.plot(ts[marked_indices[-1]:], u[marked_indices[-1]:], linestyle=styles[(len(marked_indices)) % 2],
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

    plot_switch_segments(dataset_config.ts, result, n_point_delay)


def plot_result(dataset_config, img_save_path, P_no, P_numerical, P_explicit, P_switching, Z, U,
                method: Literal['explicit', 'numerical', 'no', 'numerical_no', 'switching', 'scheduled_sampling']):
    if img_save_path is None:
        return
    ts = dataset_config.ts
    delay = dataset_config.delay
    n_point_delay = dataset_config.n_point_delay
    n_state = dataset_config.n_state
    comparison_full = f'{img_save_path}/{method}_comp_fit.png'
    difference_full = f'{img_save_path}/{method}_diff_fit.png'
    comparison_zoom = f'{img_save_path}/{method}_comp.png'
    difference_zoom = f'{img_save_path}/{method}_diff.png'
    u_path = f'{img_save_path}/{method}_u.png'
    if method == 'explicit':
        plot_comparison(ts, [P_explicit], Z, delay, n_point_delay, comparison_full, n_state)
        plot_difference(ts, [P_explicit], Z, delay, n_point_delay, difference_full, n_state)
        plot_comparison(ts, [P_explicit], Z, delay, n_point_delay, comparison_zoom, n_state, ylim=[-5, 5])
        plot_difference(ts, [P_explicit], Z, delay, n_point_delay, difference_zoom, n_state, ylim=[-5, 5])
        plot_control(ts, U, u_path, n_point_delay)
    elif method == 'no' or method == 'scheduled_sampling':
        plot_comparison(ts, [P_no], Z, delay, n_point_delay, comparison_full, n_state)
        plot_difference(ts, [P_no], Z, delay, n_point_delay, difference_full, n_state)
        plot_comparison(ts, [P_no], Z, delay, n_point_delay, comparison_zoom, n_state, ylim=[-5, 5])
        plot_difference(ts, [P_no], Z, delay, n_point_delay, difference_zoom, n_state, ylim=[-5, 5])
        plot_control(ts, U, u_path, n_point_delay)
    elif method == 'numerical':
        plot_comparison(ts, [P_numerical], Z, delay, n_point_delay, comparison_full, n_state)
        plot_difference(ts, [P_numerical], Z, delay, n_point_delay, difference_full, n_state)
        plot_comparison(ts, [P_numerical], Z, delay, n_point_delay, comparison_zoom, n_state, ylim=[-5, 5])
        plot_difference(ts, [P_numerical], Z, delay, n_point_delay, difference_zoom, n_state, ylim=[-5, 5])
        plot_control(ts, U, u_path, n_point_delay)
    elif method == 'numerical_no':
        plot_comparison(ts, [P_numerical, P_no], Z, delay, n_point_delay, comparison_full, n_state,
                        Ps_labels=['numerical', 'no'])
        plot_difference(ts, [P_numerical, P_no], Z, delay, n_point_delay, difference_full, n_state,
                        Ps_labels=['numerical', 'no'])
        plot_comparison(ts, [P_numerical, P_no], Z, delay, n_point_delay, comparison_zoom, n_state,
                        Ps_labels=['numerical', 'no'], ylim=[-5, 5])
        plot_difference(ts, [P_numerical, P_no], Z, delay, n_point_delay, difference_zoom, n_state,
                        Ps_labels=['numerical', 'no'], ylim=[-5, 5])
        plot_control(ts, U, u_path, n_point_delay)
    elif method == 'switching':
        plot_comparison(ts, [P_switching], Z, delay, n_point_delay, comparison_full, n_state)
        plot_difference(ts, [P_switching], Z, delay, n_point_delay, difference_full, n_state)
        plot_comparison(ts, [P_switching], Z, delay, n_point_delay, comparison_zoom, n_state, ylim=[-5, 5])
        plot_difference(ts, [P_switching], Z, delay, n_point_delay, difference_zoom, n_state, ylim=[-5, 5])
        plot_control(ts, U, u_path, n_point_delay)
    else:
        raise NotImplementedError()


def plot_q(ts, qs, q_des, save_path, n_state: int, ylim=None, ax=None, comment=True, figure=None):
    if ax is None:
        figure = plt.figure(figsize=set_size(width=fig_width))
        ax = figure.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

    for i in range(n_state):
        for j, q in enumerate(qs):
            ax.plot(ts[:], q[:, i], linestyle=styles[j], color=colors[i],
                    label=f'$q_{i + 1}(t)$')
        ax.plot(ts[:], q_des[:, i], label=f'$q_{{des,{i + 1}}}(t)$', linestyle='-.',
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


def plot_comparison(ts, Ps, Z, delay, n_point_delay, save_path, n_state: int, ylim=None, Ps_labels=None, ax=None,
                    comment=True, figure=None):
    if ax is None:
        figure = plt.figure(figsize=set_size(width=fig_width))
        ax = figure.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

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


def plot_difference(ts, Ps, Z, delay, n_point_delay, save_path, n_state: int, ylim=None, Ps_labels=None, ax=None,
                    comment=True, differences=None, figure=None):
    if ax is None:
        figure = plt.figure(figsize=set_size(width=fig_width))
        ax = figure.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

    if Ps_labels is None:
        Ps_labels = ['' for _ in range(len(Ps))]
    n_point_start = n_point_delay(0)
    if differences is None:
        differences = []
        for P in Ps:
            differences.append(difference(Z, P, n_point_start, n_point_delay, ts))

    for i in range(n_state):
        for j, (d, label) in enumerate(zip(differences, Ps_labels)):
            ax.plot(ts[n_point_start:-n_point_start], d[n_point_start:, i], linestyle=styles[j], color=colors[i],
                    label=f'$\Delta P^{{{label}}}_{i + 1}(t)$')

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
            ax.legend(handles=[Line2D([0], [0], color='black', linestyle='-')],
                      labels=[f'$\Delta P(t)$'], loc=legend_loc)
    if figure is not None and save_path is not None:
        figure.savefig(save_path)
        figure.clear()
        plt.close(figure)


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
