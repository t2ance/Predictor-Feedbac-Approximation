from dynamic_systems import Delay, ConstantDelay
from utils import check_dir, SimulationResult, shift
from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt

from config import DatasetConfig

colors = ['red', 'green', 'blue', 'yellow', 'black', 'cyan', 'magenta', 'white', 'pink', 'orange', 'gray', 'lightblue',
          'lightgreen', 'purple', 'brown', 'teal', 'olive', 'navy', 'lime', 'coral', 'salmon', 'aqua', 'wheat']
styles = ['-', '--', '-.', ':']
legend_loc = 'upper right'
legend_fontsize = 8
fig_width = 469.75502


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

    def draw_1d(t, title, file_name, xlabel='data', ylabel='density', xlim=None):
        if xlim is None:
            xlim = [-5, 5]
        plt.hist(t, bins=bins, density=True, alpha=alpha)
        plt.title(title)
        plt.xlim(xlim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if n_state < 5:
            plt.legend(loc='upper left')
        if img_save_path is not None:
            plt.savefig(f'{img_save_path}/{file_name}')
            plt.clf()
        else:
            plt.show()

    draw_1d(p_z_ratio_list, r'$\frac{||P||_2}{||Z||_2}$', 'p_z.png', xlim=[-2, 2])
    draw_1d(u_list, 'U', 'u.png')
    zs = np.array(z_list)
    ps = np.array(p_list)
    for i in range(n_state):
        draw_1d(zs[:, i], f'$Z_{i}$', f'z{i}.png')
        draw_1d(ps[:, i], f'$P_{i}$', f'p{i}.png', xlim=[-1, 1])


def plot_sample(feature, label, dataset_config: DatasetConfig, name: str = '1.png'):
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
    plt.plot(ts, u, label='U')
    for i in range(n_state):
        plt.scatter(ts[-1], z[i], label=f'$Z_t({i})$', c=colors[i])
        plt.scatter(ts[-1], p[i], label=f'$P_t({i})$', c=colors[i], marker='^')
    if n_state < 5:
        plt.legend(loc='upper left')
    out_dir = f'{dataset_config.dataset_base_path}/sample'
    check_dir(out_dir)
    plt.savefig(f'{out_dir}/{name}')
    plt.clf()


def plot_system(title, ts, Z, U, P, img_save_path):
    fig = plt.figure(figsize=set_size(width=fig_width))
    plt.title(title)

    plt.subplot(511)
    plt.ylim([-5, 5])
    plt.plot(ts, Z[:, 0], label='$Z_1(t)$')
    plt.ylabel('$Z_1(t)$')
    plt.grid(True)

    plt.subplot(512)
    plt.ylim([-5, 5])
    plt.plot(ts, Z[:, 1], label='$Z_2(t)$')
    plt.ylabel('$Z_2(t)$')
    plt.grid(True)

    plt.subplot(513)
    plt.ylim([-5, 5])
    plt.plot(ts, U, label='$U(t)$', color='black')
    plt.xlabel('time')
    plt.ylabel('$U(t)$')
    plt.grid(True)

    plt.subplot(514)
    plt.ylim([-5, 5])
    plt.plot(ts, P[:, 0], label='$P_1(t)$')
    plt.ylabel('$P_1(t)$')
    plt.grid(True)

    plt.subplot(515)
    plt.ylim([-5, 5])
    plt.plot(ts, P[:, 1], label='$P_2(t)$')
    plt.ylabel('$P_2(t)$')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{img_save_path}/system.png')
    fig.clear()
    plt.close(fig)


def plot_switch_segments(ts, result: SimulationResult, save_path, n_point_delay):
    U, switching_indicator = result.U, result.switching_indicator
    ts = ts[n_point_delay:]
    U = U[n_point_delay:]
    switching_indicator = switching_indicator[n_point_delay:]
    fig = plt.figure(figsize=set_size(width=fig_width))
    marked_indices = np.where(np.logical_xor(switching_indicator[:-1], switching_indicator[1:]))[0]

    color_labels = ['$U_{NO}$', '$U_{Numerical}$']
    for j in range(U.shape[-1]):
        u = U[:, j]
        plt.plot(ts[:marked_indices[0] + 1], u[:marked_indices[0] + 1], linestyle=styles[1], label=color_labels[0],
                 color=colors[j])
        for i in range(len(marked_indices) - 1):
            plt.plot(ts[marked_indices[i]:marked_indices[i + 1] + 1], u[marked_indices[i]:marked_indices[i + 1] + 1],
                     linestyle=styles[i % 2], label=color_labels[(i + 1) % 2] if i == 0 else "", color=colors[j])
        plt.plot(ts[marked_indices[-1]:], u[marked_indices[-1]:], linestyle=styles[(len(marked_indices) + 1) % 2],
                 color=colors[j])

    plt.xlabel('Time t')
    plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    plt.savefig(f'{save_path}/switching_u.png')
    fig.clear()
    plt.close(fig)


def plot_switch_system(train_config, dataset_config, result: SimulationResult, n_point_delay: int, img_save_path: str):
    Q = np.percentile(result.P_no_Ri[2 * n_point_delay:], (1 - train_config.cp_alpha) * 100)
    plt.hist(result.P_no_Ri[2 * n_point_delay:], bins=100, color='blue', alpha=0.7, label='$R$')
    plt.axvline(x=Q, color='red', linestyle='--',
                label=f'{(1 - train_config.cp_alpha) * 100}% quantile: {Q:.2f}')
    plt.legend(loc='best')
    plt.title(f'Distribution of $R$ and the ${1 - train_config.cp_alpha}$ quantile')
    plt.xlabel('$R$')
    plt.ylabel('frequency')
    plt.savefig(f'{img_save_path}/quantile.png')
    plt.close()

    plt.plot(dataset_config.ts[2 * n_point_delay:], result.P_no_Ri[2 * n_point_delay:], label='$R_t$')
    plt.xlabel('Time t')
    plt.ylabel('$R_t$')
    plt.legend(loc='best')
    plt.savefig(f'{img_save_path}/quantile_time.png')
    plt.close()

    plt.plot(dataset_config.ts[2 * n_point_delay:], result.alpha_ts[2 * n_point_delay:], label='$\\alpha_t$')
    plt.xlabel('Time t')
    plt.ylabel('$\\alpha$')
    plt.legend(loc='best')
    plt.savefig(f'{img_save_path}/alpha.png')
    plt.close()

    plt.plot(dataset_config.ts[2 * n_point_delay:], result.q_ts[2 * n_point_delay:], label='$q_t$')
    plt.xlabel('Time t')
    plt.ylabel('$q_t$')
    plt.legend(loc='best')
    plt.savefig(f'{img_save_path}/q.png')
    plt.close()

    plt.plot(dataset_config.ts[2 * n_point_delay:], result.e_ts[2 * n_point_delay:], label='$e_t$')
    plt.xlabel('Time t')
    plt.ylabel('$e_t$')
    plt.legend(loc='best')
    plt.savefig(f'{img_save_path}/e.png')
    plt.close()

    plt.plot(dataset_config.ts[2 * n_point_delay:], result.switching_indicator[2 * n_point_delay:],
             label='$\\mathbb{I}_t$')
    plt.xlabel('Time t')
    plt.ylabel('$\\mathbb{I}_t$')
    plt.legend(loc='best')
    plt.savefig(f'{img_save_path}/I.png')
    plt.close()

    plot_switch_segments(dataset_config.ts, result, img_save_path, n_point_delay)


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


def plot_uncertainty(ts, P, P_ci, Z, delay, n_point_delay, save_path, n_state: int, ylim=None):
    fig = plt.figure(figsize=set_size(width=fig_width))

    for state in range(n_state):
        plt.plot(ts[n_point_delay:], shift(P, n_point_delay)[:, state], linestyle=styles[state], color=colors[state],
                 label=f'$\hat{{P}}_{state + 1}(t-{delay})$')
        plt.plot(ts[n_point_delay:], Z[n_point_delay:, state], label=f'$Z_{state + 1}(t)$', color=colors[state])
        plt.fill_between(ts[n_point_delay:], shift(P_ci, n_point_delay)[:, state, 0],
                         shift(P_ci, n_point_delay)[:, state, 1], color=colors[state], alpha=0.3,
                         label=f"$C(\hat{{P}}_{state + 1})$")
    plt.xlabel('Time t')
    if ylim is not None:
        plt.ylim(ylim)
    if n_state < 5:
        plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    if save_path is not None:
        plt.savefig(save_path)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()


def plot_comparison(ts, Ps, Z, delay, n_point_delay, save_path, n_state: int, ylim=None, Ps_labels=None):
    fig = plt.figure(figsize=set_size(width=fig_width))
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
            plt.plot(ts[n_point_start:], P[:, i], linestyle=styles[j], color=colors[i],
                     label=f'$P^{{{label}}}_{i + 1}(t-{delay_label})$')
        plt.plot(ts[n_point_start:], Z[n_point_start:, i], label=f'$Z_{i + 1}(t)$', linestyle='--', color=colors[i])
    plt.xlabel('Time t')
    if ylim is not None:
        plt.ylim(ylim)
    if n_state < 5:
        plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    else:
        plt.legend(handles=[plt.Line2D([0], [0], color='black', linestyle='--'),
                            plt.Line2D([0], [0], color='black', linestyle='-')],
                   labels=[f'$P(t-{delay_label})$', f'$Z(t)$'], loc='best')
    if save_path is not None:
        plt.savefig(save_path)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()


def plot_difference(ts, Ps, Z, delay, n_point_delay, save_path, n_state: int, ylim=None, Ps_labels=None):
    fig = plt.figure(figsize=set_size(width=fig_width))
    if Ps_labels is None:
        Ps_labels = ['' for _ in range(len(Ps))]
    n_point_start = n_point_delay(0)
    Ps_ = []
    for P in Ps:
        P_ = np.zeros_like(Z[n_point_start:])
        for ti, t in enumerate(ts[n_point_start:]):
            P_[ti] = P[ti + n_point_start - n_point_delay(t)]
        Ps_.append(P_)
    Ps = Ps_

    for i in range(n_state):
        for j, (P, label) in enumerate(zip(Ps, Ps_labels)):
            difference = P - Z[n_point_start:]
            plt.plot(ts[n_point_start:], difference[:, i], linestyle=styles[j], color=colors[i],
                     label=f'$\Delta P^{{{label}}}_{i + 1}$')

    plt.xlabel('Time t')
    if ylim is not None:
        plt.ylim(ylim)
    if n_state < 5:
        plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    else:
        plt.legend(handles=[plt.Line2D([0], [0], color='black', linestyle='-')],
                   labels=[f'$\Delta P(t-{delay})$'], loc='best')
    if save_path is not None:
        plt.savefig(save_path)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()


def plot_control(ts, U, save_path, n_point_delay, ylim=None):
    fig = plt.figure(figsize=set_size(width=fig_width))
    assert U.ndim == 2
    n_point_start = n_point_delay(0)
    U = U.T
    for i, u in enumerate(U):
        plt.plot(ts[n_point_start:], u[n_point_start:], label=f'$U_{i + 1}(t)$', color=colors[i])
    plt.xlabel('Time t')
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    if save_path is not None:
        plt.savefig(save_path)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()


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
