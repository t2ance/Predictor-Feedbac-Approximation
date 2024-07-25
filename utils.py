import datetime
import os
import random
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DatasetConfig, TrainConfig
from dataset import PredictionDataset
from model import FNOProjection, FNOTwoStage, PIFNO, FNN

colors = ['red', 'green', 'blue', 'yellow', 'black', 'cyan', 'magenta', 'white', 'pink', 'orange', 'gray', 'lightblue',
          'lightgreen', 'purple', 'brown', 'teal', 'olive', 'navy', 'lime', 'coral', 'salmon', 'aqua', 'wheat']
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
styles = ['-', '--', '-.', ':']
legend_loc = 'upper right'
legend_fontsize = 8
fig_width = 469.75502


@dataclass
class SimulationResult:
    U: np.ndarray = None
    Z: np.ndarray = None
    P_explicit: np.ndarray = None
    P_no: np.ndarray = None
    P_no_ci: np.ndarray = None
    P_numerical: np.ndarray = None
    P_switching: np.ndarray = None
    runtime: float = None
    P_numerical_n_iters: np.ndarray = None
    p_numerical_count: int = None
    p_no_count: int = None
    P_no_Ri: np.ndarray = None
    alpha_ts: np.ndarray = None
    q_ts: np.ndarray = None
    e_ts: np.ndarray = None
    switching_indicator: np.ndarray = None


def print_args(args):
    print('=' * 100)
    print(args.__class__.__name__)
    for k, v in args.__dict__.items():
        print(f'        - {k} : {v}')
    print('=' * 100)


def draw_distribution(samples, n_state: int, img_save_path: str = None):
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


def postprocess(samples, dataset_config: DatasetConfig):
    def add_noise(tensor, std):
        noise = torch.randn_like(tensor) * std
        return tensor + noise

    print('[DEBUG] postprocessing')
    new_samples = []
    for _ in tqdm(range(dataset_config.n_augment)):
        for feature, p in samples:
            t = feature[:1]
            z = feature[1:3]
            u = feature[3:]
            z = add_noise(z, dataset_config.epsilon)
            u = add_noise(u, dataset_config.epsilon)
            # p = solve_integral_successive(Z_t=z, U_D=u, dt=dataset_config.dt, n_state=dataset_config.system.n_state,
            #                               n_points=dataset_config.n_point_delay, f=dataset_config.system.dynamic,
            #                               n_iterations=dataset_config.successive_approximation_n_iteration)
            new_samples.append((torch.concatenate([t, z, u]), p))
            # u = add_noise(u, dataset_config.epsilon)
            # feature = sample[0].cpu().numpy()
            # p = sample[1].cpu().numpy()
            # new_samples.append((torch.from_numpy(np.concatenate([t, z, u])), torch.tensor(p)))
        # print(f'[WARNING] {len(new_samples)} samples replaced by numerical solutions')
    # all_samples = new_samples
    samples_ = [*samples, *new_samples]
    random.shuffle(samples_)
    return samples_
    # return new_samples


def split_dataset(dataset, ratio):
    n_total = len(dataset)
    n_sample = int(n_total * ratio)
    random.shuffle(dataset)
    return dataset[:n_sample], dataset[n_sample:]


def prepare_datasets(training_samples, batch_size: int, training_ratio: float = None, validation_samples=None):
    assert training_ratio is not None or validation_samples is not None
    if validation_samples is not None:
        print(f'training and validation datasets are both provided: '
              f'#train {len(training_samples)} and #val {len(validation_samples)}')
        train_dataset, validate_dataset = training_samples, validation_samples
    elif training_ratio is not None:
        train_dataset, validate_dataset = split_dataset(training_samples, training_ratio)
    else:
        raise NotImplementedError()
    training_dataloader = DataLoader(PredictionDataset(train_dataset), batch_size=batch_size, shuffle=False)
    if len(validate_dataset) == 0:
        validating_dataloader = None
    else:
        validating_dataloader = DataLoader(PredictionDataset(validate_dataset), batch_size=batch_size, shuffle=False)
    return training_dataloader, validating_dataloader


def load_model(train_config, model_config, dataset_config):
    model_name = model_config.model_name
    device = train_config.device
    n_state = dataset_config.system.n_state
    n_input = dataset_config.system.n_input
    hidden_size = model_config.deeponet_n_hidden_size
    n_hidden = model_config.deeponet_n_hidden
    merge_size = model_config.deeponet_merge_size
    n_point_delay = dataset_config.n_point_delay
    n_modes_height = model_config.fno_n_modes_height
    hidden_channels = model_config.fno_hidden_channels
    n_layers = model_config.n_layer
    layer_width = model_config.ffn_layer_width
    if model_name == 'DeepONet':
        layer_size_branch = [n_point_delay * n_input + n_state] + [hidden_size] * n_hidden + [merge_size]
        layer_size_trunk = [1] + [hidden_size] * n_hidden + [merge_size]
        import deepxde as dde
        model = dde.nn.DeepONet(
            layer_size_branch,
            layer_size_trunk,
            activation="tanh",
            kernel_initializer="Glorot uniform",
            multi_output_strategy='independent',
            num_outputs=n_state
        )
    elif model_name == 'FNO':
        model = FNOProjection(
            n_modes_height=n_modes_height, hidden_channels=hidden_channels, n_state=n_state,
            n_point_delay=n_point_delay, n_input=n_input, n_layers=n_layers)
    elif model_name == 'FNOTwoStage':
        model = FNOTwoStage(
            n_modes_height=n_modes_height, hidden_channels=hidden_channels, n_layers=n_layers, dt=dataset_config.dt,
            n_state=dataset_config.n_state)
    elif model_name == 'PIFNO':
        model = PIFNO(
            n_modes_height=n_modes_height, hidden_channels=hidden_channels, n_layers=n_layers, dt=dataset_config.dt,
            n_state=dataset_config.n_state, dynamic=dataset_config.system.dynamic_tensor_batched2)
    elif model_name == 'FFN':
        model = FNN(n_state=n_state, n_point_delay=n_point_delay, n_input=n_input, n_layers=n_layers,
                    layer_width=layer_width)
    else:
        raise NotImplementedError()
    n_params = count_params(model)
    print(f'Loading {model_name} model from sketch, with {n_params} parameters')
    check_dir(train_config.model_save_path)
    np.savetxt(f'{train_config.model_save_path}/{model_config.model_name}.txt', np.array([n_params]))
    print(os.getcwd())
    pth = f'{train_config.model_save_path}/{model_config.model_name}.pth'
    if train_config.load_model and os.path.exists(pth):
        model.load_state_dict(torch.load(pth))
        print(f'Model loaded from {pth}')
        loaded = True
    else:
        if not train_config.load_model:
            print("Model doesn't need loading")
        elif not os.path.exists(pth):
            print(f"Model save path {pth} doesn't exist")
        loaded = False
    return model.to(device), loaded


def load_optimizer(parameters, train_config):
    return torch.optim.AdamW(parameters, lr=train_config.learning_rate, weight_decay=train_config.weight_decay)


def load_lr_scheduler(optimizer: torch.optim.Optimizer, config):
    assert isinstance(config, TrainConfig) or isinstance(config, DatasetConfig)
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    if config.lr_scheduler_type == 'linear_with_warmup':
        num_warmup_steps = int(config.n_epoch * config.scheduler_ratio_warmup)

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(config.n_epoch - current_step) / float(
                max(1, config.n_epoch - num_warmup_steps)))

        return LambdaLR(optimizer, lr_lambda)
    elif config.lr_scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size,
                                               gamma=config.scheduler_gamma)
    elif config.lr_scheduler_type == 'none':
        def lr_lambda(current_step):
            return 1.0

        return LambdaLR(optimizer, lr_lambda)
    else:
        raise NotImplementedError()


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


def metric(preds, trues):
    N = trues.shape[0]
    preds = np.atleast_2d(preds)
    trues = np.atleast_2d(trues)
    l2 = np.sum(np.linalg.norm(preds - trues, axis=1)) / N
    rl2 = np.sum(np.linalg.norm(preds - trues, axis=1) / np.linalg.norm(trues, axis=1)) / N
    return rl2, l2


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def predict_and_loss(inputs, labels, model):
    return model(inputs, labels)


def quantile_predict_and_loss(inputs, labels, model):
    outputs = model(inputs)
    loss = quantile_loss(outputs, labels, model.quantiles)
    return outputs, loss


def quantile_loss(predictions, targets, quantiles):
    losses = []
    for i, quantile in enumerate(quantiles):
        lower_pred = predictions[:, i, 0]
        upper_pred = predictions[:, i, 1]
        errors_lower = targets[:, i] - lower_pred
        errors_upper = targets[:, i] - upper_pred
        loss_lower = torch.max((quantile[0] - 1) * errors_lower, quantile[0] * errors_lower).mean()
        loss_upper = torch.max((quantile[1] - 1) * errors_upper, quantile[1] * errors_upper).mean()
        losses.append(loss_lower + loss_upper)
    return torch.stack(losses).mean()


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


def shift(p, n_point_delay):
    if n_point_delay == 0:
        return p
    else:
        return p[:-n_point_delay]


def plot_switch_segments(ts, result: SimulationResult, save_path):
    U, switching_indicator = result.U, result.switching_indicator
    fig = plt.figure(figsize=set_size(width=fig_width))
    marked_indices = np.where(np.logical_xor(switching_indicator[:-1], switching_indicator[1:]))[0]

    color_labels = ['$U_{NO}$', '$U_{Numerical}$']
    for j in range(U.shape[-1]):
        u = U[:, j]
        plt.plot(ts[:marked_indices[0] + 1], u[:marked_indices[0] + 1], linestyle=styles[0], label=color_labels[0],
                 color=colors[j])
        for i in range(len(marked_indices) - 1):
            plt.plot(ts[marked_indices[i]:marked_indices[i + 1] + 1], u[marked_indices[i]:marked_indices[i + 1] + 1],
                     linestyle=styles[(i + 1) % 2], label=color_labels[(i + 1) % 2] if i == 0 else "", color=colors[j])
        plt.plot(ts[marked_indices[-1]:], u[marked_indices[-1]:], linestyle=styles[len(marked_indices) % 2],
                 color=colors[j])

    plt.xlabel('Time t')
    plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    plt.savefig(f'{save_path}/u_marked.png')
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

    plot_switch_segments(dataset_config.ts, result, img_save_path)


def plot_result(dataset_config, img_save_path, P_no, P_numerical, P_explicit, P_switching, Z, U,
                method: Literal['explicit', 'numerical', 'no', 'numerical_no', 'switching', 'scheduled_sampling']):
    if img_save_path is None:
        return
    ts = dataset_config.ts
    delay = dataset_config.delay
    n_point_delay = dataset_config.n_point_delay
    n_state = dataset_config.n_state

    # comparison_full = f'{img_save_path}/comparison_full.png'
    # difference_full = f'{img_save_path}/difference_full.png'
    # comparison_zoom = f'{img_save_path}/comparison_zoom.png'
    # difference_zoom = f'{img_save_path}/difference_zoom.png'
    comparison_full = f'{img_save_path}/{method}_comparison_fit.png'
    difference_full = f'{img_save_path}/{method}_comparison_fit.png'
    comparison_zoom = f'{img_save_path}/{method}_comparison.png'
    difference_zoom = f'{img_save_path}/{method}_comparison.png'
    u_path = f'{img_save_path}/u.png'
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
    # plt.title('Uncertainty')

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
    if Ps_labels is None:
        Ps_labels = ['' for _ in range(len(Ps))]
    # plt.title('Comparison')
    for i in range(n_state):
        for j, (P, label) in enumerate(zip(Ps, Ps_labels)):
            plt.plot(ts[n_point_delay:], shift(P, n_point_delay)[:, i], linestyle=styles[j], color=colors[i],
                     label=f'$P^{{{label}}}_{i + 1}(t-{delay})$')
        plt.plot(ts[n_point_delay:], Z[n_point_delay:, i], label=f'$Z_{i + 1}(t)$', linestyle='--', color=colors[i])
    plt.xlabel('Time t')
    if ylim is not None:
        plt.ylim(ylim)
    if n_state < 5:
        plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    else:
        plt.legend(handles=[plt.Line2D([0], [0], color='black', linestyle='--'),
                            plt.Line2D([0], [0], color='black', linestyle='-')],
                   labels=[f'$P(t-{delay})$', f'$Z(t)$'],
                   loc='best')
    if save_path is not None:
        plt.savefig(save_path)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()


def plot_difference(ts, Ps, Z, delay, n_point_delay, save_path, n_state: int, ylim=None, Ps_labels=None):
    fig = plt.figure(figsize=set_size(width=fig_width))
    # plt.title('Difference')
    if Ps_labels is None:
        Ps_labels = ['' for _ in range(len(Ps))]
    for i in range(n_state):
        for j, (P, label) in enumerate(zip(Ps, Ps_labels)):
            difference = shift(P, n_point_delay) - Z[n_point_delay:]
            plt.plot(ts[n_point_delay:], difference[:, i], linestyle=styles[j], color=colors[i],
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
    U = U.T
    for i, u in enumerate(U):
        plt.plot(ts[n_point_delay:], u[n_point_delay:], label=f'$U_{i}(t)$', color=colors[i])
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


def set_size(width='thesis', fraction=1, subplots=(1, 1), height_add=0.1):
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
            if segment.ndim == 2:
                padding = np.zeros((padding_length, segment.shape[1]))
            else:
                padding = np.zeros(padding_length)
            segment = np.concatenate((padding, segment))
        elif isinstance(segment, torch.Tensor):
            if segment.ndim == 2:
                padding = torch.zeros((padding_length, segment.shape[1]))
            else:
                padding = torch.zeros(padding_length)
            segment = torch.concatenate((padding, segment))
        else:
            raise NotImplementedError()

    return segment


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_result(result, dataset_config):
    print(f'Relative L2 error: {result[0]}'
          f' || L2 error: {result[1]}'
          f' || Runtime: {result[2]}'
          f' || Successful cases: [{result[3]}/{len(dataset_config.test_points)}]')


if __name__ == '__main__':
    ...
