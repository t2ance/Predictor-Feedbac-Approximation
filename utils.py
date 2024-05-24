import os
import time
import datetime
import random

import numpy as np
import torch
from matplotlib import pyplot as plt

import deepxde as dde
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import DatasetConfig, TrainConfig
from dataset import PredictionDataset


def prepare_datasets(samples, training_ratio: float, batch_size: int, device: str):
    def split_dataset(dataset, ratio):
        n_total = len(dataset)
        n_sample = int(n_total * ratio)
        random.shuffle(dataset)
        return dataset[:n_sample], dataset[n_sample:]

    train_dataset, validate_dataset = split_dataset(samples, training_ratio)
    training_dataloader = DataLoader(PredictionDataset(train_dataset), batch_size=batch_size, shuffle=False,
                                     # generator=torch.Generator(device=device)
                                     )
    if len(validate_dataset) == 0:
        validating_dataloader = None
    else:
        validating_dataloader = DataLoader(PredictionDataset(validate_dataset), batch_size=batch_size, shuffle=False,
                                           # generator=torch.Generator(device=device)
                                           )
    return training_dataloader, validating_dataloader


def get_lr_scheduler(optimizer: torch.optim.Optimizer, train_config: TrainConfig):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    if train_config.lr_scheduler_type == 'linear_with_warmup':
        num_warmup_steps = int(train_config.n_epoch * train_config.scheduler_ratio_warmup)

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(train_config.n_epoch - current_step) / float(
                max(1, train_config.n_epoch - num_warmup_steps)))

        return LambdaLR(optimizer, lr_lambda)
    elif train_config.lr_scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_config.scheduler_step_size,
                                               gamma=train_config.scheduler_gamma)
    else:
        raise NotImplementedError()


def plot_sample(feature, label, dataset_config: DatasetConfig):
    if isinstance(feature, torch.Tensor):
        feature = feature.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    print(f'[Feature Shape]: {feature.shape}')
    print(f'[Label Shape]: {label.shape}')
    t = feature[:1]
    z = feature[1:3]
    u = feature[3:]
    p = label
    ts = np.linspace(t - dataset_config.delay, t, dataset_config.n_point_delay)
    plt.plot(ts, u, label='U')
    plt.scatter(ts[-1], z[0], label='$Z_t(0)$')
    plt.scatter(ts[-1], z[1], label='$Z_t(1)$')
    plt.scatter(ts[-1], p[0], label='$P_t(0)$')
    plt.scatter(ts[-1], p[1], label='$P_t(1)$')
    plt.legend()
    plt.show()


def metric(preds, trues):
    N = trues.shape[0]
    preds = np.atleast_2d(preds)
    trues = np.atleast_2d(trues)
    return np.sum(np.linalg.norm(preds - trues, axis=1) / np.linalg.norm(trues, axis=1)) / N, \
           np.sum(np.linalg.norm(preds - trues, axis=1)) / N


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def no_predict_and_loss(inputs, labels, model):
    z_u = inputs[:, 1:]

    if not isinstance(model, dde.nn.DeepONet):
        return model(z_u, labels)
    else:
        outputs = model([z_u, inputs[:, :1]])
        return outputs, torch.nn.MSELoss()(outputs, labels)


def plot_comparison(ts, P_hat, P, Z, delay, n_point_delay, save_path, ylim=None):
    fig = plt.figure(figsize=set_size())
    plt.title('Comparison')
    for t_i in range(2):
        if P is not None:
            plt.plot(ts[n_point_delay:], P[:-n_point_delay, t_i], label=f'$P_{t_i + 1}(t-{delay})$')
        plt.plot(ts[n_point_delay:], P_hat[:-n_point_delay, t_i], label=f'$\hat P_{t_i + 1}(t-{delay})$')
        plt.plot(ts[n_point_delay:], Z[n_point_delay:, t_i], label=f'$Z_{t_i + 1}(t)$')
    plt.xlabel('t')
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()


def plot_difference(ts, P_hat, P, Z, n_point_delay, save_path, ylim=None):
    fig = plt.figure(figsize=set_size())
    difference = P_hat[:-n_point_delay] - Z[n_point_delay:]
    plt.plot(ts[n_point_delay:], difference[:, 0], label='$\hat P_1 - Z_1$')
    plt.plot(ts[n_point_delay:], difference[:, 1], label='$\hat P_2 - Z_2$')
    if P is not None:
        difference_no = P[:-n_point_delay] - Z[n_point_delay:]
        plt.plot(ts[n_point_delay:], difference_no[:, 0], label='$\delta PNO_1$')
        plt.plot(ts[n_point_delay:], difference_no[:, 1], label='$\delta PNO_2$')
    plt.xlabel('t')
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()


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
