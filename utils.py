import datetime
import os
import random
import time
from dataclasses import dataclass
from typing import List, Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import init
from torch.optim.lr_scheduler import LambdaLR

from model import FNOProjection, GRUNet, LSTMNet, DeepONet, TimeAwareNeuralOperator


@dataclass
class SimulationResult:
    Z0: Any = None
    U: np.ndarray = None
    Z: np.ndarray = None
    D_explicit: np.ndarray = None
    D_no: np.ndarray = None
    D_numerical: np.ndarray = None
    D_switching: np.ndarray = None
    P_explicit: np.ndarray = None
    P_no: np.ndarray = None
    P_no_ci: np.ndarray = None
    P_numerical: np.ndarray = None
    P_switching: np.ndarray = None
    runtime: float = None
    avg_prediction_time: float = None
    P_numerical_n_iters: np.ndarray = None
    p_numerical_count: int = None
    p_no_count: int = None
    P_no_Ri: np.ndarray = None
    alpha_ts: np.ndarray = None
    q_ts: np.ndarray = None
    e_ts: np.ndarray = None
    switching_indicator: np.ndarray = None
    l2_p_z: float = None
    rl2_p_z: float = None
    l2_p_phat: float = None
    rl2_p_phat: float = None
    success: bool = None
    n_success: int = None
    n_parameter: int = None


@dataclass
class TestResult:
    runtime: float = None
    l2: float = None
    rl2: float = None
    success_cases: int = None
    results: List = None
    no_pred_ratio: List = None


def initialize_weights(m, init_type):
    if isinstance(m, nn.Linear):
        if init_type == 'xavier':
            init.xavier_normal_(m.weight)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight)
        else:
            raise NotImplementedError()
        if m.bias is not None:
            init.zeros_(m.bias)


def print_args(args):
    print('=' * 100)
    print(args.__class__.__name__)
    for k, v in args.__dict__.items():
        print(f'        - {k} : {v}')
    print('=' * 100)


def load_model(train_config, model_config, dataset_config, n_param_out: bool = False, model_name: str = None):
    if model_name is None:
        model_name = model_config.model_name
    device = train_config.device
    n_state = dataset_config.system.n_state
    n_input = dataset_config.system.n_input
    seq_len = dataset_config.max_n_point_delay()
    if model_name == 'DeepONet':
        model = DeepONet(hidden_size=model_config.deeponet_hidden_size, n_layer=model_config.deeponet_n_layer,
                         n_input=n_input, n_state=n_state, seq_len=seq_len, use_t=train_config.use_t,
                         z2u=model_config.z2u)
    elif model_name == 'FNO':
        n_modes_height = model_config.fno_n_modes_height
        hidden_channels = model_config.fno_hidden_channels
        model = FNOProjection(n_modes_height=n_modes_height, hidden_channels=hidden_channels,
                              n_layers=model_config.fno_n_layer, n_input=n_input, n_state=n_state, seq_len=seq_len,
                              use_t=train_config.use_t, z2u=model_config.z2u)
    elif model_name == 'GRU':
        model = GRUNet(hidden_size=model_config.gru_hidden_size, num_layers=model_config.gru_n_layer, n_input=n_input,
                       n_state=n_state, seq_len=seq_len, use_t=train_config.use_t,
                       z2u=model_config.z2u)
    elif model_name == 'LSTM':
        model = LSTMNet(hidden_size=model_config.lstm_hidden_size, num_layers=model_config.lstm_n_layer,
                        n_input=n_input, n_state=n_state, seq_len=seq_len, use_t=train_config.use_t,
                        z2u=model_config.z2u)
    elif model_name in ['FNO-GRU', 'GRU-FNO']:
        model = TimeAwareNeuralOperator(
            ffn='FNO', rnn='GRU',
            invert=model_name.startswith('GRU'),
            params={
                'fno_n_modes_height': model_config.fno_n_modes_height,
                'fno_hidden_channels': model_config.fno_hidden_channels,
                'fno_n_layers': model_config.fno_n_layer,
                'gru_n_layers': model_config.gru_n_layer,
                'gru_hidden_size': model_config.gru_hidden_size
            }, n_input=n_input, n_state=n_state, seq_len=seq_len, use_t=train_config.use_t, z2u=model_config.z2u)
    elif model_name in ['FNO-LSTM', 'LSTM-FNO']:
        model = TimeAwareNeuralOperator(
            ffn='FNO', rnn='LSTM', n_input=n_input, n_state=n_state, seq_len=seq_len, use_t=train_config.use_t,
            invert=model_name.startswith('LSTM'),
            params={
                'fno_n_modes_height': model_config.fno_n_modes_height,
                'fno_hidden_channels': model_config.fno_hidden_channels,
                'fno_n_layers': model_config.fno_n_layer,
                'lstm_n_layers': model_config.lstm_n_layer,
                'lstm_hidden_size': model_config.lstm_hidden_size
            }, z2u=model_config.z2u)
    elif model_name in ['DeepONet-GRU', 'GRU-DeepONet']:
        model = TimeAwareNeuralOperator(
            ffn='DeepONet', rnn='GRU', n_input=n_input, n_state=n_state, seq_len=seq_len, use_t=train_config.use_t,
            invert=model_name.startswith('GRU'),
            params={
                'deeponet_hidden_size': model_config.deeponet_hidden_size,
                'deeponet_n_layer': model_config.deeponet_n_layer,
                'gru_n_layers': model_config.gru_n_layer,
                'gru_hidden_size': model_config.gru_hidden_size
            }, z2u=model_config.z2u)
    elif model_name in ['DeepONet-LSTM', 'LSTM-DeepONet']:
        model = TimeAwareNeuralOperator(
            ffn='DeepONet', rnn='LSTM', n_input=n_input, n_state=n_state, seq_len=seq_len, use_t=train_config.use_t,
            invert=model_name.startswith('LSTM'),
            params={
                'deeponet_hidden_size': model_config.deeponet_hidden_size,
                'deeponet_n_layer': model_config.deeponet_n_layer,
                'lstm_n_layers': model_config.lstm_n_layer,
                'lstm_hidden_size': model_config.lstm_hidden_size
            }, z2u=model_config.z2u)
    else:
        raise NotImplementedError()
    n_params = count_params(model)
    if isinstance(model, TimeAwareNeuralOperator):
        print('ffn parameters:', count_params(model.ffn))
        print('rnn parameters:', count_params(model.rnn))
        # print('projection parameters:', count_params(model.projection))
    print(f'Using {model_name} with {n_params} parameters. {model_config.init_type} initializing.')
    initialize_weights(model, model_config.init_type)
    if n_param_out:
        return model.to(device), n_params
    else:
        return model.to(device)


def load_optimizer(parameters, train_config):
    return torch.optim.AdamW(parameters, lr=train_config.learning_rate, weight_decay=train_config.weight_decay)


def load_lr_scheduler(optimizer: torch.optim.Optimizer, config):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    num_warmup_steps = int(config.n_epoch * config.scheduler_ratio_warmup)
    total_steps = config.n_epoch
    if config.lr_scheduler_type == 'linear_with_warmup':

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(
                max(1, total_steps - num_warmup_steps)))

        return LambdaLR(optimizer, lr_lambda)
    elif config.lr_scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size,
                                               gamma=config.scheduler_gamma)
    elif config.lr_scheduler_type == 'cosine_annealing_with_warmup':
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 0.5 * (1 + torch.cos(
                torch.tensor((current_step - num_warmup_steps) / (total_steps - num_warmup_steps) * torch.pi)))

        return LambdaLR(optimizer, lr_lambda)
    elif config.lr_scheduler_type == 'none':
        def lr_lambda(current_step):
            return 1.0

        return LambdaLR(optimizer, lr_lambda)
    else:
        raise NotImplementedError()


def l2_p_z(P, Z, n_point_delay, ts):
    P = prediction_comparison(P, n_point_delay, ts)[n_point_delay(0):]
    Z = Z[2 * n_point_delay(0):]
    N = Z.shape[0]
    P = np.atleast_2d(P)
    Z = np.atleast_2d(Z)
    l2 = np.sum(np.linalg.norm(P - Z, axis=1)) / N
    rl2 = np.sum(np.linalg.norm(P - Z, axis=1) / np.linalg.norm(Z, axis=1)) / N
    return l2, rl2


def l2_p_phat(P, P_numerical, n_point_delay):
    P = np.atleast_2d(P)
    P_numerical = np.atleast_2d(P_numerical)
    N = P.shape[0]
    assert P.shape == P_numerical.shape
    P = P[n_point_delay:]
    P_numerical = P_numerical[n_point_delay:]
    l2 = np.sum(np.linalg.norm(P - P_numerical, axis=1)) / N
    rl2 = np.sum(np.linalg.norm(P - P_numerical, axis=1) / np.linalg.norm(P_numerical, axis=1)) / N
    return l2, rl2


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def prediction_comparison(P, n_point_delay, ts):
    P_ = np.zeros_like(P[n_point_delay(0):])
    for ti, t in enumerate(ts[n_point_delay(0):]):
        P_[ti] = P[ti + n_point_delay(0) - n_point_delay(t)]

    return P_


def get_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S",
                         datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).timetuple())


def count_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def pad_zeros(segment, length, leading: bool = True):
    assert len(segment) <= length

    if len(segment) < length:
        padding_length = length - len(segment)
        if isinstance(segment, np.ndarray):
            if segment.ndim == 2:
                padding = np.zeros((padding_length, segment.shape[1]))
            else:
                padding = np.zeros(padding_length)
            if leading:
                segment = np.concatenate((padding, segment))
            else:
                segment = np.concatenate((segment, padding))
        elif isinstance(segment, torch.Tensor):
            if segment.ndim == 2:
                padding = torch.zeros((padding_length, segment.shape[1]))
            else:
                padding = torch.zeros(padding_length)
            if leading:
                segment = torch.concatenate((padding, segment))
            else:
                segment = torch.concatenate((segment, padding))
        else:
            raise NotImplementedError()

    return segment


def set_everything(seed: int):
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

    # tex_fonts = {
    #     # Use LaTeX to write all text
    #     # "text.usetex": True,
    #     # "font.family": "times",
    #     # Use 10pt font in plots, to match 10pt font in document
    #     "axes.labelsize": 12,
    #     "font.size": 12,
    #     # Make the legend/label fonts a little smaller
    #     "legend.fontsize": 8,
    #     "xtick.labelsize": 10,
    #     "ytick.labelsize": 10
    # }

    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        # "font.family": "times",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6
    }

    plt.rcParams.update(tex_fonts)


def print_result(result: TestResult, dataset_config):
    l2 = result.l2
    prediction_time = result.runtime * 1000
    print(f'L2 error: {l2}'
          f' || Runtime: {prediction_time}'
          f' || Successful cases: [{result.success_cases}/{len(dataset_config.test_points)}]')
    print(f'L2 error: {l2 :.3f}'
          f' || Runtime: {prediction_time :.3f}'
          f' || Successful cases: [{result.success_cases}/{len(dataset_config.test_points):.3f}]')
    print(f'L2 error: ${l2 :.3f}$'
          f' || Runtime: ${prediction_time :.3f}$'
          f' || Successful cases: [{result.success_cases}/{len(dataset_config.test_points)}]')


if __name__ == '__main__':
    ...
