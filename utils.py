import datetime
import os
import random
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DatasetConfig, TrainConfig, ModelConfig
from dataset import PredictionDataset
from model import FNOProjection, FFN, GRUNet, LSTMNet, FNOProjectionGRU, FNOProjectionLSTM, DeepONet, DeepONetGRU, \
    DeepONetLSTM

from torch import nn
from torch.nn import init


@dataclass
class SimulationResult:
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
    l2: float = None
    success: bool = None
    n_success: int = None
    n_parameter: int = None


@dataclass
class TestResult:
    runtime: float = None
    l2: float = None
    success_cases: int = None
    results: List = None
    no_pred_ratio: List = None


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def print_args(args):
    print('=' * 100)
    print(args.__class__.__name__)
    for k, v in args.__dict__.items():
        print(f'        - {k} : {v}')
    print('=' * 100)


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


def load_cp_hyperparameters(case: str):
    """
    tlb,tub,cp_gamma,cp_alpha
    """
    if case == 'toy_id':
        return 0., 1., 0.01, 0.1, 's1'
    elif case == 'toy_ood':
        return 1.2, 1.6, 0.01, 0.1, 's1'

    elif case == 'baxter_id':
        return 0., 1., 0.01, 0.01, 's5'
    elif case == 'baxter_ood1':
        return 1., 1.1, 0.01, 0.1, 's5'
    elif case == 'baxter_ood2':
        return 1.5, 2., 0.01, 0.3, 's5'

    elif case == 'unicycle_id':
        return 0., 1., 0.01, 0.01, 's7'
    elif case == 'unicycle_ood':
        return 1., 1.5, 0.01, 0.1, 's7'

    elif case == 'toy_alpha_0.01':
        return 1.4, 1.6, 0.01, 0.01, 's1'
    elif case == 'toy_alpha_0.05':
        return 1.4, 1.6, 0.01, 0.05, 's1'
    elif case == 'toy_alpha_0.1':
        return 1.4, 1.6, 0.01, 0.1, 's1'
    elif case == 'toy_alpha_0.2':
        return 1.4, 1.6, 0.01, 0.2, 's1'
    elif case == 'toy_alpha_0.5':
        return 1.4, 1.6, 0.01, 0.5, 's1'

    elif case == 'toy_gamma_0.05':
        return 1.2, 1.6, 0.05, 0.1
    elif case == 'toy_gamma_0.1':
        return 1.2, 1.6, 0.1, 0.1
    elif case == 'toy_gamma_0.2':
        return 1.2, 1.6, 0.2, 0.1
    else:
        raise NotImplementedError()


def load_model(train_config: TrainConfig, model_config: ModelConfig, dataset_config: DatasetConfig, ffn=None,
               n_param_out: bool = False, model_name: str = None):
    if model_name is None:
        model_name = model_config.model_name
    device = train_config.device
    n_state = dataset_config.system.n_state
    n_input = dataset_config.system.n_input
    n_point_start = dataset_config.n_point_start()
    if model_name == 'DeepONet':
        model = DeepONet(n_input_branch=n_point_start * n_input, n_input_trunk=n_state,
                         layer_width=model_config.deeponet_hidden_size, n_layer=model_config.deeponet_n_layer,
                         n_output=n_state)
    elif model_name == 'FNO':
        n_modes_height = model_config.fno_n_modes_height
        hidden_channels = model_config.fno_hidden_channels
        model = FNOProjection(n_modes_height=n_modes_height, hidden_channels=hidden_channels, n_state=n_state,
                              n_layers=model_config.fno_n_layer)
    elif model_name == 'FFN':
        model = FFN(n_state=n_state, n_point_delay=n_point_start, n_input=n_input, n_layers=model_config.ffn_n_layer,
                    layer_width=model_config.ffn_layer_width)
    elif model_name == 'GRU':
        model = GRUNet(input_size=n_state + n_point_start * n_input, layer_width=model_config.gru_layer_width,
                       num_layers=model_config.gru_n_layer, output_size=n_state)
    elif model_name == 'LSTM':
        model = LSTMNet(input_size=n_state + n_point_start * n_input, layer_width=model_config.lstm_layer_width,
                        num_layers=model_config.lstm_n_layer, output_size=n_state)
    elif model_name == 'FNO-GRU':
        model = FNOProjectionGRU(
            n_modes_height=model_config.fno_n_modes_height, hidden_channels=model_config.fno_hidden_channels,
            n_state=n_state, fno_n_layers=model_config.fno_n_layer, gru_n_layers=model_config.gru_n_layer,
            gru_layer_width=model_config.gru_layer_width, ffn=ffn, residual=train_config.residual,
            zero_init=train_config.zero_init)
    elif model_name == 'FNO-LSTM':
        model = FNOProjectionLSTM(
            n_modes_height=model_config.fno_n_modes_height, hidden_channels=model_config.fno_hidden_channels,
            n_state=n_state, fno_n_layers=model_config.fno_n_layer, lstm_n_layers=model_config.lstm_n_layer,
            lstm_layer_width=model_config.lstm_layer_width, ffn=ffn, residual=train_config.residual,
            zero_init=train_config.zero_init)
    elif model_name == 'DeepONet-GRU':
        model = DeepONetGRU(n_state=n_state, gru_n_layers=model_config.gru_n_layer, n_point_start=n_point_start,
                            n_input=n_input, gru_layer_width=model_config.gru_layer_width, ffn=ffn,
                            deeponet_hidden_size=model_config.deeponet_hidden_size,
                            deeponet_n_layer=model_config.deeponet_n_layer, residual=train_config.residual,
                            zero_init=train_config.zero_init)
    elif model_name == 'DeepONet-LSTM':
        model = DeepONetLSTM(n_state=n_state, lstm_n_layers=model_config.lstm_n_layer, n_point_start=n_point_start,
                             n_input=n_input, lstm_layer_width=model_config.lstm_layer_width, ffn=ffn,
                             deeponet_hidden_size=model_config.deeponet_hidden_size,
                             deeponet_n_layer=model_config.deeponet_n_layer, residual=train_config.residual,
                             zero_init=train_config.zero_init)
    else:
        raise NotImplementedError()
    n_params = count_params(model)
    print(f'Using {model_name} with {n_params} parameters. Xavier initializing.')
    initialize_weights(model)
    if n_param_out:
        return model.to(device), n_params
    else:
        return model.to(device)


def load_optimizer(parameters, train_config):
    return torch.optim.AdamW(parameters, lr=train_config.learning_rate, weight_decay=train_config.weight_decay)


def load_lr_scheduler(optimizer: torch.optim.Optimizer, config):
    assert isinstance(config, TrainConfig) or isinstance(config, DatasetConfig)
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


'''def metric(P, Z, n_point_delay, ts):
    P = prediction_comparison(P, n_point_delay, ts)[n_point_delay(0):]
    Z = Z[2 * n_point_delay(0):]
    N = Z.shape[0]
    P = np.atleast_2d(P)
    Z = np.atleast_2d(Z)
    l2 = np.sum(np.linalg.norm(P - Z, axis=1)) / N
    rl2 = np.sum(np.linalg.norm(P - Z, axis=1) / np.linalg.norm(Z, axis=1)) / N
    return rl2, l2'''


def metric(P, P_numerical, n_point_delay):
    P = np.atleast_2d(P)
    P_numerical = np.atleast_2d(P_numerical)
    assert P.shape == P_numerical.shape
    P = P[n_point_delay:]
    P_numerical = P_numerical[n_point_delay:]
    return np.sum(np.linalg.norm(P - P_numerical, axis=1)) / P.shape[0]


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def predict_and_loss(inputs, labels, model):
    return model(inputs[:, 1:], labels)


def prediction_comparison(P, n_point_delay, ts):
    P_ = np.zeros_like(P[n_point_delay(0):])
    for ti, t in enumerate(ts[n_point_delay(0):]):
        P_[ti] = P[ti + n_point_delay(0) - n_point_delay(t)]

    return P_


def head_points(P, n_point_start):
    if n_point_start == 0:
        return P
    else:
        return P[:-n_point_start]


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
