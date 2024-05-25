import os
import pickle
import random
import time
from dataclasses import asdict
from typing import Literal, Tuple, List

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from scipy.integrate import odeint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DatasetConfig, ModelConfig, TrainConfig
from dataset import ImplicitDataset, ExplictDataset, PredictionDataset, sample_to_tensor
from dynamic_systems import solve_integral_equation, solve_integral_equation_, solve_integral_equation_neural_operator
from model import FNOProjection, FNOTwoStage, PIFNO, SampleGenerationNet
from utils import count_params, get_time_str, set_size, pad_leading_zeros, plot_comparison, plot_difference, \
    metric, check_dir, plot_sample, no_predict_and_loss, get_lr_scheduler, prepare_datasets, postprocess, \
    draw_distribution


def run(dataset_config: DatasetConfig, Z0: Tuple, method: Literal['explict', 'numerical', 'no', 'numerical_no'] = None,
        model=None, title='', save_path: str = None, img_save_path: str = None):
    system = dataset_config.system
    n_point_delay = dataset_config.n_point_delay
    ts = dataset_config.ts
    Z0 = np.array(Z0)
    dt = dataset_config.dt
    n_point = dataset_config.n_point
    U = np.zeros(n_point)
    Z = np.zeros((n_point, 2))
    P = np.zeros((n_point, 2))
    P_compare = np.zeros((n_point, 2))
    Z[n_point_delay, :] = Z0
    for t_i in range(dataset_config.n_point):
        t_minus_D_i = max(t_i - n_point_delay, 0)
        t = ts[t_i]
        if method == 'explict':
            U[t_i] = system.U_explict(t, Z0)
            if t_i > n_point_delay:
                Z[t_i, :] = system.Z_explicit(t, Z0)
        elif method == 'numerical':
            if t_i > n_point_delay:
                Z[t_i, :] = odeint(system.dynamic, Z[t_i - 1, :], [ts[t_i - 1], ts[t_i]],
                                   args=(U[t_minus_D_i - 1],))[1]
                Z_t = Z[t_i, :]
            else:
                Z_t = Z0
            P_compare[t_i, :] = solve_integral_equation(Z_t, n_point_delay, dataset_config.n_state, dt,
                                                        pad_leading_zeros(segment=U[t_minus_D_i:t_i],
                                                                          length=n_point_delay),
                                                        dynamic=dataset_config.system.dynamic)
            P[t_i, :] = solve_integral_equation_(f=system.dynamic, Z_t=Z_t, P_D=P[t_minus_D_i:t_i],
                                                 U_D=U[t_minus_D_i:t_i], dt=dt,
                                                 t=t) + dataset_config.noise()
            if t_i > n_point_delay:
                U[t_i] = system.kappa(P[t_i, :])
        elif method == 'no':
            if t_i > n_point_delay:
                Z[t_i, :] = \
                    odeint(system.dynamic, Z[t_i - 1, :], [ts[t_i - 1], ts[t_i]],
                           args=(U[t_minus_D_i - 1],))[1]
                Z_t = Z[t_i, :]
            else:
                Z_t = Z0
            P[t_i, :] = solve_integral_equation_neural_operator(
                model=model, U_D=pad_leading_zeros(segment=U[t_minus_D_i:t_i], length=n_point_delay), Z_t=Z_t, t=t)
            P_compare[t_i, :] = solve_integral_equation_(f=system.dynamic, Z_t=Z_t, P_D=P[t_minus_D_i:t_i],
                                                         U_D=U[t_minus_D_i:t_i], dt=dt,
                                                         t=t) + dataset_config.noise()
            if t_i > n_point_delay:
                U[t_i] = system.kappa(P[t_i, :])
        elif method == 'numerical_no':
            if t_i > n_point_delay:
                Z[t_i, :] = odeint(system.dynamic, Z[t_i - 1, :], [ts[t_i - 1], ts[t_i]],
                                   args=(U[t_minus_D_i - 1],))[1]
                Z_t = Z[t_i, :]
            else:
                Z_t = Z0
            P[t_i, :] = solve_integral_equation_(f=system.dynamic, Z_t=Z_t, P_D=P[t_minus_D_i:t_i],
                                                 U_D=U[t_minus_D_i:t_i], dt=dt,
                                                 t=t) + dataset_config.noise()
            P_compare[t_i, :] = solve_integral_equation_neural_operator(
                model=model, U_D=pad_leading_zeros(segment=U[t_minus_D_i:t_i], length=n_point_delay), Z_t=Z_t,
                t=t)
            if t_i > n_point_delay:
                U[t_i] = system.kappa(P[t_i, :])
        else:
            raise NotImplementedError()

    ts = dataset_config.ts
    delay = dataset_config.delay
    n_point_delay = dataset_config.n_point_delay
    if save_path is not None:
        result = {
            "u": U,
            "z": Z,
            "d": dataset_config.delay,
            "duration": dataset_config.duration,
            "n_point_duration": dataset_config.n_point_duration,
            "ts": dataset_config.ts
        }
        with open(save_path, 'wb') as file:
            pickle.dump(result, file)

    if img_save_path is not None:
        fig = plt.figure(figsize=set_size())
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

        if method != 'numerical_no' or model != 'numerical':
            P_compare = None
        plot_comparison(ts, P, P_compare, Z, delay, n_point_delay,
                        f'{img_save_path}/comparison_full.png' if img_save_path is not None else None)
        plot_comparison(ts, P, P_compare, Z, delay, n_point_delay,
                        f'{img_save_path}/comparison_zoom.png' if img_save_path is not None else None, [-5, 5])

        plot_difference(ts, P, P_compare, Z, n_point_delay,
                        f'{img_save_path}/difference_full.png' if img_save_path is not None else None)
        plot_difference(ts, P, P_compare, Z, n_point_delay,
                        f'{img_save_path}/difference_zoom.png' if img_save_path is not None else None, [-1, 1])

    if method == 'explict':
        return U, Z, None
    return U, Z, P


def run_train(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig,
              training_dataloader, validating_dataloader=None, testing_dataloader=None, img_save_path: str = None):
    model_name = model_config.model_name
    n_state = dataset_config.n_state
    device = train_config.device
    hidden_size = model_config.deeponet_n_hidden_size
    n_hidden = model_config.deeponet_n_hidden
    merge_size = model_config.deeponet_merge_size
    lr = train_config.learning_rate
    n_epoch = train_config.n_epoch
    weight_decay = train_config.weight_decay
    n_point_delay = dataset_config.n_point_delay
    n_modes_height = model_config.fno_n_modes_height
    hidden_channels = model_config.fno_hidden_channels
    n_layers = model_config.fno_n_layers
    if model_name == 'DeepONet':
        layer_size_branch = [n_point_delay + n_state] + [hidden_size] * n_hidden + [merge_size]
        layer_size_trunk = [1] + [hidden_size] * n_hidden + [merge_size]
        model = dde.nn.DeepONet(
            layer_size_branch,
            layer_size_trunk,
            activation="tanh",
            kernel_initializer="Glorot uniform",
            multi_output_strategy='independent',
            num_outputs=n_state
        ).to(device)
    elif model_name == 'FNO':
        model = FNOProjection(
            n_modes_height=n_modes_height, hidden_channels=hidden_channels, n_state=n_state,
            n_point_delay=n_point_delay, n_layers=n_layers).to(device)
    elif model_name == 'FNOTwoStage':
        model = FNOTwoStage(
            n_modes_height=n_modes_height, hidden_channels=hidden_channels, n_layers=n_layers, dt=dataset_config.dt,
            n_state=dataset_config.n_state).to(device)
    elif model_name == 'PIFNO':
        model = PIFNO(
            n_modes_height=n_modes_height, hidden_channels=hidden_channels, n_layers=n_layers, dt=dataset_config.dt,
            n_state=dataset_config.n_state, dynamic=dataset_config.system.dynamic_tensor_batched2).to(device)
    else:
        raise NotImplementedError()
    print(f'#parameters: {count_params(model)}')
    pth = f'{train_config.model_save_path}/{model_config.model_name}.pth'
    if train_config.load_model and os.path.exists(pth):
        model.load_state_dict(torch.load(f'{train_config.model_save_path}/{model_name}.pth'))
        print(f'Model loaded from {pth}, skip training!')
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = get_lr_scheduler(optimizer, train_config)
        training_loss_arr = []
        validating_loss_arr = []
        testing_loss_arr = []
        rl2_list = []
        l2_list = []
        bar = tqdm(list(range(n_epoch)))
        do_validation = validating_dataloader is not None
        do_testing = testing_dataloader is not None

        def draw():
            fig = plt.figure(figsize=set_size())
            x_metric = list(range(0, train_config.log_step * len(rl2_list), train_config.log_step))
            plt.plot(x_metric, rl2_list, label="Relative Difference")
            plt.plot(x_metric, l2_list, label="Difference")
            plt.xlabel('epoch')
            plt.legend()
            if img_save_path is not None:
                plt.savefig(f'{img_save_path}/metric.png')
                fig.clear()
                plt.close(fig)
            else:
                plt.show()

            fig = plt.figure(figsize=set_size())
            plt.plot(training_loss_arr, label="Training loss")
            if do_validation:
                plt.plot(validating_loss_arr, label="Validation loss")
            if do_testing:
                plt.plot(testing_loss_arr, label="Test loss")
            plt.yscale("log")
            plt.xlabel('epoch')
            plt.legend()
            if img_save_path is not None:
                plt.savefig(f'{img_save_path}/loss.png')
                fig.clear()
                plt.close(fig)
            else:
                plt.show()

        for epoch in bar:
            model.train()
            training_loss = 0.0
            for inputs, labels in training_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, loss = no_predict_and_loss(inputs, labels, model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            model.eval()
            with torch.no_grad():
                if do_validation:
                    validating_loss = 0.0
                    for inputs, labels in validating_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs, loss = no_predict_and_loss(inputs, labels, model)
                        validating_loss += loss.item()

                if do_testing:
                    testing_loss = 0.0
                    for inputs, labels in testing_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs, loss = no_predict_and_loss(inputs, labels, model)
                        testing_loss += loss.item()

            training_loss_t = training_loss / len(training_dataloader)
            training_loss_arr.append(training_loss_t)
            if do_validation:
                validating_loss_t = validating_loss / len(validating_dataloader)
                validating_loss_arr.append(validating_loss_t)
            if do_testing:
                testing_loss_t = testing_loss / len(testing_dataloader)
                testing_loss_arr.append(testing_loss_t)
            lr = scheduler.get_last_lr()[-1]
            desc = f'Epoch [{epoch + 1}/{n_epoch}] || Lr: {lr} || Training loss: {training_loss_t:.6f}'
            wandb.log({
                'training loss': training_loss_t,
                'lr': lr
            }, step=epoch)
            if do_validation:
                desc += f' || Validation loss: {validating_loss_t:.6f}'
                wandb.log({
                    'validation loss': validating_loss_t
                }, step=epoch)
            if do_testing:
                desc += f' || Test loss: {testing_loss_t:.6f}'
                wandb.log({
                    'test loss': testing_loss_t
                }, step=epoch)
            bar.set_description(desc)

            if (train_config.log_step > 0 and epoch % train_config.log_step == 0) or epoch == n_epoch - 1:
                rl2, l2, _ = run_test(model, dataset_config, silence=True)
                rl2_list.append(rl2)
                l2_list.append(l2)
                wandb.log({'Relative L2 error': rl2, 'L2 error': l2}, step=epoch)
            scheduler.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], train_config.scheduler_min_lr)
        draw()
        torch.save(model.state_dict(), f'{train_config.model_save_path}/{model_config.model_name}.pth')
    print('Finished Training')
    return model


def run_test(m, dataset_config: DatasetConfig, base_path: str = None, debug: bool = False, silence: bool = False,
             test_points: List = None, plot: bool = False):
    if test_points is None:
        test_points = dataset_config.test_points
    bar = test_points if silence else tqdm(test_points)
    metric_rl2_list = []
    metric_l2_list = []
    for test_point in bar:
        if not silence:
            bar.set_description(f'Solving system with initial point {np.round(test_point, decimals=2)}.')
        if base_path is not None:
            img_save_path = f'{base_path}/{np.round(test_point, decimals=2)}'
            check_dir(img_save_path)
        else:
            img_save_path = None

        if debug:
            U, Z, P = run(dataset_config=dataset_config, model=m, Z0=test_point, method='numerical_no', title='no',
                          img_save_path=img_save_path)
        else:
            U, Z, P = run(dataset_config=dataset_config, model=m, Z0=test_point, method='no', title='no',
                          img_save_path=img_save_path)
        plt.close()
        delay = dataset_config.n_point_delay
        rl2, l2 = metric(P[delay:-delay], Z[2 * delay:])

        if np.isinf(rl2):
            print(f'[WARNING] Running with initial condition Z = {test_point} failed.')
            continue
        metric_rl2_list.append(rl2)
        metric_l2_list.append(l2)

        if base_path is not None:
            np.savetxt(f'{img_save_path}/metric.txt', np.array([rl2, l2]))
    rl2 = np.nanmean(metric_rl2_list).item()
    l2 = np.nanmean(metric_l2_list).item()
    if plot or base_path is not None:
        def plot_result(data, label, title, xlabel, path):
            fig = plt.figure(figsize=set_size())
            plt.hist(data, bins=20, label=label)
            plt.legend()
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel('Frequency')
            if plot:
                plt.show()
            else:
                plt.savefig(path)
                fig.clear()
                plt.close(fig)

        plot_result(data=metric_rl2_list, label=f'Relative L2 error', title='Relative L2 error',
                    xlabel='Relative L2 error', path=f'{base_path}/rl2.png')
        plot_result(data=metric_l2_list, label=f'L2 error', title='L2 error', xlabel='L2 error',
                    path=f'{base_path}/l2.png')
    return rl2, l2, len(metric_rl2_list)


def get_training_and_validation_datasets(dataset_config: DatasetConfig, train_config: TrainConfig):
    create_dataset = not os.path.exists(
        dataset_config.training_dataset_file) or dataset_config.recreate_training_dataset

    def load():
        print('Loading training dataset')
        with open(dataset_config.training_dataset_file, 'rb') as file:
            training_samples_loaded = pickle.load(file)
        print(f'Loaded {len(training_samples_loaded)} samples')
        return training_samples_loaded

    if create_dataset:
        print('Creating training dataset')
        if dataset_config.data_generation_strategy == 'trajectory':
            training_samples = create_trajectory_dataset(dataset_config)
        elif dataset_config.data_generation_strategy == 'random':
            training_samples = create_random_dataset(dataset_config)
        elif dataset_config.data_generation_strategy == 'nn':
            training_samples = create_nn_dataset(dataset_config)
        else:
            raise NotImplementedError()

        print(f'Created {len(training_samples)} samples')
        if dataset_config.append_training_dataset:
            training_samples_loaded = load()
            if len(training_samples_loaded) > 0:
                tsf, tsl = training_samples[0]
                tslf, tsdl = training_samples_loaded[0]
                assert len(tsf) == len(tslf), f'The shapes of sample should be consistent, but {len(tsf)} ≠ {len(tslf)}'
            training_samples += training_samples_loaded
            print(f'Samples merged! {len(training_samples)} in total')
    else:
        training_samples = load()
    path = dataset_config.training_dataset_file
    if dataset_config.recreate_training_dataset:
        with open(path, 'wb') as file:
            pickle.dump(training_samples, file)
            print(f'{len(training_samples)} samples saved')

    if dataset_config.data_generation_strategy == 'trajectory' and dataset_config.postprocess:
        training_samples = postprocess(training_samples, dataset_config)
    for i, (feature, label) in enumerate(training_samples[:dataset_config.n_plot_sample]):
        plot_sample(feature, label, dataset_config, f'{str(i)}.png')
    draw_distribution(training_samples, dataset_config.dataset_base_path)
    training_dataloader, validating_dataloader = prepare_datasets(
        training_samples, train_config.training_ratio, train_config.batch_size, train_config.device)
    print(f'#Training sample: {int(len(training_samples) * train_config.training_ratio)}')
    print(f'#Validating sample: {int(len(training_samples) * (1 - train_config.training_ratio))}')
    return training_dataloader, validating_dataloader


def get_test_datasets(dataset_config, train_config):
    if not train_config.do_test:
        return None
    if not os.path.exists(dataset_config.testing_dataset_file) or dataset_config.recreate_testing_dataset:
        print('Creating testing dataset')
        testing_samples = create_trajectory_dataset(dataset_config, test_points=dataset_config.test_points)
    else:
        print('Loading testing dataset')
        with open(dataset_config.testing_dataset_file, 'rb') as file:
            testing_samples = pickle.load(file)
    testing_dataloader = DataLoader(PredictionDataset(testing_samples), batch_size=train_config.batch_size,
                                    shuffle=False)
    print(f'#Testing sample: {len(testing_samples)}')
    return testing_dataloader


def create_trajectory_dataset(dataset_config: DatasetConfig, test_points: List = None):
    all_samples = []
    if dataset_config.explicit:
        print('creating implicit datasets (Use Z(t+D) as P(t))')
    else:
        print('creating explicit datasets (Calculate P(t))')
    if test_points is None:
        bar = tqdm(list(
            np.random.uniform(dataset_config.ic_lower_bound, dataset_config.ic_upper_bound,
                              (dataset_config.n_dataset, 2))))
    else:
        bar = tqdm(test_points)
    for Z0 in bar:
        if dataset_config.explicit:
            U, Z, _ = run(method='explict', Z0=Z0, dataset_config=dataset_config)
            dataset = ImplicitDataset(
                torch.tensor(Z, dtype=torch.float32), torch.tensor(U, dtype=torch.float32),
                dataset_config.n_point_delay, dataset_config.dt)
        else:
            U, Z, P = run(method='numerical', Z0=Z0, dataset_config=dataset_config)
            dataset = ExplictDataset(
                torch.tensor(Z, dtype=torch.float32), torch.tensor(U, dtype=torch.float32),
                torch.tensor(P, dtype=torch.float32), dataset_config.n_point_delay, dataset_config.dt)
        dataset = list(dataset)
        random.shuffle(dataset)
        if dataset_config.n_sample_per_dataset >= 0:
            all_samples += dataset[:dataset_config.n_sample_per_dataset]
        else:
            all_samples += dataset
    random.shuffle(all_samples)
    return all_samples


def create_random_dataset(dataset_config: DatasetConfig):
    dt: float = dataset_config.dt
    n_point_delay: int = dataset_config.n_point_delay
    n_sample_per_dataset: int = dataset_config.n_sample_per_dataset
    n_state: int = dataset_config.n_state
    all_samples = []
    n_sample = 0
    n_id_sample = 0
    for i in tqdm(list(range(1, dataset_config.n_dataset + 1))):
        j = 0
        while j < n_sample_per_dataset:
            if dataset_config.random_u_type == 'line':
                f = lambda x: (np.random.uniform(-1, 1, 2).reshape(-1, 1) * np.array(
                    [x, np.ones_like(x)])).sum(
                    axis=0)
            elif dataset_config.random_u_type == 'poly':
                f = lambda x: 3 * (np.random.uniform(-1, 1, 5).reshape(-1, 1) * np.array(
                    [x ** 4, x ** 3, x ** 2, x, np.ones_like(x)])).sum(
                    axis=0)
            elif dataset_config.random_u_type == 'sin':
                f = lambda x: np.sin(np.sqrt(i) * x)
            elif dataset_config.random_u_type == 'exp':
                f = lambda x: np.exp(-np.sqrt(i) * x)
            elif dataset_config.random_u_type == 'chebyshev':
                f = lambda x: np.random.uniform(0, 5) * np.cos(np.random.uniform(0, 5) * np.arccos(x))
            elif dataset_config.random_u_type == 'spline':
                def f(x):
                    segment_length = np.random.uniform(0, 2)
                    start = segment_length * np.floor(np.min(x) / segment_length)
                    end = segment_length * np.ceil(np.max(x) / segment_length)

                    key_points = np.arange(start, end + segment_length, segment_length)
                    random_values = np.random.uniform(-1, 1, size=key_points.shape[0])

                    indices = np.searchsorted(key_points, x, side='right') - 1
                    x1 = key_points[indices]
                    x2 = key_points[indices + 1]
                    y1 = random_values[indices]
                    y2 = random_values[indices + 1]

                    y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
                    return y
            elif dataset_config.random_u_type == 'sinexp':
                feq = np.random.uniform(0, 100)
                k = np.random.uniform(0, 100)
                shift = np.random.uniform(0, 100)
                f = lambda x: np.exp(-k * x) * np.sin(feq * (x + shift))
            else:
                raise NotImplementedError()
            Z_t = np.random.uniform(dataset_config.ic_lower_bound, dataset_config.ic_upper_bound, 2)
            U_D = f(np.linspace(0, 1, n_point_delay))
            P_t = solve_integral_equation(Z_t=Z_t, U_D=U_D, dt=dt, n_state=n_state, n_point_delay=n_point_delay,
                                          dynamic=dataset_config.system.dynamic)
            features = sample_to_tensor(Z_t, U_D, dt * n_point_delay)
            if dataset_config.filter_ood_sample:
                def filter_out_of_distribution_sample(factor, p):
                    # return abs(P_t[0]) <= abs(Z_t[0]) * factor and abs(P_t[1]) <= abs(
                    #     Z_t[1]) * factor and np.random.uniform(0, 1) <= p
                    return np.linalg.norm(P_t) / np.linalg.norm(Z_t) <= factor \
                        and np.random.uniform(0, 1) <= p

                if filter_out_of_distribution_sample(dataset_config.ood_sample_bound, 1):
                    all_samples.append((features, torch.from_numpy(P_t)))
                    j += 1
                    n_id_sample += 1
            else:
                all_samples.append((features, torch.from_numpy(P_t)))
                j += 1
                n_id_sample += 1
            n_sample += 1
    print(
        f'Generated {n_sample} samples in total, {n_id_sample} samples in distribution, percent: {n_id_sample / n_sample}')
    random.shuffle(all_samples)
    return all_samples


def create_nn_dataset(dataset_config: DatasetConfig):
    dt: float = dataset_config.dt
    n_point_delay: int = dataset_config.n_point_delay
    n_sample_per_dataset: int = dataset_config.n_sample_per_dataset
    n_state: int = dataset_config.n_state
    n_sample = dataset_config.n_dataset * n_sample_per_dataset
    model = SampleGenerationNet(dataset_config.n_state, dataset_config.n_point_delay)
    batch_size = dataset_config.generation_net_batch_size
    n_epoch = dataset_config.generation_net_n_epoch

    def torch_uniform(shape, a, b):
        rand_tensor = torch.rand(shape)
        transformed_tensor = a + (b - a) * rand_tensor
        return transformed_tensor

    def get_random_z_p(batch_size):
        z = torch_uniform(shape=(batch_size, 2), a=dataset_config.ic_lower_bound, b=dataset_config.ic_upper_bound)
        p = torch_uniform(shape=(batch_size, 2), a=dataset_config.ic_lower_bound, b=dataset_config.ic_upper_bound)
        p_norm = p.norm(dim=1)
        z_norm = z.norm(dim=1)
        scaling = z_norm / p_norm * torch.rand(batch_size) * dataset_config.ood_sample_bound
        scaling = p * scaling.unsqueeze(-1)
        # scaling = abs(z) / abs(p) * torch.rand(batch_size).unsqueeze(-1) * dataset_config.ood_sample_bound
        return torch.hstack([z, p]), z, p * scaling

    def solve_integral_equation_batched(n_sample, u, z):
        P_D = torch.zeros((n_sample, n_state, n_point_delay))
        P_D[:, :, 0] = z
        for j in range(n_point_delay - 1):
            P_D_new = P_D.clone()
            P_D_new[:, :, j + 1] = P_D[:, :, j] \
                                   + dt * dataset_config.system.dynamic_tensor_batched1(P_D[:, :, j], j * dt, u[:, j])
            P_D = P_D_new
        p_generated = dataset_config.system.dynamic_tensor_batched2(P_D.permute(0, 2, 1), None, u).sum(dim=-1) * dt + z
        return p_generated

    class RandomDataset(Dataset):
        def __init__(self):
            self.num_samples = dataset_config.generation_net_dataset_size
            self.data = get_random_z_p(dataset_config.generation_net_dataset_size)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            inputs, z, p = self.data
            return inputs[idx], z[idx], p[idx]

    def regularization(u):
        regularization_type = dataset_config.regularization_type
        if regularization_type is None:
            return 0
        if regularization_type == 'total variation':
            return torch.abs(u[:, 1:] - u[:, :-1]).mean()
        else:
            raise NotImplementedError()

    dataloader = DataLoader(RandomDataset(), batch_size=batch_size, shuffle=False)
    bar = tqdm(range(n_epoch))

    optimizer = torch.optim.AdamW(model.parameters(), lr=dataset_config.generation_net_lr,
                                  weight_decay=dataset_config.generation_net_weight_decay)
    scheduler = get_lr_scheduler(optimizer, dataset_config)
    model.train()
    training_loss_list = []
    for epoch in bar:
        training_loss_list_in = []
        for inputs, z, p in dataloader:
            u = model(inputs)
            p_generated = solve_integral_equation_batched(len(inputs), u, z)
            ie_loss = (p - p_generated).norm(dim=1).mean()

            smooth_loss = regularization(u)
            loss = ie_loss + dataset_config.lamda * smooth_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss_list_in.append(loss.item())

        loss_epoch = sum(training_loss_list_in) / len(training_loss_list_in)
        training_loss_list.append(loss_epoch)
        scheduler.step()
        lr = scheduler.get_last_lr()[-1]
        bar.set_description(f'Epoch [{epoch + 1}/{n_epoch}] || Avg loss {loss_epoch:.6f} || Lr {lr:.6f}')
        wandb.log({
            "generation/loss": loss_epoch,
            "generation/lr": lr
        }, step=epoch)

    plt.title('training loss')
    plt.plot(training_loss_list)
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.savefig(f'{dataset_config.dataset_base_path}/loss.png')
    plt.clf()

    model.eval()
    with torch.no_grad():
        create_dataset_begin = time.time()
        inputs, z, p = get_random_z_p(n_sample)
        u = model(inputs)
        p_generated = solve_integral_equation_batched(n_sample, u, z)
        loss = (p - p_generated).norm(dim=1).mean()
        print(f'L2 error in generated dataset: {loss}')
        torch.hstack([torch.zeros_like(z)[:, 0:1], z, u])
        # all_samples = list(zip(torch.hstack([torch.zeros_like(z)[:, 0:1], z, u]), p))
        all_samples = list(zip(torch.hstack([torch.zeros_like(z)[:, 0:1], z, u]), p_generated))
        create_dataset_end = time.time()
        print(f'{create_dataset_end - create_dataset_begin} seconds used for creating dataset.')

    random.shuffle(all_samples)
    return all_samples


def main_(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig):
    training_dataloader, validating_dataloader = get_training_and_validation_datasets(dataset_config, train_config)
    testing_dataloader = get_test_datasets(dataset_config, train_config)

    check_dir(model_config.base_path)
    check_dir(train_config.model_save_path)
    model = run_train(dataset_config=dataset_config, model_config=model_config, train_config=train_config,
                      training_dataloader=training_dataloader, validating_dataloader=validating_dataloader,
                      testing_dataloader=testing_dataloader, img_save_path=model_config.base_path)
    metric_rd, metric_mse, n_success = run_test(m=model, dataset_config=dataset_config,
                                                base_path=model_config.base_path,
                                                debug=train_config.debug)
    return metric_rd, metric_mse, n_success


def main(sweep: bool = False):
    # setup_plt()
    dataset_config = DatasetConfig(
        recreate_training_dataset=True,
        recreate_testing_dataset=False,
        data_generation_strategy='nn',
        # data_generation_strategy='random',
        # dt=0.125,
        dt=0.05,
        n_dataset=1,
        n_sample_per_dataset=2000,
        append_training_dataset=False,
        n_plot_sample=10,
        filter_ood_sample=True,
        ood_sample_bound=0.2,
        generation_net_lr=1e-3,
        generation_net_n_epoch=100,
        generation_net_dataset_size=5000
    )
    model_config = ModelConfig(
        model_name='FNO',
        # model_name='FNOTwoStage',
        # model_name='PIFNO',
        fno_n_layers=5,
        fno_n_modes_height=16,
        fno_hidden_channels=64
    )
    train_config = TrainConfig(
        learning_rate=1e-4,
        n_epoch=500,
        batch_size=64,
        weight_decay=1e-4,
        log_step=-1,
        training_ratio=1.,
        load_model=False
    )
    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    if sweep:
        sweep_config = {
            'name': get_time_str(),
            'method': 'random',
            'metric': {
                'name': 'metric',
                'goal': 'minimize'
            },
            'parameters': {
                'fno_n_layers': {
                    'values': list(range(5, 10))
                },
                'fno_n_modes_height': {
                    'values': [16, 32, 64]
                },
                'fno_hidden_channels': {
                    'values': [16, 32, 64]
                },
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-4
                }
            }
        }
        sweep_id = wandb.sweep(sweep_config, project='no-sweep')

        def sweep_main(config=None):
            with wandb.init(config=config):
                config = wandb.config
                train_config.learning_rate = config.learning_rate
                model_config.fno_hidden_channels = config.fno_hidden_channels
                model_config.fno_n_layers = config.fno_n_layers
                model_config.fno_n_modes_height = config.fno_n_modes_height
                metric_rd, metric_mse, n_success = main_(dataset_config, model_config, train_config)
                wandb.log({
                    "metric": metric_mse
                })

        wandb.agent(sweep_id, sweep_main)
    else:
        wandb.init(
            project="no",
            name=get_time_str(),
            config={
                **asdict(dataset_config),
                **asdict(model_config),
                **asdict(train_config)
            }
        )
        metric_rd, metric_mse, n_success = main_(dataset_config, model_config, train_config)
        print(f'Relative L2 error: {metric_rd}')
        print(f'L2 error: {metric_mse}')
        print(f'#finished cases: {n_success}')


if __name__ == '__main__':
    main(sweep=False)
