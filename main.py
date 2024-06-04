import os
import pickle
import random
import time
from typing import Literal, Tuple, List

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
import dynamic_systems
from config import DatasetConfig, ModelConfig, TrainConfig
from dataset import ZUZDataset, ZUPDataset, PredictionDataset, sample_to_tensor
from dynamic_systems import solve_integral_eular, solve_integral_equation_neural_operator
from model import FNOProjection, FNOTwoStage, PIFNO, FullyConnectedNet, FourierNet, ChebyshevNet, BSplineNet
from utils import count_params, set_size, pad_leading_zeros, plot_comparison, plot_difference, \
    metric, check_dir, plot_sample, no_predict_and_loss, get_lr_scheduler, prepare_datasets, draw_distribution, \
    set_seed, plot_single, print_result


def run(dataset_config: DatasetConfig, Z0: Tuple, method: Literal['explicit', 'numerical', 'no', 'numerical_no'] = None,
        model=None, save_path: str = None, img_save_path: str = None):
    system: dynamic_systems.DynamicSystem = dataset_config.system
    n_point_delay = dataset_config.n_point_delay
    ts = dataset_config.ts
    Z0 = np.array(Z0)
    n_point = dataset_config.n_point
    U = np.zeros(n_point)
    Z = np.zeros((n_point, system.n_state))

    if method == 'explicit':
        P_explicit = np.zeros((n_point, system.n_state))
    else:
        P_explicit = None
    if method == 'numerical' or method == 'numerical_no':
        P_numerical = np.zeros((n_point, system.n_state))
    else:
        P_numerical = None
    if method == 'no' or method == 'numerical_no':
        P_no = np.zeros((n_point, system.n_state))
    else:
        P_no = None
    Z[n_point_delay, :] = Z0
    for t_i in range(dataset_config.n_point):
        t_minus_D_i = max(t_i - n_point_delay, 0)
        t = ts[t_i]
        if method == 'explicit':
            U[t_i] = system.U_explicit(t, Z0)
            if t_i > n_point_delay:
                Z[t_i, :] = system.Z_explicit(t, Z0)
        elif method == 'numerical':
            if t_i > n_point_delay:
                Z[t_i, :] = odeint(system.dynamic, Z[t_i - 1, :], [ts[t_i - 1], ts[t_i]], args=(U[t_minus_D_i - 1],))[1]
                Z_t = Z[t_i, :]
            else:
                Z_t = Z0
            P_numerical[t_i, :] = dynamic_systems.solve_integral(
                Z_t=Z_t, P_D=P_numerical[t_minus_D_i:t_i], U_D=U[t_minus_D_i:t_i], t=t, dataset_config=dataset_config)
            if t_i > n_point_delay:
                U[t_i] = system.kappa(P_numerical[t_i, :])
        elif method == 'no':
            if t_i > n_point_delay:
                Z[t_i, :] = odeint(system.dynamic, Z[t_i - 1, :], [ts[t_i - 1], ts[t_i]], args=(U[t_minus_D_i - 1],))[1]
                Z_t = Z[t_i, :]
            else:
                Z_t = Z0
            P_no[t_i, :] = solve_integral_equation_neural_operator(
                model=model, U_D=pad_leading_zeros(segment=U[t_minus_D_i:t_i], length=n_point_delay), Z_t=Z_t, t=t)
            if t_i > n_point_delay:
                U[t_i] = system.kappa(P_no[t_i, :])
        elif method == 'numerical_no':
            if t_i > n_point_delay:
                Z[t_i, :] = odeint(system.dynamic, Z[t_i - 1, :], [ts[t_i - 1], ts[t_i]],
                                   args=(U[t_minus_D_i - 1],))[1]
                Z_t = Z[t_i, :]
            else:
                Z_t = Z0
            P_numerical[t_i, :] = dynamic_systems.solve_integral(
                Z_t=Z_t, P_D=P_numerical[t_minus_D_i:t_i], U_D=U[t_minus_D_i:t_i], t=t, dataset_config=dataset_config)
            P_no[t_i, :] = solve_integral_equation_neural_operator(
                model=model, U_D=pad_leading_zeros(segment=U[t_minus_D_i:t_i], length=n_point_delay), Z_t=Z_t, t=t)
            if t_i > n_point_delay:
                U[t_i] = system.kappa(P_numerical[t_i, :])
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
        comparison_full = f'{img_save_path}/comparison_full.png'
        comparison_zoom = f'{img_save_path}/comparison_zoom.png'
        difference_full = f'{img_save_path}/difference_full.png'
        difference_zoom = f'{img_save_path}/difference_zoom.png'
        u_path = f'{img_save_path}/u.png'
        plot_comparison(ts, P_no, P_numerical, P_explicit, Z, delay, n_point_delay, comparison_full,
                        dataset_config.n_state)
        plot_comparison(ts, P_no, P_numerical, P_explicit, Z, delay, n_point_delay, comparison_zoom,
                        dataset_config.n_state, [-5, 5])
        plot_difference(ts, P_no, P_numerical, P_explicit, Z, n_point_delay, difference_full, dataset_config.n_state)
        plot_difference(ts, P_no, P_numerical, P_explicit, Z, n_point_delay, difference_zoom, dataset_config.n_state,
                        [-5, 5])
        plot_single(ts, U, '$U(t)$', u_path)

    if method == 'explicit':
        return U, Z, P_explicit
    elif method == 'no' or method == 'numerical_no':
        return U, Z, P_no
    elif method == 'numerical':
        return U, Z, P_numerical
    else:
        raise NotImplementedError()


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
    n_params = count_params(model)
    print(f'#parameters: {n_params}')
    np.savetxt(f'{train_config.model_save_path}/{model_config.model_name}.txt', np.array([n_params]))
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
            desc = f'Epoch [{epoch + 1}/{n_epoch}] || Lr: {lr:6f} || Training loss: {training_loss_t:.6f}'
            # wandb.log({
            #     'train/training loss': training_loss_t,
            #     'train/lr': lr,
            #     'train/epoch': epoch + 1
            # })
            if do_validation:
                desc += f' || Validation loss: {validating_loss_t:.6f}'
                # wandb.log({
                #     'train/validation loss': validating_loss_t,
                #     'train/epoch': epoch + 1
                # })
            if do_testing:
                desc += f' || Test loss: {testing_loss_t:.6f}'
                # wandb.log({
                #     'train/test loss': testing_loss_t,
                #     'train/epoch': epoch + 1
                # })
            bar.set_description(desc)

            if (train_config.log_step > 0 and epoch % train_config.log_step == 0) or epoch == n_epoch - 1:
                rl2, l2, _, n_success = run_test(model, dataset_config, method='no', base_path=model_config.base_path,
                                                 silence=True)
                rl2_list.append(rl2)
                l2_list.append(l2)
                # wandb.log({
                #     'train/Relative L2 error': rl2,
                #     'train/L2 error': l2,
                #     'train/n_success': n_success,
                #     'train/epoch': epoch + 1
                # })
            scheduler.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], train_config.scheduler_min_lr)
        draw()
        torch.save(model.state_dict(), f'{train_config.model_save_path}/{model_config.model_name}.pth')
    print('Finished Training')
    return model


def run_test(m, dataset_config: DatasetConfig, method: str, base_path: str = None, silence: bool = False,
             test_points: List = None, plot: bool = False):
    base_path = f'{base_path}/{method}'
    if test_points is None:
        test_points = dataset_config.test_points
    bar = test_points if silence else tqdm(test_points)
    rl2_list = []
    l2_list = []
    runtime_list = []
    for test_point in bar:
        if not silence:
            bar.set_description(f'Solving system with initial point {np.round(test_point, decimals=2)}.')

        img_save_path = f'{base_path}/{np.round(test_point, decimals=2)}'
        check_dir(img_save_path)
        begin = time.time()
        U, Z, P = run(dataset_config=dataset_config, model=m, Z0=test_point, method=method, img_save_path=img_save_path)
        end = time.time()
        runtime = end - begin
        plt.close()
        n_point_delay = dataset_config.n_point_delay
        rl2, l2 = metric(P[n_point_delay:-n_point_delay], Z[2 * n_point_delay:])

        if np.isinf(rl2) or np.isnan(rl2):
            if not silence:
                print(f'[WARNING] Running with initial condition Z = {test_point} with method [{method}] failed.')
            continue
        np.savetxt(f'{img_save_path}/metric.txt', np.array([rl2, l2, runtime]))
        rl2_list.append(rl2)
        l2_list.append(l2)
        runtime_list.append(runtime)
    rl2 = np.nanmean(rl2_list).item()
    l2 = np.nanmean(l2_list).item()
    runtime = np.nanmean(runtime_list).item()
    np.savetxt(f'{base_path}/metric.txt', np.array([rl2, l2, runtime]))
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

        plot_result(data=rl2_list, label=f'Relative L2 error', title='Relative L2 error',
                    xlabel='Relative L2 error', path=f'{base_path}/rl2.png')
        plot_result(data=l2_list, label=f'L2 error', title='L2 error', xlabel='L2 error',
                    path=f'{base_path}/l2.png')
    return rl2, l2, runtime, len(rl2_list)


def get_training_and_validation_datasets(dataset_config: DatasetConfig, train_config: TrainConfig):
    create_dataset = not os.path.exists(
        dataset_config.training_dataset_file) or dataset_config.recreate_training_dataset

    def load():
        print('Loading training dataset')
        training_samples_loaded = torch.load(dataset_config.training_dataset_file)
        print(f'Loaded {len(training_samples_loaded)} samples')
        return training_samples_loaded

    if create_dataset:
        print('Creating training dataset')
        check_dir(dataset_config.dataset_base_path)
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

    for i, (feature, label) in enumerate(training_samples[:dataset_config.n_plot_sample]):
        plot_sample(feature, label, dataset_config, f'{str(i)}.png')
    draw_distribution(training_samples, dataset_config.n_state, dataset_config.dataset_base_path)
    training_dataloader, validating_dataloader = prepare_datasets(
        training_samples, train_config.training_ratio, train_config.batch_size, train_config.device)
    print(f'#Training sample: {int(len(training_samples) * train_config.training_ratio)}')
    print(f'#Validating sample: {int(len(training_samples) * (1 - train_config.training_ratio))}')
    path = dataset_config.training_dataset_file
    if dataset_config.recreate_training_dataset:
        torch.save(training_samples, path)
        np.savetxt(f'{dataset_config.dataset_base_path}/n_sample.txt',
                   np.array([len(training_samples), train_config.training_ratio]))
        print(f'{len(training_samples)} samples saved')
    return training_dataloader, validating_dataloader


def get_test_datasets(dataset_config, train_config):
    if not train_config.do_test:
        return None
    if not os.path.exists(dataset_config.testing_dataset_file) or dataset_config.recreate_testing_dataset:
        print('Creating testing dataset')
        testing_samples = create_trajectory_dataset(dataset_config, test_points=dataset_config.test_points)
    else:
        print('Loading testing dataset')
        testing_samples = torch.load(dataset_config.testing_dataset_file)
    testing_dataloader = DataLoader(PredictionDataset(testing_samples), batch_size=train_config.batch_size,
                                    shuffle=False)
    print(f'#Testing sample: {len(testing_samples)}')
    return testing_dataloader


def create_trajectory_dataset(dataset_config: DatasetConfig, test_points: List = None):
    all_samples = []
    if dataset_config.z_u_p_pair:
        print('creating datasets of Z(t), U(t-D~t), P(t) pairs')
    else:
        print('creating datasets of Z(t), U(t-D~t), Z(t+D) pairs')
    if test_points is None:
        bar = tqdm(list(
            np.random.uniform(dataset_config.ic_lower_bound, dataset_config.ic_upper_bound,
                              (dataset_config.n_dataset, dataset_config.n_state))))
    else:
        bar = tqdm(test_points)
    for i, Z0 in enumerate(bar):
        img_save_path = f'{dataset_config.dataset_base_path}/example/{str(i)}'
        check_dir(img_save_path)
        if dataset_config.z_u_p_pair:
            U, Z, P = run(method='numerical', Z0=Z0, dataset_config=dataset_config, img_save_path=img_save_path)
            dataset = ZUPDataset(
                torch.tensor(Z, dtype=torch.float32), torch.tensor(U, dtype=torch.float32),
                torch.tensor(P, dtype=torch.float32), dataset_config.n_point_delay, dataset_config.dt)
        else:
            U, Z, _ = run(method='explicit', Z0=Z0, dataset_config=dataset_config, img_save_path=img_save_path)
            dataset = ZUZDataset(
                torch.tensor(Z, dtype=torch.float32), torch.tensor(U, dtype=torch.float32),
                dataset_config.n_point_delay, dataset_config.dt)
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
    n_state: int = dataset_config.n_state
    all_samples = []

    def u(random_u_type: str = dataset_config.random_u_type):
        if random_u_type == 'line':
            def func(x):
                return (np.random.uniform(-1, 1, 2).reshape(-1, 1) * np.array([x, np.ones_like(x)])).sum(axis=0)
        elif random_u_type == 'poly':
            def func(x):
                return 3 * (np.random.uniform(-1, 1, 5).reshape(-1, 1) * np.array(
                    [x ** 4, x ** 3, x ** 2, x, np.ones_like(x)])).sum(axis=0)
        elif random_u_type == 'sin':
            def func(x):
                return np.sin(np.random.uniform(0, 5) * x)
        elif random_u_type == 'exp':
            def func(x):
                return np.exp(-np.random.uniform(0, 5) * x)
        elif random_u_type == 'chebyshev':
            def func(x):
                return np.random.uniform(0, 5) * np.cos(np.random.uniform(0, 5) * np.arccos(x))
        elif random_u_type == 'spline':
            def func(x):
                segment_length = np.random.uniform(0, 2)
                start = segment_length * np.floor(np.min(x) / segment_length)
                end = segment_length * np.ceil(np.max(x) / segment_length)

                key_points = np.arange(start, end + segment_length, segment_length)
                random_values = np.random.uniform(-10, 10, size=key_points.shape[0])

                indices = np.searchsorted(key_points, x, side='right') - 1
                x1 = key_points[indices]
                x2 = key_points[indices + 1]
                y1 = random_values[indices]
                y2 = random_values[indices + 1]

                y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
                return y
        elif random_u_type == 'sinexp':
            def func(x):
                feq = np.random.uniform(0, 100)
                k = np.random.uniform(0, 100)
                shift = np.random.uniform(0, 100)
                return np.exp(-k * x) * np.sin(feq * (x + shift))
        elif random_u_type == 'sparse':
            def create_almost_zero_array(arr):
                new_arr = np.zeros_like(arr)
                num_zeros = int(arr.size * np.random.rand())
                if num_zeros > 0:
                    new_arr[-num_zeros:] = np.random.rand(num_zeros)
                return new_arr

            f_spline = u('spline')

            def func(x):
                # return create_almost_zero_array(x)
                u_ = f_spline(x) * 0.2
                num_zeros = int(len(u_) * np.random.rand())
                u_[:num_zeros] = 0
                return u_
        else:
            raise NotImplementedError()
        return func

    def get_random_sample(func):
        # Z_t = np.random.uniform(dataset_config.ic_lower_bound, dataset_config.ic_upper_bound, 2)
        Z_t = np.random.randn(2) * max(abs(dataset_config.ic_lower_bound), abs(dataset_config.ic_upper_bound))
        U_D = func(np.linspace(0, 1, n_point_delay))
        P_t = solve_integral_eular(Z_t=Z_t, U_D=U_D, dt=dt, n_state=n_state, n_points=n_point_delay,
                                   f=dataset_config.system.dynamic)
        features = sample_to_tensor(Z_t, U_D, dt * n_point_delay)
        sample = (features, torch.from_numpy(P_t))
        return sample, features, Z_t, U_D, P_t

    print(f'Generating dataset using {dataset_config.random_u_type}')
    f = u()
    n_sample = dataset_config.n_dataset * dataset_config.n_sample_per_dataset
    n_id_sample = 0
    n_total_sample = 0
    bar = tqdm(total=n_sample)
    print(f'Begin generating {n_sample} samples using {dataset_config.random_u_type}')
    while n_id_sample < n_sample:
        sample, _, Z_t, _, P_t = get_random_sample(f)
        if dataset_config.filter_ood_sample:
            def filter_out_of_distribution_sample(factor, p):
                return np.linalg.norm(P_t) / np.linalg.norm(Z_t) <= factor \
                    and np.random.uniform(0, 1) <= p

            if filter_out_of_distribution_sample(dataset_config.ood_sample_bound, 1):
                all_samples.append(sample)
                n_id_sample += 1
                bar.update(1)
        else:
            all_samples.append(sample)
            n_id_sample += 1
            bar.update(1)
        n_total_sample += 1
    print(
        f'Generated {n_total_sample} samples in total, {n_id_sample} samples in distribution,'
        f' percent: {n_id_sample / n_total_sample}')

    f_sparse = u('sparse')
    print(f'Begin generating {dataset_config.n_sample_sparse} samples with sparse U')
    for _ in tqdm(list(range(dataset_config.n_sample_sparse))):
        sample, _, _, _, _ = get_random_sample(f_sparse)
        all_samples.append(sample)

    random.shuffle(all_samples)
    print(f'{len(all_samples)} samples in total')
    return all_samples


def create_nn_dataset(dataset_config: DatasetConfig):
    dt: float = dataset_config.dt
    n_point_delay: int = dataset_config.n_point_delay
    n_sample_per_dataset: int = dataset_config.n_sample_per_dataset
    n_state: int = dataset_config.n_state
    n_sample = dataset_config.n_dataset * n_sample_per_dataset
    print(f'Generating dataset using [{dataset_config.net_type}] network')
    if dataset_config.net_type == 'fc':
        model = FullyConnectedNet(dataset_config.n_state, dataset_config.n_point_delay)
    elif dataset_config.net_type == 'fourier':
        model = FourierNet(dataset_config.n_state, dataset_config.fourier_n_mode,
                           np.linspace(0, dataset_config.delay, dataset_config.n_point_delay))
    elif dataset_config.net_type == 'chebyshev':
        model = ChebyshevNet(dataset_config.n_state, dataset_config.chebyshev_n_term,
                             np.linspace(0, 1, dataset_config.n_point_delay))
    elif dataset_config.net_type == 'bspline':
        model = BSplineNet(dataset_config.n_state, dataset_config.bspline_n_knot, dataset_config.bspline_degree,
                           np.linspace(0, dataset_config.delay, dataset_config.n_point_delay))
    else:
        raise NotImplementedError()
    batch_size = dataset_config.net_batch_size
    n_epoch = dataset_config.net_n_epoch

    class RandomDataset(Dataset):
        def __init__(self):
            self.num_samples = dataset_config.net_dataset_size
            self.data = get_random_z_p(dataset_config.net_dataset_size)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            inputs, z, p = self.data
            return inputs[idx], z[idx], p[idx]

    def torch_uniform(shape, a, b):
        rand_tensor = torch.rand(shape)
        transformed_tensor = a + (b - a) * rand_tensor
        return transformed_tensor

    def get_random_z_p(batch_size):
        # z = torch_uniform(shape=(batch_size, 2), a=dataset_config.ic_lower_bound, b=dataset_config.ic_upper_bound)
        # p = torch_uniform(shape=(batch_size, 2), a=dataset_config.ic_lower_bound, b=dataset_config.ic_upper_bound)
        scale = max(abs(dataset_config.ic_lower_bound), abs(dataset_config.ic_upper_bound))
        z = torch.randn(size=(batch_size, 2)) * scale
        p = torch.randn(size=(batch_size, 2)) * scale
        # p_norm = p.norm(dim=1)
        # z_norm = z.norm(dim=1)
        # scaling = z_norm / p_norm * torch.rand(batch_size) * dataset_config.ood_sample_bound
        # scaling = p * scaling.unsqueeze(-1)
        scaling = abs(z) / abs(p) * torch.rand(batch_size).unsqueeze(-1) * dataset_config.ood_sample_bound
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

    def regularization(u):
        regularization_type = dataset_config.regularization_type
        if regularization_type is None:
            return 0
        if regularization_type == 'total variation':
            return torch.abs(u[:, 1:] - u[:, :-1]).mean()
        elif regularization_type == 'dirichlet energy':
            diff_f = u[1:] - u[:-1]
            gradients = diff_f / dataset_config.dt
            return (gradients ** 2).sum()
        else:
            raise NotImplementedError()

    model_save_path = f'{dataset_config.dataset_base_path}/{dataset_config.net_type}.pth'
    if not dataset_config.load_net:
        dataloader = DataLoader(RandomDataset(), batch_size=batch_size, shuffle=False)
        bar = tqdm(range(n_epoch))

        optimizer = torch.optim.AdamW(model.parameters(), lr=dataset_config.net_lr,
                                      weight_decay=dataset_config.net_weight_decay)
        scheduler = get_lr_scheduler(optimizer, dataset_config)
        model.train()
        training_total_loss_list = []
        training_smooth_loss_list = []
        training_ie_loss_list = []
        for epoch in bar:
            smooth_loss_list = []
            ie_loss_list = []
            total_loss_list = []
            for inputs, z, p in dataloader:
                u = model(inputs)
                p_true = solve_integral_equation_batched(len(inputs), u, z)
                ie_loss = (p - p_true).norm(dim=1).mean()

                smooth_loss = regularization(u)
                loss = ie_loss + dataset_config.lamda * smooth_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                smooth_loss_list.append(smooth_loss.item())
                ie_loss_list.append(ie_loss.item())
                total_loss_list.append(loss.item())

            total_loss_avg = sum(total_loss_list) / len(total_loss_list)
            smooth_loss_avg = sum(smooth_loss_list) / len(smooth_loss_list)
            ie_loss_avg = sum(ie_loss_list) / len(ie_loss_list)
            training_total_loss_list.append(total_loss_avg)
            training_smooth_loss_list.append(smooth_loss_avg)
            training_ie_loss_list.append(ie_loss_avg)
            scheduler.step()
            lr = scheduler.get_last_lr()[-1]
            bar.set_description(f'Epoch [{epoch + 1}/{n_epoch}] || Total loss {total_loss_avg:.6f} '
                                f'|| Smooth loss {smooth_loss_avg:.6f} || IE loss {ie_loss_avg:.6f} || Lr {lr:.6f}')
            # wandb.log({
            #     "generation/total loss": total_loss_avg,
            #     "generation/integral equation loss": ie_loss_avg,
            #     "generation/smoothness loss": smooth_loss_avg,
            #     "generation/lr": lr,
            #     "generation/epoch": epoch
            # })

        plt.title('training loss')
        plt.plot(training_total_loss_list, label='Total loss')
        plt.plot(training_ie_loss_list, label='Integral equation loss')
        plt.plot(training_smooth_loss_list, label='Smoothness loss')
        plt.legend()
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.savefig(f'{dataset_config.dataset_base_path}/loss.png')
        plt.clf()
        torch.save(model.state_dict(), model_save_path)
    else:
        print('Load generation model from', model_save_path)
        model.load_state_dict(torch.load(model_save_path))

    model.eval()
    with torch.no_grad():
        create_dataset_begin = time.time()
        inputs, z, p = get_random_z_p(n_sample)
        u = model(inputs)
        p_true = solve_integral_equation_batched(n_sample, u, z)
        loss = (p - p_true).norm(dim=1).mean()
        print(f'L2 error in generated dataset: {loss}')
        all_samples = list(zip(torch.hstack([torch.zeros_like(z)[:, 0:1], z, u]).clone(), p_true.clone()))
        create_dataset_end = time.time()
        print(f'{create_dataset_end - create_dataset_begin} seconds used for creating dataset.')

    random.shuffle(all_samples)
    return all_samples


def main(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig):
    training_dataloader, validating_dataloader = get_training_and_validation_datasets(dataset_config, train_config)
    testing_dataloader = get_test_datasets(dataset_config, train_config)

    check_dir(model_config.base_path)
    check_dir(train_config.model_save_path)
    model = run_train(dataset_config=dataset_config, model_config=model_config, train_config=train_config,
                      training_dataloader=training_dataloader, validating_dataloader=validating_dataloader,
                      testing_dataloader=testing_dataloader, img_save_path=model_config.base_path)
    return (
        run_test(m=model, dataset_config=dataset_config, base_path=model_config.base_path, method='no'),
        run_test(m=model, dataset_config=dataset_config, base_path=model_config.base_path, method='numerical'),
        run_test(m=model, dataset_config=dataset_config, base_path=model_config.base_path, method='numerical_no')
    )


if __name__ == '__main__':
    set_seed(0)
    dataset_config, model_config, train_config = config.get_config()
    print(f'Running with system {config.system}')
    result_no, result_numerical, result_numerical_no = main(dataset_config, model_config, train_config)
    print('NO Result')
    print_result(result_no, dataset_config)
    print('Numerical Result')
    print_result(result_numerical, dataset_config)
    print('Numerical-NO Result')
    print_result(result_numerical_no, dataset_config)
