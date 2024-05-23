import os
import pickle
import random
from dataclasses import asdict
from typing import Literal, Tuple, List

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from scipy.integrate import odeint
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DatasetConfig, ModelConfig, TrainConfig
from dataset import ImplicitDataset, ExplictDataset, PredictionDataset, sample_to_tensor
from dynamic_systems import DynamicSystem, predict_integral, predict_integral_general, predict_neural_operator
from model import FNOProjection, FNOTwoStage, PIFNO
from utils import count_params, get_time_str, set_size, setup_plt, pad_leading_zeros


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


def run(dataset_config: DatasetConfig, Z0: Tuple, model=None,
        method: Literal['explict', 'numerical', 'no', 'numerical_no'] = None, title='', save_path: str = None,
        img_save_path: str = None):
    system = dataset_config.system
    n_point_delay = dataset_config.n_point_delay
    c = dataset_config.system_c
    n = dataset_config.system_n
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
            P[t_i, :] = predict_integral_general(f=system.dynamic, Z_t=Z_t, P_D=P[t_minus_D_i:t_i],
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
            P[t_i, :] = predict_neural_operator(model=model, U_D=pad_leading_zeros(segment=U[t_minus_D_i:t_i],
                                                                                   length=n_point_delay),
                                                Z_t=Z_t, t=t)
            P_compare[t_i, :] = predict_integral_general(f=system.dynamic, Z_t=Z_t, P_D=P[t_minus_D_i:t_i],
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
            P[t_i, :] = predict_integral_general(f=system.dynamic, Z_t=Z_t, P_D=P[t_minus_D_i:t_i],
                                                 U_D=U[t_minus_D_i:t_i], dt=dt,
                                                 t=t) + dataset_config.noise()
            P_compare[t_i, :] = predict_neural_operator(
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

        if method != 'numerical_no':
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


def no_predict(inputs, model):
    time_step = inputs[:, :1]
    z_u = inputs[:, 1:]

    inputs = [z_u, time_step]
    if not isinstance(model, dde.nn.DeepONet):
        inputs = z_u
    return model(inputs)


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
            n_modes_height=n_modes_height, hidden_channels=hidden_channels, in_features=n_state + n_point_delay,
            out_features=n_state, n_layers=n_layers, dt=dataset_config.dt, n_state=dataset_config.n_state,
            dynamic=DynamicSystem).to(device)
    else:
        raise NotImplementedError()
    print(f'#parameters: {count_params(model)}')
    pth = f'{train_config.model_save_path}/{model_config.model_name}.pth'
    if train_config.load_model and os.path.exists(pth):
        model.load_state_dict(torch.load(f'{train_config.model_save_path}/{model_name}.pth'))
        print(f'Model loaded from {pth}, skip training!')
    else:
        if hasattr(model, 'loss'):
            criterion = model.loss
        else:
            criterion = torch.nn.MSELoss()
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
            for inputs, label in training_dataloader:
                inputs, label = inputs.to(device), label.to(device)
                outputs = no_predict(inputs, model)
                loss = criterion(outputs, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            model.eval()
            with torch.no_grad():
                if do_validation:
                    validating_loss = 0.0
                    for inputs, label in validating_dataloader:
                        inputs, label = inputs.to(device), label.to(device)
                        outputs = no_predict(inputs, model)
                        loss = criterion(outputs, label)
                        validating_loss += loss.item()

                if do_testing:
                    testing_loss = 0.0
                    for inputs, label in testing_dataloader:
                        inputs, label = inputs.to(device), label.to(device)
                        outputs = no_predict(inputs, model)
                        loss = criterion(outputs, label)
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
                wandb.log({
                    'Relative L2 error': rl2,
                    'L2 error': l2
                }, step=epoch)
            scheduler.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], train_config.scheduler_min_lr)
        draw()
        torch.save(model.state_dict(), f'{train_config.model_save_path}/{model_config.model_name}.pth')
    print('Finished Training')
    return model


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def metric(preds, trues):
    N = trues.shape[0]
    preds = np.atleast_2d(preds)
    trues = np.atleast_2d(trues)
    return np.sum(np.linalg.norm(preds - trues, axis=1) / np.linalg.norm(trues, axis=1)) / N, \
           np.sum(np.linalg.norm(preds - trues, axis=1)) / N


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
            # plt.ylim([0, 20])
            # plt.xlim([0, 5])
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


def postprocess(samples, dataset_config: DatasetConfig):
    print('[DEBUG] postprocessing')
    new_samples = []
    random.shuffle(samples)
    for sample in samples:
        feature = sample[0].cpu().numpy()
        t = feature[:1]
        z = feature[1:3]
        u = feature[3:]
        p = sample[1].cpu().numpy()
        p = predict_integral(Z_t=z, U_D=u, dt=dataset_config.dt, n_state=2,
                             n_point_delay=dataset_config.n_point_delay, dynamic=dataset_config.system.dynamic)
        new_samples.append((torch.from_numpy(np.concatenate([t, z, u])), torch.tensor(p)))
    print(f'[WARNING] {len(new_samples)} samples replaced by numerical solutions')
    all_samples = new_samples
    return all_samples


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
        if dataset_config.trajectory:
            training_samples = create_trajectory_dataset(dataset_config)
        else:
            training_samples = create_stateless_dataset(dataset_config)
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

    if dataset_config.trajectory and dataset_config.postprocess:
        training_samples = postprocess(training_samples, dataset_config)
    for feature, label in training_samples[:dataset_config.n_plot_sample]:
        plot_sample(feature, label, dataset_config)

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
                                    shuffle=False,
                                    # generator=torch.Generator(device=train_config.device)
                                    )
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


def create_stateless_dataset(dataset_config: DatasetConfig, filter_ood_sample=True):
    dt: float = dataset_config.dt
    n_point_delay: int = dataset_config.n_point_delay
    n_sample_per_dataset: int = dataset_config.n_sample_per_dataset
    n_state: int = dataset_config.n_state
    all_samples = []
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
            P_t = predict_integral(Z_t=Z_t, U_D=U_D, dt=dt, n_state=n_state, n_point_delay=n_point_delay,
                                   dynamic=DynamicSystem.dynamic_static)
            features = sample_to_tensor(Z_t, U_D, dt * n_point_delay)
            if filter_ood_sample:
                def filter_out_of_distribution_sample(factor, p):
                    return abs(P_t[0]) <= abs(Z_t[0]) * factor and abs(P_t[1]) <= abs(
                        Z_t[1]) * factor and np.random.uniform(0, 1) <= p

                if filter_out_of_distribution_sample(0.1, 1):
                    all_samples.append((features, torch.from_numpy(P_t)))
                    j += 1
            else:
                all_samples.append((features, torch.from_numpy(P_t)))
                j += 1
    random.shuffle(all_samples)
    return all_samples


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
        recreate_training_dataset=False,
        recreate_testing_dataset=False,
        trajectory=False,
        dt=0.125,
        n_dataset=2000,
        duration=8,
        delay=3.,
        n_sample_per_dataset=1,
        ic_lower_bound=-2,
        ic_upper_bound=2,
        system_n=3,
        system_c=5,
        append_training_dataset=False
    )
    model_config = ModelConfig(
        # model_name='FNO',
        model_name='FNOTwoStage',
        fno_n_layers=2,
        fno_n_modes_height=4,
        fno_hidden_channels=16
    )
    train_config = TrainConfig(
        learning_rate=1e-3,
        n_epoch=400,
        batch_size=64,
        weight_decay=0.0001376,
        log_step=100,
        training_ratio=1.,
        load_model=False
    )
    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    if sweep:
        sweep_config = {
            'name': get_time_str(),
            'method': 'bayes',
            'metric': {
                'name': 'metric',
                'goal': 'minimize'
            },
            'parameters': {
                'fno_n_layers': {
                    'values': [1, 2]
                },
                'fno_n_modes_height': {
                    'values': [8, 16, 32]
                },
                'fno_hidden_channels': {
                    'values': [16, 32, 64]
                },
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-4,
                    'max': 1e-3
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
