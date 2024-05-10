import os
import pickle
import random
from typing import Literal, Tuple, List

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DatasetConfig, ModelConfig, TrainConfig
from dataset import ImplicitDataset, ExplictDataset, PredictionDataset, sample_to_tensor
from dynamic_systems import DynamicSystem, predict_integral
from model import FNOProjection, FNOSum
from utils import count_params


def plot_comparison(ts, P, P_compare, Z, delay, n_point_delay, save_path):
    plt.title('Comparison')
    for t_i in range(2):
        if P_compare is not None:
            plt.plot(ts[n_point_delay:], P_compare[:-n_point_delay, t_i], label=f'$PNO_{t_i + 1}(t-{delay})$')
        plt.plot(ts[n_point_delay:], P[:-n_point_delay, t_i], label=f'$\hat P_{t_i + 1}(t-{delay})$')
        plt.plot(ts[n_point_delay:], Z[n_point_delay:, t_i], label=f'$Z_{t_i + 1}(t)$')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def plot_difference(ts, P, P_compare, Z, n_point_delay, save_path):
    plt.ylim([-1, 1])
    difference = P[:-n_point_delay] - Z[n_point_delay:]
    plt.plot(ts[n_point_delay:], difference[:, 0], label='$\delta P_1$')
    plt.plot(ts[n_point_delay:], difference[:, 1], label='$\delta P_2$')
    if P_compare is not None:
        difference_no = P_compare[:-n_point_delay] - Z[n_point_delay:]
        plt.plot(ts[n_point_delay:], difference_no[:, 0], label='$\delta PNO_1$')
        plt.plot(ts[n_point_delay:], difference_no[:, 1], label='$\delta PNO_2$')

    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def run(dataset_config: DatasetConfig, Z0: Tuple, model=None,
        method: Literal['explict', 'numerical', 'no', 'numerical_no'] = None, title='', save_path: str = None,
        img_save_path: str = None):
    system = DynamicSystem(Z0=np.array(list(Z0)), dataset_config=dataset_config, method=method)
    for _ in range(dataset_config.n_point):
        system.step(model)

    U = system.U
    Z = system.Z
    P = system.P
    P_compare = system.P_compare
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

        plt.figure(figsize=(10, 8))
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

        if img_save_path is not None:
            plt.savefig(f'{img_save_path}/system.png')
            plt.clf()
        else:
            plt.show()
        if method != 'numerical_no':
            P_compare = None
        plot_comparison(ts, P, P_compare, Z, delay, n_point_delay,
                        f'{img_save_path}/comparison_full.png' if img_save_path is not None else None)
        plt.ylim([-5, 5])
        plot_comparison(ts, P, P_compare, Z, delay, n_point_delay,
                        f'{img_save_path}/comparison_zoom.png' if img_save_path is not None else None)

        plot_difference(ts, P, P_compare, Z, n_point_delay,
                        f'{img_save_path}/difference_full.png' if img_save_path is not None else None)
        plt.ylim([-1, 1])
        plot_difference(ts, P, P_compare, Z, n_point_delay,
                        f'{img_save_path}/difference_zoom.png' if img_save_path is not None else None)

    if method == 'explict':
        return U, Z, None
    return U, Z, P


def no_predict(inputs, model):
    time_step = inputs[:, :1]
    z_u = inputs[:, 1:]

    inputs = [z_u, time_step]
    if isinstance(model, FNOProjection):
        inputs = z_u
    return model(inputs)


def run_train(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig,
              training_dataloader=None, validating_dataloader=None, testing_dataloader=None, img_save_path: str = None):
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
            n_point_delay=n_point_delay, n_layers=n_layers, dt=dataset_config.dt).to(device)
    elif model_name == 'FNOSum':
        model = FNOSum(
            n_modes_height=n_modes_height, hidden_channels=hidden_channels, in_features=n_state + n_point_delay,
            out_features=n_state, n_layers=n_layers).to(device)
    else:
        raise NotImplementedError()
    print(f'#parameters: {count_params(model)}')
    pth = f'{train_config.model_save_path}/{model_config.model_name}.pth'
    if train_config.load_model and os.path.exists(pth):
        model.load_state_dict(torch.load(f'{train_config.model_save_path}/{model_name}.pth'))
        print(f'Model loaded from {pth}, skip training!')
    else:
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_config.scheduler_step_size,
                                                    gamma=train_config.scheduler_gamma)
        training_loss_arr = []
        validating_loss_arr = []
        testing_loss_arr = []
        bar = tqdm(list(range(n_epoch)))
        for epoch in bar:
            model.train()
            training_loss = 0.0
            for inputs, label in training_dataloader:
                outputs = no_predict(inputs, model)
                loss = criterion(outputs, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            model.eval()
            with torch.no_grad():
                validating_loss = 0.0
                for inputs, label in validating_dataloader:
                    outputs = no_predict(inputs, model)
                    loss = criterion(outputs, label)
                    validating_loss += loss.item()

                if testing_dataloader is not None:
                    testing_loss = 0.0
                    for inputs, label in testing_dataloader:
                        outputs = no_predict(inputs, model)
                        loss = criterion(outputs, label)
                        testing_loss += loss.item()

            training_loss_t = training_loss / len(training_dataloader)
            training_loss_arr.append(training_loss_t)
            validating_loss_t = validating_loss / len(validating_dataloader)
            validating_loss_arr.append(validating_loss_t)
            if testing_dataloader is not None:
                testing_loss_t = testing_loss / len(testing_dataloader)
                testing_loss_arr.append(testing_loss_t)
            desc = f'Epoch [{epoch + 1}/{n_epoch}] || Learning rate: {scheduler.get_last_lr()} || ' \
                   f'Training loss: {training_loss_t:.6f} || Validation loss: {validating_loss_t:.6f}'
            if testing_dataloader is not None:
                desc += f' || Test loss: {testing_loss_t:.6f}'
            bar.set_description(desc)
            if epoch % train_config.log_step == 0 or epoch == n_epoch - 1:
                metric_list = run_test(model, dataset_config, plot=True, silence=True)
                print(np.nanmean(metric_list))
                plt.plot(training_loss_arr, label="Training loss")
                plt.plot(validating_loss_arr, label="Validation loss")
                if testing_dataloader is not None:
                    plt.plot(testing_loss_arr, label="Test loss")
                plt.yscale("log")
                plt.legend()
                if img_save_path is not None:
                    plt.savefig(f'{img_save_path}/loss.png')
                    plt.clf()
                else:
                    plt.show()
                torch.save(model.state_dict(), f'{train_config.model_save_path}/{model_config.model_name}.pth')
            scheduler.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], train_config.scheduler_min_lr)
    print('Finished Training')
    return model


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def metric(preds, trues):
    N = trues.shape[0]
    preds = np.atleast_2d(preds)
    trues = np.atleast_2d(trues)
    return np.sum(np.linalg.norm(preds - trues, axis=1) / np.linalg.norm(trues, axis=1)) / N


def run_test(m, dataset_config: DatasetConfig, base_path: str = None, debug: bool = False, silence: bool = False,
             test_points: List = None, plot: bool = False):
    if test_points is None:
        test_points = dataset_config.test_points
    bar = test_points if silence else tqdm(test_points)
    metric_list = []
    for test_point in bar:
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
        delay = dataset_config.n_point_delay
        diff = metric(P[delay:-delay], Z[2 * delay:])

        if np.isinf(diff):
            continue
        metric_list.append(diff)

        if base_path is not None:
            np.savetxt(f'{img_save_path}/metric.txt', np.atleast_1d(diff))
    if plot or base_path is not None:
        plt.hist(metric_list, bins=20)
        plt.legend()
        plt.title('difference')
        plt.xlabel('difference')
        plt.ylabel('frequency')
        if plot:
            plt.show()
        else:
            plt.savefig(f'{base_path}/metric.png')
    return metric_list


def postprocess(samples):
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
                             n_point_delay=dataset_config.n_point_delay, dynamic=DynamicSystem.dynamic_static)
        new_samples.append((torch.from_numpy(np.concatenate([t, z, u])), torch.tensor(p)))
    print(f'[WARNING] {len(new_samples)} samples replaced by numerical solutions')
    all_samples = new_samples
    return all_samples
    # return samples


def plot_sample(feature, label, dataset_config: DatasetConfig):
    if isinstance(feature, torch.Tensor):
        feature = feature.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    print(f'[Feature Shape]: {feature.shape}')
    print(f'[Label Shape]: {feature.shape}')
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
    if not os.path.exists(dataset_config.training_dataset_file) or dataset_config.recreate_training_dataset:
        print('Creating training dataset')
        if dataset_config.trajectory:
            training_samples = create_trajectory_dataset(dataset_config)
        else:
            training_samples = create_stateless_dataset(dataset_config)
    else:
        print('Loading training dataset')
        with open(dataset_config.training_dataset_file, 'rb') as file:
            training_samples = pickle.load(file)
    if dataset_config.trajectory and dataset_config.postprocess:
        training_samples = postprocess(training_samples)
    for feature, label in training_samples[:dataset_config.n_plot_sample]:
        plot_sample(feature, label, dataset_config)
    training_dataloader, validating_dataloader = prepare_datasets(
        training_samples, train_config.training_ratio, train_config.batch_size, train_config.device)

    print(f'#Training sample: {int(len(training_samples) * train_config.training_ratio)}')
    print(f'#Validating sample: {int(len(training_samples) * (1 - train_config.training_ratio))}')
    return training_dataloader, validating_dataloader


def get_test_datasets(dataset_config, train_config):
    if not os.path.exists(dataset_config.testing_dataset_file) or dataset_config.recreate_testing_dataset:
        print('Creating testing dataset')
        testing_samples = create_trajectory_dataset(dataset_config, test_points=dataset_config.test_points)
    else:
        print('Loading testing dataset')
        with open(dataset_config.testing_dataset_file, 'rb') as file:
            testing_samples = pickle.load(file)
    testing_dataloader = DataLoader(PredictionDataset(testing_samples), batch_size=train_config.batch_size,
                                    shuffle=True, generator=torch.Generator(device=train_config.device))
    print(f'#Testing sample: {len(testing_samples)}')
    return testing_dataloader


def create_trajectory_dataset(dataset_config: DatasetConfig, test_points: List = None, save: bool = True):
    all_samples = []
    if dataset_config.implicit:
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
        if dataset_config.implicit:
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
    if save:
        path = dataset_config.training_dataset_file if test_points is None else dataset_config.testing_dataset_file
        with open(path, 'wb') as file:
            pickle.dump(all_samples, file)
    return all_samples


def create_stateless_dataset(dataset_config: DatasetConfig, save=True, filter=True):
    dt: float = dataset_config.dt
    n_point_delay: int = dataset_config.n_point_delay
    n_sample_per_dataset: int = dataset_config.n_sample_per_dataset
    dataset_file: str = dataset_config.training_dataset_file
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
                    start = segment_length * np.floor(np.min(x - segment_length) / segment_length)
                    end = segment_length * np.ceil(np.max(x + segment_length) / segment_length)

                    key_points = np.arange(start, end + segment_length, segment_length)
                    random_values = np.random.uniform(-1, 1, size=key_points.shape)

                    y = np.zeros_like(x)
                    for i, xi in enumerate(x):
                        index = np.searchsorted(key_points, xi, side='right') - 1
                        x1, x2 = key_points[index], key_points[index + 1]
                        y1, y2 = random_values[index], random_values[index + 1]
                        y[i] = y1 * (1 - (xi - x1) / (x2 - x1)) + y2 * (xi - x1) / (x2 - x1)
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
            # U_D = U_D + DynamicSystem.kappa_static(Z_t) - U_D[0]
            P_t = predict_integral(Z_t=Z_t, U_D=U_D, dt=dt, n_state=n_state, n_point_delay=n_point_delay,
                                   dynamic=DynamicSystem.dynamic_static)
            features = sample_to_tensor(Z_t, U_D, dt * n_point_delay)
            if filter:
                def filter_test(factor, p):
                    return abs(P_t[0]) <= abs(Z_t[0]) * factor and abs(P_t[1]) <= abs(
                        Z_t[1]) * factor and np.random.uniform(0, 1) <= p

                if filter_test(5, 1e-3) or filter_test(2, 1e-2) or filter_test(1, 1e-1) or filter_test(0.5, 1):
                    all_samples.append((features, torch.from_numpy(P_t)))
                    j += 1
            else:
                all_samples.append((features, torch.from_numpy(P_t)))
                j += 1

    random.shuffle(all_samples)
    if save:
        with open(dataset_file, 'wb') as file:
            pickle.dump(all_samples, file)
    return all_samples


def prepare_datasets(samples, training_ratio: float, batch_size: int, device: str):
    def split_dataset(dataset, ratio):
        n_total = len(dataset)
        n_sample = int(n_total * ratio)
        random.shuffle(dataset)
        return dataset[:n_sample], dataset[n_sample:]

    train_dataset, validate_dataset = split_dataset(samples, training_ratio)
    training_dataloader = DataLoader(PredictionDataset(train_dataset), batch_size=batch_size, shuffle=True,
                                     generator=torch.Generator(device=device))
    validating_dataloader = DataLoader(PredictionDataset(validate_dataset), batch_size=batch_size, shuffle=True,
                                       generator=torch.Generator(device=device))
    return training_dataloader, validating_dataloader


def setup_plt():
    def set_size(width, fraction=1, subplots=(1, 1), height_add=0):
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

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "times",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 12,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    }
    plt.rcParams.update(tex_fonts)
    fig_width, fig_height = set_size(width='thesis', fraction=1)


if __name__ == '__main__':
    # setup_plt()
    dataset_config = DatasetConfig(
        recreate_training_dataset=False,
        recreate_testing_dataset=False,
        trajectory=False,
        random_u_type='spline',
        dt=0.1,
        n_dataset=50000,
        duration=8,
        delay=3.,
        n_sample_per_dataset=1,
        ic_lower_bound=-1,
        ic_upper_bound=1,
        postprocess=False
    )
    model_config = ModelConfig(
        model_name='FNO',
        # fno_n_layers=6,
        # deeponet_n_hidden_size=256,
        # deeponet_merge_size=128,
        # deeponet_n_hidden=6,
        fno_n_layers=3,
        fno_n_modes_height=4,
        fno_hidden_channels=16,
        # fno_n_modes_height=32,
        # fno_hidden_channels=64
    )
    train_config = TrainConfig(
        learning_rate=1e-3,
        n_epoch=200,
        batch_size=64,
        scheduler_step_size=1,
        scheduler_gamma=0.99,
        scheduler_min_lr=3e-6,
        weight_decay=1e-3,
        load_model=False
    )
    training_dataloader, validating_dataloader = get_training_and_validation_datasets(dataset_config, train_config)
    testing_dataloader = get_test_datasets(dataset_config, train_config)

    check_dir(model_config.base_path)
    check_dir(train_config.model_save_path)
    model = run_train(dataset_config=dataset_config, model_config=model_config, train_config=train_config,
                      training_dataloader=training_dataloader, validating_dataloader=validating_dataloader,
                      testing_dataloader=testing_dataloader, img_save_path=model_config.base_path)
    run_test(m=model, dataset_config=dataset_config, base_path=model_config.base_path, debug=train_config.debug)
