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
from dataset import ImplicitDataset, ExplictDataset, sample_to_tensor, PredictionDataset
from dynamic_systems import DynamicSystem, predict_integral
from model import PredictionFNO
from utils import count_params


def plot_comparison(ts, P, P_compare, Z, delay, n_point_delay, save_path):
    plt.title('Comparison')
    for t_i in range(2):
        if P_compare is not None:
            plt.plot(ts[n_point_delay:], P_compare[:-n_point_delay, t_i], label=f'$PNO_{t_i + 1}(t-{delay})$')
        plt.plot(ts[n_point_delay:], P[:-n_point_delay, t_i], label=f'$P_{t_i + 1}(t-{delay})$')
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
    plt.plot(ts[n_point_delay:], difference[:, 0], label='difference of prediction1')
    plt.plot(ts[n_point_delay:], difference[:, 1], label='difference of prediction2')
    if P_compare is not None:
        difference_no = P_compare[:-n_point_delay] - Z[n_point_delay:]
        plt.plot(ts[n_point_delay:], difference_no[:, 0], label='difference of no prediction1')
        plt.plot(ts[n_point_delay:], difference_no[:, 1], label='difference of no prediction2')

    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def run(dataset_config: DatasetConfig,
        Z0: Tuple,
        plot: bool = False,
        model=None,
        method: Literal['explict', 'numerical', 'no', 'numerical_no'] = None,
        title='',
        save_path: str = None,
        img_save_path: str = None):
    system = DynamicSystem(Z0=np.array(list(Z0)), dataset_config=dataset_config, method=method)
    for _ in range(dataset_config.n_point):
        system.step_in(model)

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
    if plot:
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
    if isinstance(model, PredictionFNO):
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
        model = PredictionFNO(
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

                testing_loss = 0.0
                for inputs, label in testing_dataloader:
                    outputs = no_predict(inputs, model)
                    loss = criterion(outputs, label)
                    testing_loss += loss.item()

            training_loss_t = training_loss / len(training_dataloader)
            validating_loss_t = validating_loss / len(validating_dataloader)
            testing_loss_t = testing_loss / len(testing_dataloader)
            training_loss_arr.append(training_loss_t)
            validating_loss_arr.append(validating_loss_t)
            testing_loss_arr.append(testing_loss_t)
            desc = f'Epoch [{epoch + 1}/{n_epoch}] || Learning rate: {scheduler.get_last_lr()} || ' \
                   f'Training loss: {training_loss_t} || Validation loss: {validating_loss_t} || ' \
                   f'Test loss: {testing_loss_t}'
            bar.set_description(desc)
            if epoch % train_config.log_step == 0 or epoch == n_epoch - 1:
                plt.plot(training_loss_arr, label="Training loss")
                plt.plot(validating_loss_arr, label="Validation loss")
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


def metric(u_preds, u_trues):
    N = u_trues.shape[0]
    u_preds = np.atleast_2d(u_preds)
    u_trues = np.atleast_2d(u_trues)
    return np.sum(np.linalg.norm(u_preds - u_trues, axis=1) / np.linalg.norm(u_trues, axis=1)) / N


def run_test(m, dataset_config: DatasetConfig, base_path: str, debug: bool):
    bar = tqdm(dataset_config.test_points)
    for test_point in bar:
        bar.set_description(f'Solving system with initial point {test_point}.')
        img_save_path = f'{base_path}/{test_point}'
        check_dir(img_save_path)
        if debug:
            run(dataset_config=dataset_config, model=m, Z0=test_point, method='numerical_no', plot=True,
                title='no', img_save_path=img_save_path)
        else:
            run(dataset_config=dataset_config, model=m, Z0=test_point, method='no', plot=True, title='no',
                img_save_path=img_save_path)


def postprocess(samples):
    print('[DEBUG] postprocessing')
    new_samples = []
    random.shuffle(samples)
    epsilon_sum = 0
    for sample in samples:
        #     z = feature[1:3]
        #     u = feature[3:]
        # print(f'average p: {p_sum / len(samples)}')
        # print(f'average z: {z_sum / len(samples)}')
        # print(f'average u: {u_sum / len(samples)}')

        feature = sample[0].cpu().numpy()
        t = feature[:1]
        z = feature[1:3]
        u = feature[3:]
        p = sample[1].cpu().numpy()
        epsilon_z = 0.1
        # epsilon_z = np.random.uniform(0, 1., len(z))
        z += epsilon_z
        # epsilon_u = np.random.uniform(0, .01, len(u))
        # u += epsilon_u
        # epsilon_z = 0.1
        # u0 = u[0]
        # z1 = z[0]
        # z2 = z[1]
        # u_compare = -z1 - 2 * z2 - 1 / 3 * z2 ** 3
        # loss += (u0 - u_compare) ** 2
        # epsilon_sum += epsilon_z
        p = predict_integral(Z_t=z, U_D=u, dt=dataset_config.dt, n_state=2,
                             n_point_delay=dataset_config.n_point_delay)
        new_samples.append((torch.from_numpy(np.concatenate([t, z, u])), torch.tensor(p)))
    # print(epsilon_sum / len(samples))
    # print(loss)
    #     t_z_u = sample[0].cpu().numpy()
    #     t = t_z_u[0]
    #     z = t_z_u[1:3]
    #     u = t_z_u[3:]
    #     p = sample[1].cpu().numpy()
    #     ts = np.linspace(t - dataset_config.delay, t, dataset_config.n_point_delay)
    #     plt.plot(ts, u, label='u')
    #     plt.scatter(ts[-1], z[0], label='Zt_0')
    #     plt.scatter(ts[-1], z[1], label='Zt_1')
    #     plt.scatter(ts[-1], p[0], label='Pt_0')
    #     plt.scatter(ts[-1], p[1], label='Pt_1')
    #     plt.legend()
    #     plt.show()
    #     plt.clf()
    # time.sleep(0.5)
    print(f'[WARNING] {len(new_samples)} samples replaced by numerical solutions')
    all_samples = new_samples
    return all_samples
    # return samples


def get_dataset(dataset_config: DatasetConfig, train_config: TrainConfig, n_dataset: int, file_path: str,
                recreate: bool):
    if not os.path.exists(file_path) or recreate:
        print('Creating dataset')
        if dataset_config.trajectory:
            samples = create_trajectory_dataset(dataset_config, n_dataset=n_dataset)
        else:
            samples = create_stateless_dataset(dataset_config, n_dataset=n_dataset)
        print('Data of dataset')
    else:
        print('Loading dataset')
        with open(file_path, 'rb') as file:
            samples = pickle.load(file)
    # samples = postprocess(samples)
    return DataLoader(PredictionDataset(samples), batch_size=train_config.batch_size, shuffle=True,
                      generator=torch.Generator(device=train_config.device))


def get_training_and_validation_datasets(dataset_config: DatasetConfig, train_config: TrainConfig):
    if not os.path.exists(dataset_config.training_dataset_file) or dataset_config.recreate_training_dataset:
        print('Creating training dataset')
        if dataset_config.trajectory:
            training_samples = create_trajectory_dataset(dataset_config, n_dataset=dataset_config.n_dataset)
        else:
            training_samples = create_stateless_dataset(dataset_config, n_dataset=dataset_config.n_dataset)
    else:
        print('Loading training dataset')
        with open(dataset_config.training_dataset_file, 'rb') as file:
            training_samples = pickle.load(file)
    # training_samples = postprocess(training_samples)

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


def create_trajectory_dataset(dataset_config: DatasetConfig, n_dataset: int = None, test_points: List = None,
                              save: bool = True):
    all_samples = []
    if dataset_config.implicit:
        print('creating implicit datasets (Use Z(t+D) as P(t))')
    else:
        print('creating explicit datasets (Calculate P(t))')
    if test_points is None:
        bar = tqdm(list(
            np.random.uniform(dataset_config.ic_lower_bound, dataset_config.ic_upper_bound,
                              (n_dataset, 2))))
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


def create_stateless_dataset(dataset_config: DatasetConfig, n_dataset: int):
    dt: float = dataset_config.dt
    n_point_delay: int = dataset_config.n_point_delay
    n_sample_per_dataset: int = dataset_config.n_sample_per_dataset
    dataset_file: str = dataset_config.training_dataset_file
    n_state: int = dataset_config.n_state
    all_samples = []
    # FIXME
    for _ in tqdm(list(range(10, n_dataset + 10))):
        for _ in range(n_sample_per_dataset):
            # f_rand = np.random.randint(0, 3)
            # if f_rand == 0:
            #     f = np.cos
            # elif f_rand == 1:
            #     f = np.sin
            # else:
            # def f(x):
            #     x = x.reshape(1, -1)
            #     grid = np.arange(1, 100).reshape(-1, 1)
            #     series = (np.sin(grid * x) + np.cos(grid * x)) / (grid ** 2)
            #     return series.sum(axis=0)
            f = lambda x: (np.random.uniform(-3, 3, 10).reshape(-1, 1) * np.array(
                [x ** 9, x ** 8, x ** 7, x ** 6, x ** 5, x ** 4, x ** 3, x ** 2, x, np.ones_like(x)])).sum(
                axis=0)
            # f = lambda x: sum([np.cos(i) + np.sin(i) for i in range(100)])
            # U_D = f(np.sqrt(i) * np.linspace(0, dataset_config.delay, n_point_delay)) * dataset_config.u_scaling
            Z_t = np.random.uniform(dataset_config.ic_lower_bound, dataset_config.ic_upper_bound, 2)
            U_D = f(np.linspace(0, dataset_config.delay, n_point_delay))

            U_D = U_D + DynamicSystem.kappa_static(Z_t) - U_D[0]

            P_t = predict_integral(Z_t=Z_t, U_D=U_D, dt=dt, n_state=n_state, n_point_delay=n_point_delay,
                                   dynamic=DynamicSystem.dynamic_static)
            features = sample_to_tensor(Z_t, U_D, dt * n_point_delay)
            all_samples.append((features, torch.from_numpy(P_t)))
            # plt.plot(U_D, label='u')
            # plt.scatter(n_point_delay, Z_t[0], label='z0')
            # plt.scatter(n_point_delay, Z_t[1], label='z1')
            # plt.scatter(n_point_delay, P_t[0], label='p0')
            # plt.scatter(n_point_delay, P_t[1], label='p1')
            # plt.legend()
            # plt.ylim([-2, 2])
            # plt.show()
            # plt.clf()
            # print('_')
            # print(Z_t)
            # print(P_t)
    random.shuffle(all_samples)
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


if __name__ == '__main__':
    dataset_config = DatasetConfig(
        recreate_training_dataset=True,
        recreate_testing_dataset=True,
        trajectory=False,
        dt=0.1,
        n_dataset=100,
        duration=8,
        delay=1,
        n_sample_per_dataset=100,
        ic_lower_bound=-1,
        ic_upper_bound=1,
        u_scaling=1.,
        system_c=1.
    )
    model_config = ModelConfig(
        model_name='FNO',
        # deeponet_n_hidden_size=256,
        # deeponet_merge_size=128,
        # deeponet_n_hidden=6,
        # fno_n_layers=20,
        # fno_n_modes_height=16,
        # fno_hidden_channels=64
    )
    train_config = TrainConfig(
        learning_rate=1e-3,
        n_epoch=200,
        batch_size=64,
        scheduler_step_size=1,
        scheduler_gamma=0.96,
        scheduler_min_lr=3e-6,
        weight_decay=1e-3,
        load_model=False,
        debug=False
    )
    # dataset_config.system_c = 1
    training_dataloader, validating_dataloader = get_training_and_validation_datasets(dataset_config, train_config)
    # dataset_config.system_c = 3
    testing_dataloader = get_test_datasets(dataset_config, train_config)

    check_dir(model_config.base_path)
    check_dir(train_config.model_save_path)
    model = run_train(dataset_config=dataset_config, model_config=model_config, train_config=train_config,
                      training_dataloader=training_dataloader, validating_dataloader=validating_dataloader,
                      testing_dataloader=testing_dataloader, img_save_path=model_config.base_path)
    run_test(m=model, dataset_config=dataset_config, base_path=model_config.base_path, debug=train_config.debug)
