import os
import pickle
import random
from typing import Literal, Tuple

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DatasetConfig, ModelConfig, TrainConfig
from dataset import ImplicitDataset, ExplictDataset, sample_to_tensor, PredictionDataset
from model import PredictionFNO
from system1 import control_law_explict, solve_z_explict, control_law, system, predict_integral_general, \
    predict_integral
from utils import count_params, pad_leading_zeros


def predict_neural_operator(model, U_D, Z_t, t):
    u_tensor = torch.tensor(U_D, dtype=torch.float32).view(1, -1)
    z_tensor = torch.tensor(Z_t, dtype=torch.float32).view(1, -1)
    inputs = [torch.cat([z_tensor, u_tensor], dim=1), torch.tensor(t, dtype=torch.float32).view(1, -1)]
    if isinstance(model, PredictionFNO):
        outputs = model(inputs[0])
    else:
        outputs = model(inputs)
    return outputs.to('cpu').detach().numpy()[0]


def run(dataset_config: DatasetConfig,
        Z0: Tuple,
        silence: bool = False,
        plot: bool = False,
        model=None,
        method: Literal['explict', 'numerical', 'no'] = None,
        title='',
        save_path: str = None,
        img_save_path: str = None,
        cut: bool = False):
    if not silence:
        print(f'Solving with method "{method}"')
    delay: float = dataset_config.delay
    duration: float = dataset_config.duration
    n_point: int = dataset_config.n_point
    n_point_duration: int = dataset_config.n_point_duration
    n_point_delay: int = dataset_config.n_point_delay
    ts: np.ndarray = dataset_config.ts
    dt: float = dataset_config.dt
    U = np.zeros(n_point)
    Z = np.zeros((n_point, 2))
    P = np.zeros((n_point, 2))
    Z0 = np.array(Z0)
    Z[n_point_delay, :] = Z0
    sequence = range(n_point) if silence else tqdm(list(range(n_point)))
    for t_i in sequence:
        t_minus_D_i = max(t_i - n_point_delay, 0)
        t = ts[t_i]
        if method == 'explict':
            U[t_i] = control_law_explict(t, Z0, delay)
            if t_i > n_point_delay:
                Z[t_i, :] = solve_z_explict(t, delay, Z0)
        elif method == 'numerical':
            if t_i > n_point_delay:
                Z[t_i, :] = odeint(system, Z[t_i - 1, :], [ts[t_i - 1], ts[t_i]], args=(U[t_minus_D_i - 1],))[1]
                Z_t = Z[t_i, :]
            else:
                Z_t = Z0
            P[t_i, :] = predict_integral_general(f=system, Z_t=Z_t, P_D=P[t_minus_D_i:t_i], U_D=U[t_minus_D_i:t_i],
                                                 dt=dt, t=t)
            if t_i > n_point_delay:
                U[t_i] = control_law(P[t_i, :])
        elif method == 'no':
            if t_i > n_point_delay:
                Z[t_i, :] = odeint(system, Z[t_i - 1, :], [ts[t_i - 1], ts[t_i]], args=(U[t_minus_D_i - 1],))[1]
                Z_t = Z[t_i, :]
            else:
                Z_t = Z0
            P[t_i, :] = predict_neural_operator(
                model=model, U_D=pad_leading_zeros(segment=U[t_minus_D_i:t_i], length=n_point_delay), Z_t=Z_t, t=t)
            if t_i > n_point_delay:
                U[t_i] = control_law(P[t_i, :])
        else:
            raise NotImplementedError()
    if cut:
        P = P[n_point_delay:, :]
        U = U[n_point_delay:]
        Z = Z[n_point_delay:, :]
        ts = ts[n_point_delay:]
    if not silence:
        print(f'Finish solving')
    if save_path is not None:
        result = {
            "u": U,
            "z": Z,
            "d": delay,
            "duration": duration,
            "n_point_duration": n_point_duration,
            "ts": ts
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

        plt.title('Comparison')
        for t_i in range(2):
            plt.plot(ts[n_point_delay:], P[:-n_point_delay, t_i], label=f'$P_{t_i + 1}(t-{delay})$')
            plt.plot(ts[n_point_delay:], Z[n_point_delay:, t_i], label=f'$Z_{t_i + 1}(t)$')
        plt.legend()
        if img_save_path is not None:
            plt.savefig(f'{img_save_path}/comparison_full.png')
            plt.clf()
        else:
            plt.show()
        plt.title('Comparison')
        plt.ylim([-5, 5])
        for t_i in range(2):
            plt.plot(ts[n_point_delay:], P[:-n_point_delay, t_i], label=f'$P_{t_i + 1}(t-{delay})$')
            plt.plot(ts[n_point_delay:], Z[n_point_delay:, t_i], label=f'$Z_{t_i + 1}(t)$')
        plt.legend()
        if img_save_path is not None:
            plt.savefig(f'{img_save_path}/comparison_zoom.png')
            plt.clf()
        else:
            plt.show()
        plt.title('Difference')
        difference = P[:-n_point_delay] - Z[n_point_delay:]
        plt.ylim([-1, 1])
        plt.plot(ts[n_point_delay:], difference[:, 0], label='difference of prediction1')
        plt.plot(ts[n_point_delay:], difference[:, 1], label='difference of prediction2')
        plt.legend()
        if img_save_path is not None:
            plt.savefig(f'{img_save_path}/difference.png')
            plt.clf()
        else:
            plt.show()
    if method == 'explict':
        return U, Z, None
    return U, Z, P


def no_predict(inputs, device, model):
    inputs = inputs.to(device)
    time_step = inputs[:, :1]
    z_u = inputs[:, 1:]

    inputs = [z_u, time_step]
    if isinstance(model, PredictionFNO):
        inputs = z_u
    return model(inputs)


def run_train(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig,
              training_dataloader=None, validating_dataloader=None, img_save_path: str = None):
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
        model.load_state_dict(torch.load('model_state_dict.pth'))
        print(f'Model loaded from {pth}, skip training!')
    else:
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_config.scheduler_step_size,
                                                    gamma=train_config.scheduler_gamma)

        training_loss_arr = []
        validating_loss_arr = []
        bar = tqdm(list(range(n_epoch)))
        for epoch in bar:
            model.train()
            training_loss = 0.0
            for inputs, label in training_dataloader:
                label = label.to(device)
                outputs = no_predict(inputs, device, model)
                loss = criterion(outputs, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss += loss.item()

            model.eval()
            with torch.no_grad():
                validating_loss = 0.0
                for inputs, label in validating_dataloader:
                    outputs = no_predict(inputs, device, model)
                    label = label.to(device)
                    loss = criterion(outputs, label)
                    validating_loss += loss.item()

                training_loss_t = training_loss / len(training_dataloader)
                validating_loss_t = validating_loss / len(validating_dataloader)
                training_loss_arr.append(training_loss_t)
                validating_loss_arr.append(validating_loss_t)
                desc = f'Epoch [{epoch + 1}/{n_epoch}] || Learning rate: {scheduler.get_last_lr()} || ' \
                       f'Training loss: {training_loss_t} || Validation loss: {validating_loss_t}'
                bar.set_description(desc)
            if epoch % 10 == 0:
                plt.figure()
                plt.plot(training_loss_arr, label="Training loss")
                plt.plot(validating_loss_arr, label="Validation loss")
                plt.yscale("log")
                plt.legend()
                if img_save_path is not None:
                    plt.savefig(f'{img_save_path}/loss.png')
                    plt.clf()
                else:
                    plt.show()
            #
            scheduler.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], train_config.scheduler_min_lr)

        plt.figure()
        plt.plot(training_loss_arr, label="Training loss")
        plt.plot(validating_loss_arr, label="Validation loss")
        plt.yscale("log")
        plt.legend()
        if img_save_path is not None:
            plt.savefig(f'{img_save_path}/loss.png')
            plt.clf()
        else:
            plt.show()
        torch.save(model.state_dict(), f'{train_config.model_save_path}/{model_config.model_name}.pth')
    print('Finished Training')
    return model


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def metric(u_preds, u_trues):
    N = u_trues.shape[0]
    u_preds = np.atleast_2d(u_preds)
    u_trues = np.atleast_2d(u_trues)
    epsilon = np.sum(np.linalg.norm(u_preds - u_trues, axis=1) / np.linalg.norm(u_trues, axis=1)) / N
    return epsilon


def run_test(m, dataset_config: DatasetConfig, base_path: str):
    bar = tqdm(dataset_config.test_points)
    for test_point in bar:
        bar.set_description(f'Solving system with initial point {test_point}.')
        img_save_path = f'{base_path}/{test_point}'
        check_dir(img_save_path)
        run(dataset_config=dataset_config, silence=True, model=m, Z0=test_point, method='no', plot=True, title='no',
            img_save_path=img_save_path)


def get_dataset(dataset_config: DatasetConfig, train_config: TrainConfig):
    if not os.path.exists(dataset_config.dataset_file) or dataset_config.recreate_dataset:
        if dataset_config.trajectory:
            samples = create_trajectory_dataset(dataset_config)
        else:
            samples = create_stateless_dataset(
                dataset_config.n_dataset, dataset_config.dt, dataset_config.n_point_delay,
                dataset_config.n_sample_per_dataset, dataset_config.dataset_file, dataset_config.n_state
            )
    else:
        with open(dataset_config.dataset_file, 'rb') as file:
            samples = pickle.load(file)
    training_dataloader, validating_dataloader = prepare_dataset(
        samples, train_config.training_ratio, train_config.batch_size, train_config.device)
    return training_dataloader, validating_dataloader


def create_trajectory_dataset(dataset_config: DatasetConfig):
    all_samples = []
    if dataset_config.implicit:
        print('creating implicit datasets (Use Z(t+D) as P(t))')
    else:
        print('creating explicit datasets (Calculate P(t))')
    for z in tqdm(list(np.random.uniform(0, 1, (dataset_config.n_dataset, 2)))):
        if dataset_config.implicit:
            U, Z, _ = run(method='explict', silence=True, Z0=z, dataset_config=dataset_config)
            dataset = ImplicitDataset(
                torch.tensor(Z, dtype=torch.float32), torch.tensor(U, dtype=torch.float32),
                dataset_config.n_point_delay, dataset_config.dt)
        else:
            U, Z, P = run(method='numerical', silence=True, Z0=z, dataset_config=dataset_config)
            dataset = ExplictDataset(
                torch.tensor(Z, dtype=torch.float32), torch.tensor(U, dtype=torch.float32),
                torch.tensor(P, dtype=torch.float32), dataset_config.n_point_delay, dataset_config.dt)
        dataset = list(dataset)
        random.shuffle(dataset)
        all_samples += dataset[:dataset_config.n_sample_per_dataset]
    new_samples = []
    for feature, label in all_samples:
        z = feature[1:3].cpu().numpy()
        u = feature[3:].cpu().numpy()
        p = predict_integral(Z_t=z, U_D=u, dt=dataset_config.dt, n_state=2,
                             n_point_delay=dataset_config.n_point_delay)
        new_samples.append((feature, torch.tensor(p)))
    # all_samples = new_samples

    #     label = label.cpu().numpy()
    #     plt.plot(u, label='u')
    #     plt.scatter(dataset_config.n_point_delay, z[0], label='z0')
    #     plt.scatter(dataset_config.n_point_delay, z[1], label='z1')
    #     plt.scatter(dataset_config.n_point_delay, u[0], label='p0')
    #     plt.scatter(dataset_config.n_point_delay, u[1], label='p1')
    #     plt.legend()
    #     plt.ylim([-2, 2])
    #     plt.show()
    #     plt.clf()
    random.shuffle(all_samples)
    with open(dataset_config.dataset_file, 'wb') as file:
        pickle.dump(all_samples, file)
    return all_samples


def create_stateless_dataset(n_dataset: int, dt: float, n_point_delay: int, n_sample_per_dataset: int,
                             dataset_file: str, n_state: int):
    all_samples = []
    # FIXME
    for i in tqdm(list(range(3, n_dataset))):
        for _ in range(n_sample_per_dataset):
            f_rand = np.random.randint(0, 3)
            shift = np.random.uniform(-1, 1)
            # shift = 0
            if f_rand == 0:
                f = np.cos
            elif f_rand == 1:
                f = np.sin
            else:
                f = lambda x: np.exp(-x)
            U_D = f(np.sqrt(i) * np.linspace(0, 1, n_point_delay)) + shift
            Z_t = np.random.uniform(0, 1, 2)
            P_t = predict_integral(Z_t=Z_t, U_D=U_D, dt=dt, n_state=n_state, n_point_delay=n_point_delay)
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
    random.shuffle(all_samples)
    with open(dataset_file, 'wb') as file:
        pickle.dump(all_samples, file)
    return all_samples


def prepare_dataset(samples, training_ratio: float, batch_size: int, device: str):
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
    dataset_config = DatasetConfig(recreate_dataset=False, trajectory=True, dt=0.01, n_dataset=1000, duration=8,
                                   delay=3, n_sample_per_dataset=250)
    model_config = ModelConfig(model_name='DeepONet', deeponet_n_hidden=3, fno_n_layers=6)
    train_config = TrainConfig(learning_rate=1e-3, n_epoch=500, batch_size=512, scheduler_step_size=1,
                               scheduler_gamma=0.98, scheduler_min_lr=1e-6, weight_decay=1e-4)

    training_dataloader, validating_dataloader = get_dataset(dataset_config, train_config)
    check_dir(model_config.base_path)
    check_dir(train_config.model_save_path)
    model = run_train(dataset_config=dataset_config, model_config=model_config, train_config=train_config,
                      training_dataloader=training_dataloader, validating_dataloader=validating_dataloader,
                      img_save_path=model_config.base_path)
    run_test(m=model, dataset_config=dataset_config, base_path=model_config.base_path)
