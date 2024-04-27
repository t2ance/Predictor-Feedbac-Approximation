import os
import pickle
import random
from typing import Literal, Tuple, List

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
from system1 import control_law_explict, solve_z_explict, control_law, system, integral_prediction_general
from utils import count_params, padding_leading_zero


def predict_neural_operator(model, U_D, Z_t, t):
    u_tensor = torch.tensor(U_D, dtype=torch.float32).view(1, -1)
    z_tensor = torch.tensor(Z_t, dtype=torch.float32).view(1, -1)
    inputs = [torch.cat([z_tensor, u_tensor], dim=1), torch.tensor(t, dtype=torch.float32).view(1, -1)]
    if isinstance(model, PredictionFNO):
        outputs = model(inputs[0])
    else:
        outputs = model(inputs)
    [P1, P2] = outputs.to('cpu').detach().numpy()[0]
    return P1, P2


def run(delay: float, Z0: Tuple, duration: float, dt: float, silence: bool = False, plot: bool = False,
        model=None, method: Literal['explict', 'numerical', 'no'] = None, title='', save_path: str = None,
        img_save_path: str = None, cut: bool = False):
    if not silence:
        print(f'Solving with method "{method}"')
    n_point = int(duration / dt)
    n_point_delay = int(delay / dt)
    n_point_total = n_point + n_point_delay
    ts = np.linspace(-delay, duration, n_point_total)

    U = np.zeros(n_point_total)
    Z = np.zeros((n_point_total, 2))
    P = np.zeros((n_point_total, 2))
    Z0 = np.array(Z0)
    Z[n_point_delay, :] = Z0

    if silence:
        # sequence = range(n_point_delay, n_point_total)
        sequence = range(n_point_total)
    else:
        # sequence = tqdm(list(range(n_point_delay, n_point_total)))
        sequence = tqdm(list(range(n_point_total)))

    for t_i in sequence:
        t_minus_D_i = t_i - n_point_delay
        t = ts[t_i]
        if method == 'explict':
            U[t_i] = control_law_explict(t, Z0, delay)
            if t_i > n_point_delay:
                Z[t_i, :] = solve_z_explict(t, delay, Z0)
        elif method == 'no':
            if t_i > n_point_delay:
                Z[t_i, :] = solve_z_explict(t, delay, Z0)
                Z_t = Z[t_i, :]
            else:
                Z_t = Z0
            U_input = padding_leading_zero(U, t_minus_D_i, t_i)
            P[t_i, :] = predict_neural_operator(model=model, U_D=U_input, Z_t=Z_t, t=t)
            U[t_i] = control_law(P[t_i, :])
        elif method == 'numerical':
            if t_i > n_point_delay:
                Z[t_i, :] = odeint(system, Z[t_i - 1, :], [ts[t_i - 1], ts[t_i]], args=(U[t_minus_D_i - 1],))[1]
                Z_t = Z[t_i, :]
            else:
                Z_t = Z0
            # P1_t, P2_t = integral_prediction_explict(t=ts[t_i], delay=delay, Z1_t=Z1_t, Z2_t=Z2_t,
            #                                          U_D=U[t_minus_D_i:t_i], ts_D=ts[t_minus_D_i:t_i], dt=dt)
            P1_t, P2_t = integral_prediction_general(
                f=system, Z_t=Z_t, P_D=P[t_minus_D_i:t_i], U_D=U[t_minus_D_i:t_i], dt=dt, t=t)
            P[t_i, :] = [P1_t, P2_t]
            if t_i > n_point_delay:
                U_t = control_law(P[t_i, :])
                U[t_i] = U_t
            # if t_i + 1 < n_point_total:
            # Z[t_i + 1, :] = ode_forward(Z_t, system(Z[t_i, :], t, U[t_minus_D_i]), dt)
            # Z[t_i + 1, :] = odeint(system, Z_t, [ts[t_i], ts[t_i + 1]], args=(U[t_minus_D_i],))[1]
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
            "n_point": n_point,
            "ts": ts
        }
        with open(save_path, 'wb') as file:
            pickle.dump(result, file)
    if plot:
        plt.figure(figsize=(10, 8))
        plt.title(title)

        plt.subplot(511)
        plt.plot(ts, Z[:, 0], label='$Z_1(t)$')
        plt.ylabel('$Z_1(t)$')
        plt.grid(True)

        plt.subplot(512)
        plt.plot(ts, Z[:, 1], label='$Z_2(t)$')
        plt.ylabel('$Z_2(t)$')
        plt.grid(True)

        plt.subplot(513)
        plt.plot(ts, U, label='$U(t)$', color='black')
        plt.xlabel('time')
        plt.ylabel('$U(t)$')
        plt.grid(True)

        plt.subplot(514)
        plt.plot(ts, P[:, 0], label='$P_1(t)$')
        plt.ylabel('$P_1(t)$')
        plt.grid(True)

        plt.subplot(515)
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


def no_predict(model_name, inputs, device, model):
    inputs = inputs.to(device)
    time_step = inputs[:, :1]
    z_u = inputs[:, 1:]

    inputs = [z_u, time_step]
    if model_name == 'FNO':
        inputs = z_u
    return model(inputs)


def run_train(n_state: int, hidden_size: int, n_hidden: int, merge_size: int, lr: float, n_epoch: int,
              weight_decay: float, training_dataloader=None, validating_dataloader=None, n_delay_point=None,
              n_modes_height=8, hidden_channels=16, model_name: Literal['DeepONet', 'FNO'] = 'DeepONet', device='cuda',
              img_save_path: str = None):
    if model_name == 'DeepONet':
        layer_size_branch = [n_delay_point + n_state] + [hidden_size] * n_hidden + [merge_size]
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
            n_modes_height=n_modes_height, hidden_channels=hidden_channels, in_features=n_delay_point + n_state,
            out_features=n_state).to(device)
    else:
        raise NotImplementedError()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f'#parameters: {count_params(model)}')
    training_loss_arr = []
    validating_loss_arr = []
    for epoch in list(range(n_epoch)):
        model.train()
        training_loss = 0.0
        for inputs, label in training_dataloader:
            label = label.to(device)
            outputs = no_predict(model_name, inputs, device, model)
            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        model.eval()
        with torch.no_grad():
            validating_loss = 0.0
            for inputs, label in validating_dataloader:
                outputs = no_predict(model_name, inputs, device, model)
                label = label.to(device)
                loss = criterion(outputs, label)
                validating_loss += loss.item()

            training_loss_t = training_loss / len(training_dataloader)
            validating_loss_t = validating_loss / len(validating_dataloader)
            training_loss_arr.append(training_loss_t)
            validating_loss_arr.append(validating_loss_t)
            print(
                f'Epoch [{epoch + 1}/{n_epoch}] || Training loss: {training_loss_t} || Validating loss: {validating_loss_t}')

    plt.figure()
    plt.plot(training_loss_arr, label="Train Loss")
    plt.plot(validating_loss_arr, label="Validate Loss")
    plt.yscale("log")
    plt.legend()
    if img_save_path is not None:
        plt.savefig(f'{img_save_path}/loss.png')
        plt.clf()
    else:
        plt.show()

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


def run_test(test_points: List[Tuple[float, float]], base_path: str, delay: float, duration: float, ts: np.ndarray,
             n_point_delay: int, dt: float):
    for test_point in test_points:
        img_save_path = f'{base_path}/{test_point}'
        check_dir(img_save_path)
        U_no, Z_no, P_no = run(model=m, delay=delay, Z0=test_point, method='no', plot=True, title='no',
                               duration=duration, img_save_path=img_save_path, dt=dt)
        U_explict, Z_explict, P_explict = run(model=m, delay=delay, Z0=test_point, method='explict', plot=False,
                                              title='explict', duration=duration,
                                              img_save_path=img_save_path, dt=dt)
        t_comparison = ts[n_point_delay:]
        plt.title('Comparison')
        for t_i in range(2):
            plt.plot(ts[n_point_delay:], P_no[:-n_point_delay, t_i], label=f'$P_{t_i + 1}(t-{delay})$')
            plt.plot(ts[n_point_delay:], Z_explict[n_point_delay:, t_i], label=f'$Z_{t_i + 1}(t)$')
        plt.legend()
        plt.savefig(f'{img_save_path}/comparison.png')
        plt.clf()
        plt.title('Difference')
        difference = P_no[:-n_point_delay] - Z_explict[n_point_delay:]
        plt.ylim([-1, 1])
        plt.plot(t_comparison, difference[:, 0], label='difference of prediction1')
        plt.plot(t_comparison, difference[:, 1], label='difference of prediction2')
        plt.legend()
        plt.savefig(f'{img_save_path}/difference.png')
        plt.clf()


def get_dataset(dataset_config: DatasetConfig, train_config: TrainConfig):
    if not os.path.exists(dataset_config.dataset_file) or dataset_config.recreate_dataset:
        if dataset_config.trajectory:
            samples_ = create_trajectory_dataset(
                dataset_config.n_dataset, dataset_config.duration, dataset_config.delay, dataset_config.n_delay_point,
                dataset_config.n_sample_per_dataset, dataset_config.implicit, dataset_config.dataset_file,
                dataset_config.dt
            )
        else:
            samples_ = create_stateless_dataset(
                dataset_config.n_dataset, dataset_config.dt, dataset_config.n_delay_point,
                dataset_config.n_sample_per_dataset, dataset_config.dataset_file, dataset_config.n_state
            )
    else:
        with open(dataset_config.dataset_file, 'rb') as file:
            samples_ = pickle.load(file)
    training_dataloader, validating_dataloader = prepare_dataset(
        samples_, train_config.training_ratio, train_config.batch_size, train_config.device)
    return training_dataloader, validating_dataloader


def create_trajectory_dataset(n_dataset: int, duration: float, delay: float, n_delay_point: int,
                              n_sample_per_dataset: int, implicit: bool, dataset_file: str, dt: float):
    all_samples = []
    if implicit:
        print('creating implicit datasets (Use Z(t+D) as P(t))')
    else:
        print('creating explicit datasets (Calculate P(t))')
    for z in tqdm(list(np.random.uniform(0, 1, (n_dataset, 2)))):
        if implicit:
            U, Z, _ = run(method='explict', silence=True, duration=duration, delay=delay, Z0=z, dt=dt)
            dataset = ImplicitDataset(
                torch.tensor(Z, dtype=torch.float32), torch.tensor(U, dtype=torch.float32), n_delay_point, dt)
        else:
            U, Z, P = run(method='numerical', silence=True, duration=duration, delay=delay, Z0=z, dt=dt)
            dataset = ExplictDataset(
                torch.tensor(Z, dtype=torch.float32), torch.tensor(U, dtype=torch.float32),
                torch.tensor(P, dtype=torch.float32), n_delay_point, dt)
        dataset = list(dataset)
        random.shuffle(dataset)
        all_samples += dataset[:n_sample_per_dataset]
    random.shuffle(all_samples)
    with open(dataset_file, 'wb') as file:
        pickle.dump(all_samples, file)
    return all_samples


def split_list(list, ratio):
    n_total = len(list)
    n_sample = int(n_total * ratio)
    random.shuffle(list)
    return list[:n_sample], list[n_sample:]


def create_stateless_dataset(n_dataset: int, dt: float, n_delay_point: int, n_sample_per_dataset: int,
                             dataset_file: str, n_state: int):
    all_samples = []
    for i in range(n_dataset * n_sample_per_dataset):
        U_D = np.sin(np.sqrt(i) * np.linspace(0, 1, n_delay_point))
        P_D = np.zeros((n_delay_point, n_state))
        Z_t = np.random.uniform(0, 1, 2)
        P_D[0, :] = Z_t
        for j in range(n_delay_point - 1):
            P_D[j + 1] = P_D[j] + dt * system(P_D[j], j * dt, U_D[j])
        label = integral_prediction_general(f=system, Z_t=Z_t, P_D=P_D, U_D=U_D, dt=dt, t=dt * n_delay_point)
        features = sample_to_tensor(Z_t, U_D, dt * n_delay_point)
        all_samples.append((features, torch.from_numpy(label)))
    with open(dataset_file, 'wb') as file:
        pickle.dump(all_samples, file)
    return all_samples


def prepare_dataset(samples, training_ratio: float, batch_size: int, device: str):
    train_dataset, validate_dataset = split_list(samples, training_ratio)
    training_dataloader = DataLoader(PredictionDataset(train_dataset), batch_size=batch_size, shuffle=True,
                                     generator=torch.Generator(device=device))
    validating_dataloader = DataLoader(PredictionDataset(validate_dataset), batch_size=batch_size, shuffle=True,
                                       generator=torch.Generator(device=device))
    return training_dataloader, validating_dataloader


if __name__ == '__main__':
    dataset_config = DatasetConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()

    training_dataloader, validating_dataloader = get_dataset(dataset_config, train_config)
    check_dir(model_config.base_path)
    m = run_train(n_state=dataset_config.n_state, training_dataloader=training_dataloader,
                  validating_dataloader=validating_dataloader, n_delay_point=dataset_config.n_delay_point,
                  weight_decay=train_config.weight_decay, n_hidden=model_config.deeponet_n_hidden,
                  n_epoch=train_config.n_epoch, merge_size=model_config.deeponet_n_merge_size,
                  hidden_size=model_config.deeponet_n_hidden_size, lr=train_config.learning_rate,
                  img_save_path=model_config.base_path,
                  n_modes_height=model_config.fno_n_modes_height, hidden_channels=model_config.fno_hidden_channels,
                  model_name=model_config.model_name)
    run_test(test_points=dataset_config.test_points, base_path=model_config.base_path, delay=dataset_config.delay,
             duration=dataset_config.duration, ts=dataset_config.ts, n_point_delay=dataset_config.n_delay_point,
             dt=dataset_config.dt)
