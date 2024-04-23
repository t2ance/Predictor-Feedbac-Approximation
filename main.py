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
from dataset import ImplicitDataset, PredictionDataset, ExplictDataset
from model import PredictionFNO
from utils import count_params, padding_leading_zero


def system(z, t, u_t_minus_D, Z0, D):
    z1, z2 = z
    if t < D:
        return Z0[0] + Z0[1] * t, Z0[1]
    else:
        try:
            return [z2 - z2 ** 2 * u_t_minus_D, u_t_minus_D]
        except:
            print(z, t, D, Z0)
            raise NotImplementedError()


def control_law_explict(t, Z0, D):
    z1_0, z2_0 = Z0
    if t >= 0:
        term1 = z1_0 + (2 + D) * z2_0 + (1 / 3) * z2_0 ** 3
        term2 = z1_0 + (1 + D) * z2_0 + (1 / 3) * z2_0 ** 3
        u_t = -np.exp(D - t) * (term1 + (D - t) * term2)
        return u_t
    elif t >= -D:
        return 0
    else:
        raise NotImplementedError()


def control_law(P_t):
    p1t = P_t[0]
    p2t = P_t[1]
    return -p1t - 2 * p2t - 1 / 3 * p2t ** 3


def solve_z_explict(t, D, Z0):
    if t < 0:
        raise NotImplementedError()
    if t < D:
        return Z0[0] + Z0[1] * t, Z0[1]
    z1_0 = Z0[0]
    z2_0 = Z0[1]
    z2_D = Z0[1]
    middle_term = z1_0 + D * z2_0 + (1 / 3) * z2_0 ** 3
    term1 = np.exp(D - t) * ((1 + t - D) * middle_term + (t - D) * z2_D)
    term2 = - (1 / 3) * np.exp(3 * (D - t)) * ((D - t) * middle_term + (1 - t + D) * z2_D) ** 3
    Z1 = term1 + term2

    Z2 = np.exp(D - t) * ((D - t) * middle_term + (1 - t + D) * z2_D)
    return Z1, Z2


def solve_z_ode(u, i, span, D, Z):
    raise NotImplementedError()
    return odeint(system, Z[i, :], span, args=(D))


def solve_z_neural_operator(model, Us, z_t, time_step):
    u_tensor = torch.tensor(Us, dtype=torch.float32).view(1, -1)
    z_tensor = torch.tensor(z_t, dtype=torch.float32).view(1, -1)
    inputs = [torch.cat([z_tensor, u_tensor], dim=1), torch.tensor(time_step, dtype=torch.float32).view(1, -1)]
    if isinstance(model, PredictionFNO):
        inputs = inputs[0]
    outputs = model(inputs)
    [P1, P2] = outputs.to('cpu').detach().numpy()[0]
    return P1, P2


def integral_prediction_for_example(t, D, Z0, Z, i, n_point_theta=1000):
    d_theta = 1 / n_point_theta
    term1 = sum(
        [control_law_explict(theta, Z0, D) * d_theta for theta in
         np.linspace(t - D, t, n_point_theta)]
    )
    term2 = sum(
        [(t - theta) * control_law_explict(theta, Z0, D) * d_theta for theta in
         np.linspace(t - D, t, n_point_theta)]
    )
    # term3 = sum([(2 + t - theta) * control_law_explict(theta, Z0[0], Z0[1], D) * 0.001 for theta in
    #              np.linspace(t - D, t, 1000)])
    Z1_t = Z[i, :][0]
    Z2_t = Z[i, :][1]
    P1_t = Z1_t + D * Z2_t + term2 - Z2_t ** 2 * term1 - Z2_t * term1 ** 2
    P2_t = Z2_t + term1
    return [P1_t, P2_t]


def integral_prediction_general(t, D, Z0, n_point_theta=1000):
    d_theta = 1 / n_point_theta
    term1 = np.array(
        [f(solve_z_explict(theta + D, D, Z0), control_law_explict(theta, Z0, D)) * d_theta for theta in
         np.linspace(t - D, t, n_point_theta)]
    ).sum(axis=0)
    return term1 + solve_z_explict(t, D, Z0)


def run(D: float, Z0: Tuple, duration: float, n_point: int, silence: bool = False, plot: bool = False, model=None,
        method: Literal['explict', 'numerical', 'no'] = None, title='', save_path: str = None,
        img_save_path: str = None):
    if not silence:
        print(f'Solving with method "{method}"')
    ts = np.linspace(0, duration, n_point)
    dt = ts[1] - ts[0]
    D_steps = int(D / dt)
    Z = np.zeros((len(ts), 2))
    P = np.zeros((len(ts), 2))
    U = np.zeros(len(ts))
    Z[0, :] = Z0

    if silence:
        sequence = range(len(ts))
    else:
        sequence = tqdm(list(range(len(ts))))

    for i in sequence:
        t = ts[i]
        Z[i, :] = solve_z_explict(ts[i], D, Z0)
        # P[i, :] = integral_prediction_general(t, D, Z0)
        if t < D:
            U[i] = control_law_explict(t, Z0, D)
        else:
            if method == 'explict':
                U[i] = control_law_explict(t, Z0, D)
                # U_t = -Z1_t - (2 + D) * Z2_t - 1 / 3 * Z2_t ** 3 - term3
                # print(U_t - U[i])
                P[i, :] = integral_prediction_for_example(t, D, Z0, Z, i)
            elif method == 'no':
                U_input = padding_leading_zero(U, i, D_steps)
                P[i, :] = solve_z_neural_operator(model, U_input, Z[i, :], i)
                U[i] = control_law(P[i, :])
            elif method == 'numerical':
                # sum_ = np.array([f(p, u) for p, u in zip(P[i - D_steps:i, :], U[i - D_steps:i])]).sum(axis=0)
                sum_ = np.array([f(p, u) for p, u in zip(Z[i:i + D_steps, :], U[i - D_steps:i])]).sum(axis=0)
                integral = sum_ * dt
                P[i, :] = integral + Z[i, :]
                # U[i] = control_law(P[i, :])
                U[i] = control_law_explict(t, Z0, D)
            else:
                raise NotImplementedError()
    if not silence:
        print(f'Finish solving')
    if save_path is not None:
        result = {
            "u": U,
            "z": Z,
            "d": D,
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
        plt.ylim([0, 1])
        plt.plot(ts, P[:, 0], label='$P_1(t)$')
        plt.ylabel('$P_1(t)$')
        plt.grid(True)

        plt.subplot(515)
        plt.ylim([-1, 1])
        plt.plot(ts, P[:, 1], label='$P_2(t)$')
        plt.ylabel('$P_2(t)$')
        plt.grid(True)
        plt.clf()

        plt.title('Comparison')
        # plt.ylim([-5, 5])
        for i in range(2):
            plt.plot(ts[D_steps:], P[:-D_steps, i], label=f'$P_{i + 1}(t-{D})$')
            plt.plot(ts[D_steps:], Z[D_steps:, i], label=f'$Z_{i + 1}(t)$')
        plt.legend()

        plt.tight_layout()
        if img_save_path is not None:
            plt.savefig(f'{img_save_path}/system.png')
            plt.clf()
        else:
            plt.show()
    return U, Z, P


def no_predict(model_name, inputs, device, model):
    inputs = inputs.to(device)
    time_step = inputs[:, :1]
    z_u = inputs[:, 1:]

    inputs = [z_u, time_step]
    if model_name == 'FNO':
        inputs = z_u
    return model(inputs)


def run_train(n_state: int, batch_size: int, hidden_size: int, n_hidden: int, merge_size: int, lr: float, n_epoch: int,
              weight_decay: float, training_dataloader=None, validating_dataloader=None, D_steps=None,
              dataset_path: str = None, n_modes_height=8, hidden_channels=16,
              model_name: Literal['DeepONet', 'FNO'] = 'DeepONet', device='cuda', img_save_path: str = None):
    if training_dataloader is None:
        with open(dataset_path, 'rb') as file:
            dataset = pickle.load(file)
        Z = dataset['z']
        U = dataset['u']
        D = dataset['d']
        ts = dataset['ts']
        D_steps = int(D / (ts[1] - ts[0]))
        prediction_dataset = ImplicitDataset(
            torch.tensor(Z, dtype=torch.float32),
            torch.tensor(U, dtype=torch.float32),
            D_steps)
        training_dataloader = DataLoader(
            prediction_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))

    layer_size_branch = [D_steps + n_state] + [hidden_size] * n_hidden + [merge_size]
    layer_size_trunk = [1] + [hidden_size] * n_hidden + [merge_size]
    if model_name == 'DeepONet':
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
            n_modes_height=n_modes_height, hidden_channels=hidden_channels, in_features=D_steps + n_state,
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


def split_list(list, ratio):
    n_total = len(list)
    n_sample = int(n_total * ratio)
    random.shuffle(list)
    return list[:n_sample], list[n_sample:]


def create_trajectory_dataset(n_dataset: int, duration: float, D: float, n_point: int, D_steps: int,
                              n_sample_per_dataset: int, implicit: bool, dataset_file: str):
    all_samples = []
    if implicit:
        print('creating implicit datasets')
    else:
        print('creating explicit datasets')
    for z in tqdm(list(np.random.uniform(0, 1, (n_dataset, 2)))):
        U, Z, P = run(method='explict', silence=True, duration=duration, D=D, Z0=z, n_point=n_point)
        if implicit:
            dataset = ImplicitDataset(
                torch.tensor(Z, dtype=torch.float32), torch.tensor(U, dtype=torch.float32), D_steps)
        else:
            dataset = ExplictDataset(
                torch.tensor(Z, dtype=torch.float32), torch.tensor(U, dtype=torch.float32),
                torch.tensor(P, dtype=torch.float32), D_steps)
        dataset = list(dataset)
        random.shuffle(dataset)
        all_samples += dataset[:n_sample_per_dataset]
    random.shuffle(all_samples)
    with open(dataset_file, 'wb') as file:
        pickle.dump(all_samples, file)
    return all_samples


def predict_by_integral(f, Z_t, U, D_steps, dt, n_state):
    P = np.zeros((n_state, D_steps))
    P[:, 0] = Z_t
    for i in range(D_steps - 1):
        increment = f(P[:, i], U[i])
        P[:, i + 1] = P[:, i] + dt * increment
    return P[:, -1]


def f(Z_t, U_t_minus_D):
    return np.array([Z_t[1] - Z_t[1] ** 2 * U_t_minus_D, U_t_minus_D])


def create_stateless_dataset(n_dataset: int, dt: float, D_steps: int,
                             n_sample_per_dataset: int, dataset_file: str, n_state: int):
    all_samples = []
    for i in range(n_dataset * n_sample_per_dataset):
        U = np.sin(np.sqrt(i) * np.linspace(0, 1, D_steps))

        Z_t = np.random.uniform(0, 1, 2)
        P = predict_by_integral(f, Z_t, U, D_steps, dt, n_state)
        features = np.concatenate([[i * dt], Z_t, U])
        label = P
        all_samples.append((features, label))
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


def run_test(test_points: List[Tuple[float, float]], base_path: str, delay: float, duration: float, n_point: int,
             ts: np.ndarray, D_steps: int):
    for test_point in test_points:
        img_save_path = f'{base_path}/{test_point}'
        check_dir(img_save_path)
        U_no, Z_no, P_no = run(model=m, D=delay, Z0=test_point, method='no', plot=True, title='no', duration=duration,
                               n_point=n_point, img_save_path=img_save_path)
        U_explict, Z_explict, P_explict = run(model=m, D=delay, Z0=test_point, method='explict', plot=False,
                                              title='explict', duration=duration, n_point=n_point,
                                              img_save_path=img_save_path)
        t_comparison = ts[D_steps:]
        plt.title('Comparison')
        plt.plot(t_comparison, P_no[:-D_steps], label='approximation')
        plt.plot(t_comparison, Z_explict[D_steps:], label='ground truth')
        plt.legend()
        plt.savefig(f'{img_save_path}/comparison.png')
        plt.clf()
        plt.title('Difference')
        difference = P_no[:-D_steps] - Z_explict[D_steps:]
        plt.ylim([-1, 1])
        plt.plot(t_comparison, difference[:, 0], label='difference of prediction1')
        plt.plot(t_comparison, difference[:, 1], label='difference of prediction2')
        plt.legend()
        plt.savefig(f'{img_save_path}/difference.png')
        plt.clf()
        # epsilon_z = metric(P_no[D_steps:-D_steps], Z_explict[2 * D_steps:])
        # epsilon_u = metric(U_no, U_explict)
        # print(f'error z: {epsilon_z}, error u: {epsilon_u}')


if __name__ == '__main__':
    dataset_config = DatasetConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()

    if not os.path.exists(dataset_config.dataset_file) or dataset_config.recreate_dataset:
        if dataset_config.trajectory:
            samples_ = create_trajectory_dataset(
                dataset_config.n_dataset, dataset_config.duration, dataset_config.delay, dataset_config.n_point,
                dataset_config.n_delay_step, dataset_config.n_sample_per_dataset, dataset_config.implicit,
                dataset_config.dataset_file)
        else:
            samples_ = create_stateless_dataset(dataset_config.n_dataset, dataset_config.dt,
                                                dataset_config.n_delay_step, dataset_config.n_sample_per_dataset,
                                                dataset_config.dataset_file, dataset_config.n_state)
    else:
        with open(dataset_config.dataset_file, 'rb') as file:
            samples_ = pickle.load(file)
    training_dataloader, validating_dataloader = prepare_dataset(
        samples_, train_config.training_ratio, train_config.batch_size, train_config.device)
    check_dir(model_config.base_path)
    m = run_train(n_state=dataset_config.n_state, training_dataloader=training_dataloader,
                  validating_dataloader=validating_dataloader, D_steps=dataset_config.n_delay_step,
                  weight_decay=train_config.weight_decay, n_hidden=model_config.deeponet_n_hidden,
                  n_epoch=train_config.n_epoch, merge_size=model_config.deeponet_n_merge_size,
                  hidden_size=model_config.deeponet_n_hidden_size, lr=train_config.learning_rate,
                  batch_size=train_config.batch_size, img_save_path=model_config.base_path,
                  n_modes_height=model_config.fno_n_modes_height, hidden_channels=model_config.fno_hidden_channels,
                  model_name=model_config.model_name)
    run_test(dataset_config.test_points, model_config.base_path, dataset_config.delay, dataset_config.duration,
             dataset_config.n_point, dataset_config.time_steps, dataset_config.n_delay_step)
