import argparse
import os
import random
import time
import uuid
import warnings
from typing import Literal, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from scipy.integrate import odeint
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from config import DatasetConfig, ModelConfig, TrainConfig
from dataset import ZUZDataset, ZUPDataset, PredictionDataset, sample_to_tensor
from dynamic_systems import solve_integral_nn, DynamicSystem, solve_integral, solve_integral_successive_batched
from model import GRUNet, LSTMNet
from plot_utils import plot_result, set_size, plot_switch_system, difference
from utils import pad_leading_zeros, metric, check_dir, predict_and_loss, load_lr_scheduler, prepare_datasets, \
    set_everything, print_result, postprocess, load_model, load_optimizer, prediction_comparison, print_args, \
    get_time_str, SimulationResult

warnings.filterwarnings('ignore')


def simulation(dataset_config: DatasetConfig, train_config: TrainConfig, Z0: Tuple | np.ndarray | List,
               method: Literal['explicit', 'numerical', 'no', 'numerical_no', 'switching', 'scheduled_sampling'] = None,
               model=None, img_save_path: str = None, silence: bool = True):
    system: DynamicSystem = dataset_config.system
    ts = dataset_config.ts
    Z0 = np.array(Z0)
    n_point = dataset_config.n_point
    n_point_start = dataset_config.n_point_start()
    max_n_point_delay = dataset_config.max_n_point_delay()
    alpha = train_config.cp_alpha
    alpha_t = train_config.cp_alpha
    U = np.zeros((n_point, system.n_input))
    Z = np.zeros((n_point, system.n_state))
    P_numerical_n_iters = np.zeros(n_point)
    switching_indicator = np.zeros(n_point)
    P_explicit = np.zeros((n_point, system.n_state))
    P_numerical = np.zeros((n_point, system.n_state))
    P_no = np.zeros((n_point, system.n_state))
    P_no_ci = np.zeros((n_point, system.n_state, 2))
    P_no_Ri = np.zeros(n_point)
    alpha_ts = np.zeros(n_point)
    q_ts = np.zeros(n_point)
    e_ts = np.zeros(n_point)
    P_switching = np.zeros((n_point, system.n_state))

    p_numerical_count = 0
    p_no_count = 0
    Z[n_point_start, :] = Z0
    runtime = 0.
    if isinstance(model, GRUNet) or isinstance(model, LSTMNet):
        model.reset_state()
    if silence:
        bar = range(dataset_config.n_point)
    else:
        bar = tqdm(range(dataset_config.n_point))
    for t_i in bar:
        t = ts[t_i]
        # t_i_delayed = max(t_i - n_point_start, 0)
        t_i_delayed = max(t_i - dataset_config.n_point_delay(t), 0)
        if method == 'explicit':
            U[t_i] = system.U_explicit(t, Z0)
            if t_i > n_point_start:
                Z[t_i, :] = system.Z_explicit(t, Z0)
        else:
            # estimate system state
            if t_i > n_point_start:
                Z[t_i, :] = odeint(system.dynamic, Z[t_i - 1, :], [ts[t_i - 1], ts[t_i]], args=(U[t_i_delayed - 1],))[1]
                Z_t = Z[t_i, :]
            else:
                Z_t = Z0 + dataset_config.noise()
            Z_t += dataset_config.noise()
            # estimate prediction and control signal
            if method == 'numerical':
                begin = time.time()
                solution = solve_integral(Z_t=Z_t, P_D=P_numerical[t_i_delayed:t_i], U_D=U[t_i_delayed:t_i], t=t,
                                          dataset_config=dataset_config, delay=dataset_config.delay)
                P_numerical[t_i, :] = solution.solution
                P_numerical_n_iters[t_i] = solution.n_iter
                end = time.time()
                runtime += end - begin
                if t_i >= n_point_start:
                    U[t_i] = system.kappa(P_numerical[t_i, :], t)
            elif method == 'no':
                U_D = pad_leading_zeros(segment=U[t_i_delayed:t_i], length=max_n_point_delay)
                begin = time.time()
                P_no[t_i, :] = solve_integral_nn(model=model, U_D=U_D, Z_t=Z_t, t=t)
                end = time.time()
                runtime += end - begin
                if t_i >= n_point_start:
                    U[t_i] = system.kappa(P_no[t_i, :], t)
            elif method == 'numerical_no':
                begin = time.time()
                solution = solve_integral(
                    Z_t=Z_t, P_D=P_numerical[t_i_delayed:t_i], U_D=U[t_i_delayed:t_i], t=t,
                    dataset_config=dataset_config, delay=dataset_config.delay)
                P_numerical[t_i, :] = solution.solution
                P_numerical_n_iters[t_i] = solution.n_iter

                U_D = pad_leading_zeros(segment=U[t_i_delayed:t_i], length=n_point_start)
                P_no[t_i, :] = solve_integral_nn(model=model, U_D=U_D, Z_t=Z_t, t=t)
                end = time.time()
                runtime += end - begin
                if t_i >= n_point_start:
                    U[t_i] = system.kappa(P_numerical[t_i, :], t)
            elif method == 'switching':
                U_D = pad_leading_zeros(segment=U[t_i_delayed:t_i], length=max_n_point_delay)
                begin = time.time()
                P_no[t_i, :] = solve_integral_nn(model=model, U_D=U_D, Z_t=Z_t, t=t)
                if t_i >= n_point_start:
                    # actuate the controller
                    start_point = 2 * n_point_start
                    # start_point = n_point_start
                    if t_i >= start_point:
                        # System switching
                        # (1) Get the uncertainty of no model
                        P_no_Ri[t_i] = np.linalg.norm(P_no[t_i - n_point_start, :] - Z_t)
                        if train_config.cp_adaptive:
                            Q = np.percentile(P_no_Ri[start_point:t_i + 1], (1 - min(1, max(alpha_t, 0))) * 100)
                        else:
                            Q = np.percentile(P_no_Ri[start_point:t_i + 1], (1 - alpha) * 100)
                        e_t = 0 if P_no_Ri[t_i] <= Q else 1
                        # (2) Assign the indicator
                        if train_config.cp_switching_type == 'switching':
                            if e_t == 0:
                                # switch to no scheme if possible
                                if switching_indicator[t_i - 1] == 1:
                                    t_last_no = np.where(switching_indicator[:t_i] == 0)[0][-1]
                                    if t_i - t_last_no > n_point_start:
                                        # switch back
                                        switching_indicator[t_i] = 0
                                    else:
                                        # cannot switch
                                        switching_indicator[t_i] = 1
                                else:
                                    # keeping using no
                                    switching_indicator[t_i] = 0
                            else:
                                # use to numerical scheme
                                switching_indicator[t_i] = 1
                        elif train_config.cp_switching_type == 'alternating':
                            # alternating between no and numerical arbitrarily
                            # if indicator is 1, then use numerical scheme, otherwise no scheme
                            switching_indicator[t_i] = e_t

                        # (3) Select the sub-controller
                        if switching_indicator[t_i] == 0:
                            P_switching[t_i, :] = P_no[t_i, :]
                            p_no_count += 1
                        else:
                            P_numerical[t_i, :] = solve_integral(
                                Z_t=Z_t, P_D=P_numerical[t_i_delayed:t_i], U_D=U[t_i_delayed:t_i], t=t,
                                dataset_config=dataset_config, delay=dataset_config.delay).solution
                            P_switching[t_i, :] = P_numerical[t_i, :]
                            p_numerical_count += 1
                        alpha_t += train_config.cp_gamma * (train_config.cp_alpha - e_t)
                        alpha_ts[t_i] = alpha_t
                        q_ts[t_i] = Q
                        e_ts[t_i] = e_t
                    else:
                        # Warm start
                        P_numerical[t_i, :] = solve_integral(
                            Z_t=Z_t, P_D=P_numerical[t_i_delayed:t_i], U_D=U[t_i_delayed:t_i], t=t,
                            dataset_config=dataset_config, delay=dataset_config.delay).solution
                        P_switching[t_i, :] = P_numerical[t_i, :]
                        p_numerical_count += 1
                        switching_indicator[t_i] = 1

                    U[t_i] = system.kappa(P_switching[t_i, :], t)
                end = time.time()
                runtime += end - begin
            elif method == 'scheduled_sampling':
                solution = solve_integral(
                    Z_t=Z_t, P_D=P_no[t_i_delayed:t_i], U_D=U[t_i_delayed:t_i], t=t, dataset_config=dataset_config,
                    delay=dataset_config.delay)
                P_numerical[t_i, :] = solution.solution
                if np.random.binomial(n=1, p=train_config.scheduled_sampling_p) == 1:
                    # Teacher Forcing
                    P_no[t_i, :] = solution.solution
                else:
                    # Not Teacher Forcing
                    P_no[t_i, :] = solve_integral_nn(model=model, U_D=pad_leading_zeros(
                        segment=U[t_i_delayed:t_i], length=max_n_point_delay), Z_t=Z_t, t=t)
                if t_i >= n_point_start:
                    U[t_i] = system.kappa(P_no[t_i, :], t)
            else:
                raise NotImplementedError()

    plot_result(dataset_config, img_save_path, P_no, P_numerical, P_explicit, P_switching, Z, U, method)

    D_explicit = difference(Z, P_explicit, n_point_start, dataset_config.n_point_delay, ts)
    D_no = difference(Z, P_no, n_point_start, dataset_config.n_point_delay, ts)
    D_numerical = difference(Z, P_numerical, n_point_start, dataset_config.n_point_delay, ts)
    D_switching = difference(Z, P_switching, n_point_start, dataset_config.n_point_delay, ts)
    return SimulationResult(
        U=U, Z=Z, D_explicit=D_explicit, D_no=D_no, D_numerical=D_numerical, D_switching=D_switching,
        P_explicit=P_explicit, P_no=P_no, P_no_ci=P_no_ci, P_numerical=P_numerical,
        P_switching=P_switching, runtime=runtime, P_numerical_n_iters=P_numerical_n_iters,
        p_numerical_count=p_numerical_count, p_no_count=p_no_count, P_no_Ri=P_no_Ri, alpha_ts=alpha_ts, q_ts=q_ts,
        e_ts=e_ts, switching_indicator=switching_indicator, avg_prediction_time=runtime / n_point)


def model_train(dataset_config: DatasetConfig, train_config: TrainConfig, model, optimizer, scheduler, device,
                training_dataloader, predict_and_loss, adversarial_epsilon: float = 0., n_state: int = None):
    model.train()
    training_loss = 0.0
    adversarial_loss = 0.0
    for inputs, labels in training_dataloader:
        # normal training
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, loss = predict_and_loss(inputs, labels, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

        # adversarial training
        if adversarial_epsilon > 0:
            inputs.requires_grad = True
            outputs, loss = predict_and_loss(inputs, labels, model)
            model.zero_grad()
            loss.backward()

            inputs_grad = inputs.grad.data
            # only adversarial the state part
            mask = torch.zeros_like(inputs_grad, dtype=torch.bool)
            mask[:, 1:1 + n_state] = 1
            inputs_grad[~mask] = 0

            adversarial_inputs = inputs + torch.rand((1,), device=device) * adversarial_epsilon * inputs_grad.sign()
            adversarial_labels = solve_integral_successive_batched(
                Z_t=np.array(adversarial_inputs[:, 1:1 + n_state].detach().cpu().numpy()),
                U_D=np.array(adversarial_inputs[:, 1 + n_state:].detach().cpu().numpy()),
                dt=dataset_config.dt, n_points=dataset_config.n_point_delay,
                f=dataset_config.system.dynamic, n_iterations=dataset_config.successive_approximation_n_iteration,
                adaptive=False)
            adversarial_labels = torch.from_numpy(adversarial_labels)
            outputs, loss = predict_and_loss(adversarial_inputs.to(device, dtype=torch.float32),
                                             adversarial_labels.to(device, dtype=torch.float32), model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adversarial_loss += loss.item()
    model.eval()
    lr = scheduler.get_last_lr()[-1]
    scheduler.step()
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(param_group['lr'], train_config.scheduler_min_lr)
    avg_training_loss = training_loss / len(training_dataloader)
    avg_adversarial_loss = adversarial_loss / len(training_dataloader)
    return avg_training_loss, avg_adversarial_loss, lr


def model_validate(model, device, dataloader):
    with torch.no_grad():
        validating_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, loss = predict_and_loss(inputs, labels, model)
            validating_loss += loss.item()
        return validating_loss / len(dataloader)


def run_offline_training(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig):
    device = train_config.device
    adversarial_epsilon = train_config.adversarial_epsilon
    n_epoch = train_config.n_epoch
    img_save_path = model_config.base_path

    model, model_loaded = load_model(train_config, model_config, dataset_config)
    optimizer = load_optimizer(model.parameters(), train_config)
    scheduler = load_lr_scheduler(optimizer, train_config)

    if not train_config.do_training:
        return model
    else:
        training_dataloader, validating_dataloader = load_training_and_validation_datasets(dataset_config, train_config)
    testing_dataloader = load_test_datasets(dataset_config, train_config)
    check_dir(model_config.base_path)
    check_dir(train_config.model_save_path)

    if model_loaded:
        print(f'Model loaded, skip training!')
    else:
        training_loss_arr = []
        validating_loss_arr = []
        testing_loss_arr = []
        adversarial_loss_arr = []
        rl2_list = []
        l2_list = []
        bar = tqdm(list(range(n_epoch)))
        do_validation = validating_dataloader is not None
        do_testing = testing_dataloader is not None
        do_adversarial = train_config.adversarial_epsilon > 0

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
                p = f'{img_save_path}/loss.png'
                plt.savefig(p)
                print(f'save loss curve to {p}')
                fig.clear()
                plt.close(fig)
            else:
                plt.show()

        for epoch in bar:
            training_loss_t, adversarial_loss_t, lr = model_train(
                dataset_config, train_config, model, optimizer, scheduler, device, training_dataloader,
                predict_and_loss, adversarial_epsilon, dataset_config.n_state)
            training_loss_arr.append(training_loss_t)
            desc = f'Epoch [{epoch + 1}/{n_epoch}] || Lr: {lr:6f} || Training loss: {training_loss_t:.6f}'
            if do_validation:
                validating_loss_t = model_validate(model, device, validating_dataloader)
                validating_loss_arr.append(validating_loss_t)
                desc += f' || Validation loss: {validating_loss_t:.6f}'
            else:
                validating_loss_t = 0
            if do_testing:
                testing_loss_t = model_validate(model, device, testing_dataloader)
                testing_loss_arr.append(testing_loss_t)
                desc += f' || Test loss: {testing_loss_t:.6f}'
            else:
                testing_loss_t = 0
            if do_adversarial:
                adversarial_loss_arr.append(adversarial_loss_t)
                desc += f' || Adversarial loss: {adversarial_loss_t:.6f}'
            bar.set_description(desc)
            wandb.log({
                'training loss': training_loss_t,
                'validation loss': validating_loss_t,
                'test loss': testing_loss_t,
                'learning rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
            if (train_config.log_step > 0 and epoch % train_config.log_step == 0) or epoch == n_epoch - 1:
                ...
        draw()
    print('Finished Training')
    return model


def run_scheduled_sampling_training(dataset_config: DatasetConfig, model_config: ModelConfig,
                                    train_config: TrainConfig):
    warnings.filterwarnings('error')
    device = train_config.device
    n_epoch = train_config.n_epoch
    img_save_path = model_config.base_path
    model, model_loaded = load_model(train_config, model_config, dataset_config)
    optimizer = load_optimizer(model.parameters(), train_config)
    scheduler = load_lr_scheduler(optimizer, train_config)
    training_loss_arr = []
    scheduled_sampling_p_arr = []
    check_dir(img_save_path)
    print('Begin Training...')
    dataloader = None
    for epoch in range(train_config.n_epoch):
        if epoch % train_config.scheduled_sampling_frequency == 0 or dataloader is None:
            train_config.set_scheduled_sampling_p(epoch)
            scheduled_sampling_p_arr.append(train_config.scheduled_sampling_p)

            Z0 = np.random.uniform(low=dataset_config.ic_lower_bound, high=dataset_config.ic_upper_bound,
                                   size=(dataset_config.n_state,))
            try:
                result = simulation(dataset_config, train_config, Z0, 'scheduled_sampling', model,
                                    img_save_path=img_save_path)
            except Warning as e:
                print("Warning caught:", e)
                print(f'Abnormal value encountered at epoch {epoch}, skip this epoch!')
                continue
            predictions = prediction_comparison(result.P_numerical, dataset_config.n_point_delay, dataset_config.ts)
            true_values = result.Z[dataset_config.n_point_delay(0):]

            Ps = np.array(predictions)
            Zs = np.array(true_values)
            horizon = np.array(dataset_config.ts[dataset_config.n_point_delay(0):])
            Us = np.array([result.U[t_i: t_i + dataset_config.n_point_delay(0)] for t_i in range(len(horizon))])
            if np.isnan(Ps).any() or np.isnan(Zs).any() or np.isnan(
                    horizon).any() or np.isnan(Us).any() or np.isinf(Ps).any() or np.isinf(
                Zs).any() or np.isinf(horizon).any() or np.isinf(Us).any():
                training_loss_arr.append(0)
                continue
            samples = []
            for t_i, (p, z, t) in enumerate(zip(Ps, Zs, horizon)):
                u = Us[t_i]
                samples.append((sample_to_tensor(z, u, t.reshape(-1)), torch.from_numpy(p)))
            dataloader = DataLoader(PredictionDataset(samples), batch_size=train_config.batch_size, shuffle=False)

        training_loss_t, _, _ = model_train(dataset_config, train_config, model, optimizer, scheduler, device,
                                            dataloader, predict_and_loss)
        print(f'Epoch [{epoch + 1}/{n_epoch}]'
              f'|| Scheduled Sampling Rate {train_config.scheduled_sampling_p}'
              f'|| Training loss: {training_loss_t:.6f}')
        wandb.log({
            'sampling rate': train_config.scheduled_sampling_p,
            'training loss': training_loss_t,
            'learning rate': optimizer.param_groups[0]['lr'],
        }, step=epoch)
        training_loss_arr.append(training_loss_t)

    fig = plt.figure(figsize=set_size())
    plt.plot(training_loss_arr, label="Training loss")
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'{img_save_path}/loss.png')
    fig.clear()
    plt.close(fig)

    fig = plt.figure(figsize=set_size())
    plt.plot(scheduled_sampling_p_arr, label="Scheduled Sampling P")
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'{img_save_path}/p.png')
    fig.clear()
    plt.close(fig)
    return model


def run_sequence_training(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig):
    device = train_config.device
    img_save_path = model_config.base_path
    model, model_loaded = load_model(train_config, model_config, dataset_config)
    optimizer = load_optimizer(model.parameters(), train_config)
    scheduler = load_lr_scheduler(optimizer, train_config)
    training_loss_arr = []
    check_dir(img_save_path)
    max_n_point_delay = dataset_config.max_n_point_delay()
    print('Begin Generating Dataset...')
    samples_all_dataset = []
    for _ in tqdm(range(dataset_config.n_dataset)):
        Z0 = np.random.uniform(low=dataset_config.ic_lower_bound, high=dataset_config.ic_upper_bound,
                               size=(dataset_config.n_state,))
        result = simulation(dataset_config, train_config, Z0, 'numerical', model)

        n_point_start = dataset_config.n_point_start()
        # predictions = head_points(result.P_numerical, n_point_start)
        n_point_delay = dataset_config.n_point_delay
        predictions = prediction_comparison(result.P_numerical, n_point_delay, dataset_config.ts)
        true_values = result.Z[n_point_start:]

        Ps = np.array(predictions)
        Zs = np.array(true_values)
        horizon = np.array(dataset_config.ts[n_point_start:])
        # Us = [result.U[idx: idx + n_point_delay(t)] for idx, t in enumerate(horizon)]
        Us = [pad_leading_zeros(result.U[idx: idx + n_point_delay(t)], max_n_point_delay) for idx, t in
              enumerate(horizon)]

        samples = []
        for t_i, (p, z, t) in enumerate(zip(Ps, Zs, horizon)):
            samples.append((sample_to_tensor(z, Us[t_i], t.reshape(-1)), torch.from_numpy(p)))
        samples_all_dataset.append(samples)

    model.train()
    print('Begin Training...')
    for epoch in tqdm(range(train_config.n_epoch)):
        training_loss = 0.0
        np.random.shuffle(samples_all_dataset)
        n_epoch = 0
        for i in range(0, len(samples_all_dataset), train_config.batch_size):
            sequences = samples_all_dataset[i:i + train_config.batch_size]
            if isinstance(model, GRUNet) or isinstance(model, LSTMNet):
                optimizer.zero_grad()
                model.reset_state()
                losses = []
                for batch in zip(*sequences):
                    inputs, labels = torch.vstack([batch[i][0] for i in range(train_config.batch_size)]), torch.vstack(
                        [batch[i][1] for i in range(train_config.batch_size)])
                    inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
                    outputs, loss = predict_and_loss(inputs, labels, model)
                    losses.append(loss)
                loss_ = sum(losses) / len(losses)
                loss_.backward()
                optimizer.step()
                training_loss += loss_.item()
            else:
                losses = []
                for batch in zip(*sequences):
                    optimizer.zero_grad()
                    inputs, labels = torch.vstack([batch[i][0] for i in range(train_config.batch_size)]), torch.vstack(
                        [batch[i][1] for i in range(train_config.batch_size)])
                    inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
                    outputs, loss = predict_and_loss(inputs, labels, model)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.detach())
                training_loss += (sum(losses) / len(losses)).item()
            n_epoch += 1
        training_loss /= n_epoch
        scheduler.step()

        wandb.log({
            'training loss': training_loss,
            'lr': optimizer.param_groups[0]['lr'],
        }, step=epoch)
        training_loss_arr.append(training_loss)

    fig = plt.figure(figsize=set_size())
    plt.plot(training_loss_arr, label="Training loss")
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'{img_save_path}/loss.png')
    fig.clear()
    plt.close(fig)

    return model


def run_test(m, dataset_config: DatasetConfig, train_config: TrainConfig, method: str, base_path: str = None,
             silence: bool = False, test_points: List = None, plot: bool = False):
    base_path = f'{base_path}/{method}'
    if test_points is None:
        test_points = dataset_config.test_points

    bar = test_points if silence else tqdm(test_points)
    rl2_list = []
    l2_list = []
    prediction_time = []
    n_iter_list = []
    for test_point, name in bar:
        if not silence:
            bar.set_description(f'Solving system with initial point {np.round(test_point, decimals=3)}.')
            bar.set_description(f'Save to {name}.')

        img_save_path = f'{base_path}/{name}'
        check_dir(img_save_path)
        result = simulation(dataset_config=dataset_config, train_config=train_config, model=m, Z0=test_point,
                            method=method, img_save_path=img_save_path)
        plt.close()
        n_point_start = dataset_config.n_point_start()
        if method == 'no':
            P = result.P_no
        elif method == 'numerical':
            P = result.P_numerical
        elif method == 'numerical_no':
            P = result.P_no
        elif method == 'switching':
            P = result.P_switching
        else:
            raise NotImplementedError()

        rl2, l2 = metric(P, result.Z, dataset_config.n_point_delay, dataset_config.ts)
        if np.isinf(rl2) or np.isnan(rl2):
            if not silence:
                print(f'[WARNING] Running with initial condition Z = {test_point} with method [{method}] failed.')
            continue

        if method == 'switching':
            print()
            plot_switch_system(train_config, dataset_config, result, n_point_start, img_save_path)
            print('no count:', result.p_no_count)
            print('numerical count:', result.p_numerical_count)

        np.savetxt(f'{img_save_path}/metric.txt', np.array([rl2, l2, result.runtime]))
        np.savetxt(f'{img_save_path}/test_point.txt', test_point)
        rl2_list.append(rl2)
        l2_list.append(l2)
        prediction_time.append(result.avg_prediction_time)
        n_iter_list.append(result.P_numerical_n_iters)
    rl2 = np.nanmean(rl2_list).item()
    l2 = np.nanmean(l2_list).item()
    prediction = np.nanmean(prediction_time).item()
    to_save = [rl2, l2, prediction]
    if method == 'numerical':
        n_iter = np.concatenate(n_iter_list).mean()
        to_save.append(n_iter)
        print(f'Numerical method uses {n_iter} iterations on average.')
    np.savetxt(f'{base_path}/metric.txt', np.array(to_save))
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
    return rl2, l2, prediction, len(rl2_list)


def load_training_and_validation_datasets(dataset_config: DatasetConfig, train_config: TrainConfig):
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
        train_points = list(
            np.random.uniform(dataset_config.ic_lower_bound, dataset_config.ic_upper_bound,
                              (int(dataset_config.n_dataset * train_config.training_ratio), dataset_config.n_state)))
        validation_points = list(
            np.random.uniform(dataset_config.ic_lower_bound, dataset_config.ic_upper_bound,
                              (dataset_config.n_dataset - int(dataset_config.n_dataset * train_config.training_ratio),
                               dataset_config.n_state)))
        if dataset_config.data_generation_strategy == 'trajectory':
            training_samples = create_trajectory_dataset(dataset_config, train_config, train_points)
            validation_samples = create_trajectory_dataset(dataset_config, train_config, validation_points)
        else:
            raise NotImplementedError()

        print(f'Created {len(training_samples)} samples')
        if dataset_config.postprocess:
            training_samples = postprocess(training_samples, dataset_config)
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

    training_dataloader, validating_dataloader = prepare_datasets(
        training_samples, train_config.batch_size, training_ratio=train_config.training_ratio,
        validation_samples=validation_samples)

    print(f'#Training sample: {int(len(training_samples) * train_config.training_ratio)}')
    print(f'#Validating sample: {len(training_samples) - int(len(training_samples) * train_config.training_ratio)}')
    path = dataset_config.training_dataset_file
    if dataset_config.recreate_training_dataset:
        torch.save(training_samples, path)
        np.savetxt(f'{dataset_config.dataset_base_path}/n_sample.txt',
                   np.array([len(training_samples), train_config.training_ratio]))
        print(f'{len(training_samples)} samples saved')
    return training_dataloader, validating_dataloader


def load_test_datasets(dataset_config, train_config):
    if not train_config.do_testing:
        return None
    if not os.path.exists(dataset_config.testing_dataset_file) or dataset_config.recreate_testing_dataset:
        print('Creating testing dataset')
        testing_samples = create_trajectory_dataset(dataset_config, train_config, dataset_config.test_points)
    else:
        print('Loading testing dataset')
        testing_samples = torch.load(dataset_config.testing_dataset_file)
    testing_dataloader = DataLoader(PredictionDataset(testing_samples), batch_size=train_config.batch_size,
                                    shuffle=False)
    print(f'#Testing sample: {len(testing_samples)}')
    return testing_dataloader


def create_trajectory_dataset(dataset_config: DatasetConfig, train_config: TrainConfig,
                              initial_conditions: List = None):
    all_samples = []
    if dataset_config.z_u_p_pair:
        print('creating datasets of Z(t), U(t-D~t), P(t) pairs')
    else:
        print('creating datasets of Z(t), U(t-D~t), Z(t+D) pairs')
    if initial_conditions is None:
        bar = tqdm(list(np.random.uniform(dataset_config.ic_lower_bound, dataset_config.ic_upper_bound,
                                          (dataset_config.n_dataset, dataset_config.n_state))))
    else:
        bar = tqdm(initial_conditions)
    for i, Z0 in enumerate(bar):
        if dataset_config.z_u_p_pair:
            result = simulation(method='numerical', Z0=Z0, dataset_config=dataset_config, train_config=train_config)
            dataset = ZUPDataset(
                torch.tensor(result.Z, dtype=torch.float32), torch.tensor(result.U, dtype=torch.float32),
                torch.tensor(result.P_numerical, dtype=torch.float32), dataset_config.n_point_delay,
                dataset_config.dt, dataset_config.ts)
        else:
            result = simulation(method='explicit', Z0=Z0, dataset_config=dataset_config, train_config=train_config)
            dataset = ZUZDataset(
                torch.tensor(result.Z, dtype=torch.float32), torch.tensor(result.U, dtype=torch.float32),
                dataset_config.n_point_delay, dataset_config.dt)
        dataset = list(dataset)
        random.shuffle(dataset)
        if dataset_config.n_sample_per_dataset >= 0:
            all_samples += dataset[:dataset_config.n_sample_per_dataset]
        else:
            all_samples += dataset
        wandb.log({
            "number dataset": i + 1
        })
    random.shuffle(all_samples)
    return all_samples


def main(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig):
    if train_config.training_type == 'offline' or train_config.training_type == 'switching':
        model = run_offline_training(dataset_config=dataset_config, model_config=model_config,
                                     train_config=train_config)
    elif train_config.training_type == 'sequence':
        model = run_sequence_training(dataset_config=dataset_config, model_config=model_config,
                                      train_config=train_config)
    elif train_config.training_type == 'scheduled sampling':
        model = run_scheduled_sampling_training(dataset_config=dataset_config, model_config=model_config,
                                                train_config=train_config)
    else:
        raise NotImplementedError()
    model_save_path = f'{train_config.model_save_path}/{model_config.model_name}.pth'
    torch.save(model.state_dict(), model_save_path)
    arti_model = wandb.Artifact('model', type='model')
    arti_model.add_file(model_save_path)
    wandb.log_artifact(arti_model)

    test_points = [(tp, uuid.uuid4()) for tp in dataset_config.test_points]
    print('All test points:')
    print(test_points)

    if train_config.training_type == 'switching':
        return {
            'no': run_test(m=model, dataset_config=dataset_config, train_config=train_config,
                           base_path=model_config.base_path, test_points=test_points, method='no'),
            'switching': run_test(m=model, dataset_config=dataset_config, train_config=train_config,
                                  base_path=model_config.base_path, test_points=test_points, method='switching'),
            'numerical': run_test(m=model, dataset_config=dataset_config, train_config=train_config,
                                  base_path=model_config.base_path, test_points=test_points, method='numerical'),
            'numerical_no': run_test(m=model, dataset_config=dataset_config, train_config=train_config,
                                     base_path=model_config.base_path, test_points=test_points, method='numerical_no')
        }
    else:
        return {
            'no': run_test(m=model, dataset_config=dataset_config, train_config=train_config,
                           base_path=model_config.base_path, test_points=test_points, method='no'),
            'numerical': run_test(m=model, dataset_config=dataset_config, train_config=train_config,
                                  base_path=model_config.base_path, test_points=test_points, method='numerical'),
            'numerical_no': run_test(m=model, dataset_config=dataset_config, train_config=train_config,
                                     base_path=model_config.base_path, test_points=test_points, method='numerical_no')
        }


if __name__ == '__main__':
    set_everything(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, default='s7')
    parser.add_argument('-delay', type=float, default=None)
    parser.add_argument('-training_type', type=str, default='sequence')
    parser.add_argument('-model_name', type=str, default='GRU')
    parser.add_argument('-tlb', type=float, default=0.)
    parser.add_argument('-tub', type=float, default=1.)
    parser.add_argument('-cp_gamma', type=float, default=0.01)
    parser.add_argument('-cp_alpha', type=float, default=0.1)

    args = parser.parse_args()
    dataset_config_, model_config_, train_config_ = config.get_config(system_=args.s, delay=args.delay,
                                                                      model_name=args.model_name)
    # dataset_config_.n_dataset = 1
    # train_config_.batch_size = 1
    assert torch.cuda.is_available()
    train_config_.training_type = args.training_type
    if args.training_type == 'offline' or args.training_type == 'sequence':
        ...
    elif args.training_type == 'switching':
        ...
    elif args.training_type == 'scheduled sampling':
        if dataset_config_.system_ == 's1':
            train_config_.n_epoch = 3000
        else:
            train_config_.n_epoch = 2000
        train_config_.lr_scheduler_type = 'none'
    else:
        raise NotImplementedError()

    print_args(dataset_config_)
    print_args(model_config_)
    print_args(train_config_)
    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    wandb.init(
        project="no",
        name=f'{train_config_.system} {train_config_.training_type} {model_config_.model_name} {dataset_config_.delay.__class__.__name__} {get_time_str()}'
    )
    results_ = main(dataset_config_, model_config_, train_config_)
    for method_, result_ in results_.items():
        print(method_)
        print_result(result_, dataset_config_)
        speedup = results_["numerical"][2] / result_[2]
        print(f'Speedup w.r.t numerical: {speedup :.3f}; $\\times {speedup:.3f}$')
    wandb.finish()
