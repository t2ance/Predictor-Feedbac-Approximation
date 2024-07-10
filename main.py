import argparse
import os
import random
import time
from typing import Literal, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from scipy.integrate import odeint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
import dynamic_systems
from config import DatasetConfig, ModelConfig, TrainConfig
from dataset import ZUZDataset, ZUPDataset, PredictionDataset, sample_to_tensor
from dynamic_systems import solve_integral_eular, solve_integral_nn, solve_integral_ci_nn
from model import FullyConnectedNet, FourierNet, ChebyshevNet, BSplineNet, MapieRegressors, QuantileRegressionModel
from utils import set_size, pad_leading_zeros, plot_comparison, plot_difference, \
    metric, check_dir, plot_sample, predict_and_loss, load_lr_scheduler, prepare_datasets, draw_distribution, \
    set_seed, plot_single, print_result, postprocess, load_model, load_optimizer, plot_uncertainty, shift, \
    split_dataset, quantile_predict_and_loss, print_args, get_time_str, SimulationResult


def simulation(dataset_config: DatasetConfig, Z0: Tuple | np.ndarray | List,
               method: Literal['explicit', 'numerical', 'no', 'numerical_no', 'switching', 'scheduled_sampling'] = None,
               model=None, mapie_regressors=None, img_save_path: str = None, uncertainty_out: bool = False):
    system: dynamic_systems.DynamicSystem = dataset_config.system
    n_point_delay = dataset_config.n_point_delay
    n_state = dataset_config.n_state
    ts = dataset_config.ts
    Z0 = np.array(Z0)
    n_point = dataset_config.n_point
    U = np.zeros((n_point, system.n_input))
    Z = np.zeros((n_point, system.n_state))

    if method == 'explicit':
        P_explicit = np.zeros((n_point, system.n_state))
    else:
        P_explicit = None
    if method == 'numerical' or method == 'numerical_no' or method == 'switching' or method == 'scheduled_sampling':
        P_numerical = np.zeros((n_point, system.n_state))
        P_numerical_n_iters = np.zeros(n_point)
    else:
        P_numerical = None
        P_numerical_n_iters = None

    if method == 'no' or method == 'numerical_no' or method == 'switching' or method == 'scheduled_sampling':
        P_no = np.zeros((n_point, system.n_state))
        if mapie_regressors is not None:
            P_no_ci = np.zeros((n_point, system.n_state, 2))
        else:
            P_no_ci = None
    else:
        P_no = None
        P_no_ci = None
    p_numerical_count = 0
    p_no_count = 0
    Z[n_point_delay, :] = Z0
    inference_time = 0.
    for t_i in range(dataset_config.n_point):
        t_minus_D_i = max(t_i - n_point_delay, 0)
        t = ts[t_i]
        if method == 'explicit':
            U[t_i] = system.U_explicit(t, Z0)
            if t_i > n_point_delay:
                Z[t_i, :] = system.Z_explicit(t, Z0)
        else:
            # estimate system state
            if t_i > n_point_delay:
                Z[t_i, :] = odeint(system.dynamic, Z[t_i - 1, :], [ts[t_i - 1], ts[t_i]], args=(U[t_minus_D_i - 1],))[1]
                Z_t = Z[t_i, :]
            else:
                Z_t = Z0 + dataset_config.noise()
            Z_t += dataset_config.noise()
            # estimate prediction and control signal
            if method == 'numerical':
                begin = time.time()
                solution = dynamic_systems.solve_integral(
                    Z_t=Z_t, P_D=P_numerical[t_minus_D_i:t_i], U_D=U[t_minus_D_i:t_i], t=t,
                    dataset_config=dataset_config)
                P_numerical[t_i, :] = solution.solution
                P_numerical_n_iters[t_i] = solution.n_iter
                end = time.time()
                inference_time += end - begin
                if t_i > n_point_delay:
                    U[t_i] = system.kappa(P_numerical[t_i, :])
            elif method == 'no':
                U_D = pad_leading_zeros(segment=U[t_minus_D_i:t_i], length=n_point_delay)
                begin = time.time()
                P_no[t_i, :] = solve_integral_nn(model=model, U_D=U_D, Z_t=Z_t)
                end = time.time()
                inference_time += end - begin
                if mapie_regressors is not None:
                    P_no_ci[t_i, :, :], _ = solve_integral_ci_nn(mapie_regressors, U_D, Z_t)
                if t_i > n_point_delay:
                    U[t_i] = system.kappa(P_no[t_i, :])
            elif method == 'numerical_no':
                solution = dynamic_systems.solve_integral(
                    Z_t=Z_t, P_D=P_numerical[t_minus_D_i:t_i], U_D=U[t_minus_D_i:t_i], t=t,
                    dataset_config=dataset_config)
                P_numerical[t_i, :] = solution.solution
                P_numerical_n_iters[t_i] = solution.n_iter

                U_D = pad_leading_zeros(segment=U[t_minus_D_i:t_i], length=n_point_delay)
                P_no[t_i, :] = solve_integral_nn(model=model, U_D=U_D, Z_t=Z_t)
                if mapie_regressors is not None:
                    P_no_ci[t_i, :, :], _ = solve_integral_ci_nn(mapie_regressors, U_D, Z_t)
                if t_i > n_point_delay:
                    U[t_i] = system.kappa(P_numerical[t_i, :])
            elif method == 'switching':
                P_numerical[t_i, :] = dynamic_systems.solve_integral(
                    Z_t=Z_t, P_D=P_numerical[t_minus_D_i:t_i], U_D=U[t_minus_D_i:t_i], t=t,
                    dataset_config=dataset_config)
                U_D = pad_leading_zeros(segment=U[t_minus_D_i:t_i], length=n_point_delay)
                P_no[t_i, :] = solve_integral_nn(model=model, U_D=U_D, Z_t=Z_t)
                P_no_ci[t_i, :, :], _ = solve_integral_ci_nn(mapie_regressors, U_D, Z_t)
                if t_i > n_point_delay:
                    if ((Z_t > P_no_ci[t_i - n_point_delay, :, 0]) & (
                            Z_t < P_no_ci[t_i - n_point_delay, :, 1])).all():
                        U[t_i] = system.kappa(P_no[t_i, :])
                        p_no_count += 1
                    else:
                        U[t_i] = system.kappa(P_numerical[t_i, :])
                        p_numerical_count += 1
            elif method == 'scheduled_sampling':
                if np.random.binomial(n=1, p=train_config.scheduled_sampling_p) == 1:
                    # Teacher Forcing
                    solution = dynamic_systems.solve_integral(
                        Z_t=Z_t, P_D=P_no[t_minus_D_i:t_i], U_D=U[t_minus_D_i:t_i], t=t, dataset_config=dataset_config)
                    P_numerical[t_i, :] = solution.solution
                    P_numerical_n_iters[t_i] = solution.n_iter
                else:
                    # Not Teacher Forcing
                    P_no[t_i, :] = solve_integral_nn(
                        model=model, U_D=pad_leading_zeros(segment=U[t_minus_D_i:t_i], length=n_point_delay), Z_t=Z_t)
                if t_i > n_point_delay:
                    U[t_i] = system.kappa(P_no[t_i, :])
            else:
                raise NotImplementedError()
    ts = dataset_config.ts
    delay = dataset_config.delay
    n_point_delay = dataset_config.n_point_delay

    if img_save_path is not None:
        comparison_full = f'{img_save_path}/comparison_full.png'
        comparison_zoom = f'{img_save_path}/comparison_zoom.png'
        difference_full = f'{img_save_path}/difference_full.png'
        difference_zoom = f'{img_save_path}/difference_zoom.png'
        uncertainty_zoom = f'{img_save_path}/uncertainty_zoom.png'
        uncertainty_full = f'{img_save_path}/uncertainty_full.png'
        u_path = f'{img_save_path}/u.png'
        plot_comparison(ts, P_no, P_numerical, P_explicit, Z, delay, n_point_delay, comparison_full, n_state)
        plot_comparison(ts, P_no, P_numerical, P_explicit, Z, delay, n_point_delay, comparison_zoom, n_state, [-5, 5])
        plot_difference(ts, P_no, P_numerical, P_explicit, Z, delay, n_point_delay, difference_full, n_state)
        plot_difference(ts, P_no, P_numerical, P_explicit, Z, delay, n_point_delay, difference_zoom, n_state, [-5, 5])
        plot_single(ts, U, '$U(t)$', u_path)
        if P_no_ci is not None:
            plot_uncertainty(ts, P_no, P_no_ci, Z, delay, n_point_delay, uncertainty_full, n_state)
            plot_uncertainty(ts, P_no, P_no_ci, Z, delay, n_point_delay, uncertainty_zoom, n_state, [-5, 5])

    return SimulationResult(U=U, Z=Z, P_explicit=P_explicit, P_no=P_no, P_no_ci=P_no_ci, P_numerical=P_numerical,
                            runtime=inference_time, P_numerical_n_iters=P_numerical_n_iters)


def model_train(model, optimizer, scheduler, device, training_dataloader, predict_and_loss,
                adversarial_epsilon: float = 0., n_state: int = None):
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
            adversarial_labels = dynamic_systems.solve_integral_successive_batched(
                Z_t=np.array(adversarial_inputs[:, 1:1 + n_state].detach().cpu().numpy()),
                U_D=np.array(adversarial_inputs[:, 1 + n_state:].detach().cpu().numpy()),
                dt=dataset_config.dt, n_state=dataset_config.system.n_state, n_points=dataset_config.n_point_delay,
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


def model_validate(model, device, validating_dataloader):
    with torch.no_grad():
        validating_loss = 0.0
        for inputs, labels in validating_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, loss = predict_and_loss(inputs, labels, model)
            validating_loss += loss.item()
        return validating_loss / len(validating_dataloader)


def run_offline_training(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig):
    device = train_config.device
    adversarial_epsilon = train_config.adversarial_epsilon
    n_epoch = train_config.n_epoch
    img_save_path = model_config.base_path

    model, model_loaded = load_model(train_config, model_config, dataset_config)
    optimizer = load_optimizer(model.parameters(), train_config)
    scheduler = load_lr_scheduler(optimizer, train_config)

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
                model, optimizer, scheduler, device, training_dataloader, predict_and_loss, adversarial_epsilon,
                dataset_config.n_state)
            training_loss_arr.append(training_loss_t)
            desc = f'Epoch [{epoch + 1}/{n_epoch}] || Lr: {lr:6f} || Training loss: {training_loss_t:.6f}'
            if do_validation:
                validating_loss_t = model_validate(model, device, validating_dataloader)
                validating_loss_arr.append(validating_loss_t)
                desc += f' || Validation loss: {validating_loss_t:.6f}'
            if do_testing:
                testing_loss_t = model_validate(model, device, testing_dataloader)
                testing_loss_arr.append(testing_loss_t)
                desc += f' || Test loss: {testing_loss_t:.6f}'
            if do_adversarial:
                adversarial_loss_arr.append(adversarial_loss_t)
                desc += f' || Adversarial loss: {adversarial_loss_t:.6f}'
            bar.set_description(desc)

            if (train_config.log_step > 0 and epoch % train_config.log_step == 0) or epoch == n_epoch - 1:
                rl2, l2, _, n_success = run_test(
                    model, dataset_config, method='no', base_path=model_config.base_path, silence=True)
                rl2_list.append(rl2)
                l2_list.append(l2)
        draw()
        torch.save(model.state_dict(), f'{train_config.model_save_path}/{model_config.model_name}.pth')
    print('Finished Training')
    return model


def run_active_training(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig):
    device = train_config.device
    n_epoch = train_config.n_epoch
    model, model_loaded = load_model(train_config, model_config, dataset_config)
    optimizer = load_optimizer(model.parameters(), train_config)
    scheduler = load_lr_scheduler(optimizer, train_config)
    mapie_regressors = MapieRegressors(model, n_output=dataset_config.n_state, device=device, alpha=train_config.alpha)

    training_dataloader, validating_dataloader = load_training_and_validation_datasets(dataset_config, train_config)
    bar = tqdm(list(range(train_config.n_epoch)))
    # train
    if not model_loaded:
        print('Start training')
        for epoch in bar:
            training_loss_t, _, _ = model_train(model, optimizer, scheduler, device, training_dataloader,
                                                predict_and_loss)
            bar.set_description(f'Epoch [{epoch + 1}/{n_epoch}] || Training loss: {training_loss_t:.6f}')
    # calibrate
    X, y = zip(*[[batch[0].numpy(), batch[1].numpy()] for batch in validating_dataloader])
    X, y = np.vstack(X), np.vstack(y)
    mapie_regressors.fit(X, y)
    while True:
        samples = []
        print('Collecting incremental data')
        while len(samples) < dataset_config.n_sample:
            Z0 = np.random.uniform(low=dataset_config.ic_lower_bound, high=dataset_config.ic_upper_bound,
                                   size=(dataset_config.n_state,))
            result = simulation(dataset_config, Z0, 'switching', model,
                                mapie_regressors=mapie_regressors,
                                img_save_path='./misc/test3', uncertainty_out=True)
            predictions = shift(result.P_no, dataset_config.n_point_delay)
            conf_intervals = shift(result.P_no_ci, dataset_config.n_point_delay)
            true_values = result.Z[dataset_config.n_point_delay:]
            conf_intervals.sort(axis=-1)
            out_of_interval = (true_values < conf_intervals[:, :, 0]) | (true_values > conf_intervals[:, :, 1])
            out_of_interval_indices = np.where(out_of_interval.all(axis=1))[0]
            for t_i, (p, z, t) in enumerate(
                    zip(list(predictions), list(true_values), list(dataset_config.ts[dataset_config.n_point_delay:]))):
                if t_i not in out_of_interval_indices:
                    continue

                u = result.U[t_i: t_i + dataset_config.n_point_delay]
                P = dynamic_systems.solve_integral_successive(
                    Z_t=z, n_points=dataset_config.n_point_delay, n_state=dataset_config.n_state, dt=dataset_config.dt,
                    U_D=u, f=dataset_config.system.dynamic,
                    n_iterations=dataset_config.successive_approximation_n_iteration, adaptive=False)
                samples.append((torch.from_numpy(np.concatenate([t.reshape(-1), z, u])), torch.from_numpy(P)))
            print(f'{len(samples)}/{dataset_config.n_sample}')
        print(f'Incremental training use {len(samples)} samples')
        train_dataset, validate_dataset = split_dataset(samples, train_config.training_ratio)
        training_dataloader = DataLoader(PredictionDataset(train_dataset), batch_size=train_config.batch_size,
                                         shuffle=False)
        validating_dataloader = DataLoader(PredictionDataset(validate_dataset), batch_size=train_config.batch_size,
                                           shuffle=False)
        bar1 = tqdm(list(range(train_config.n_epoch)))
        for epoch in bar1:
            training_loss_t, _, _ = model_train(model, optimizer, scheduler, device, training_dataloader,
                                                predict_and_loss)
            bar1.set_description(f'Epoch [{epoch + 1}/{n_epoch}] || Training loss: {training_loss_t:.6f}')
        result_no = run_test(m=model, dataset_config=dataset_config, base_path=model_config.base_path, method='no')
        print_result(result_no, dataset_config)
        X, y = zip(*[[batch[0].numpy(), batch[1].numpy()] for batch in validating_dataloader])
        X, y = np.vstack(X), np.vstack(y)
        mapie_regressors.fit(X, y)
    return model


def run_quantile_training(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig):
    device = train_config.device
    n_epoch = train_config.n_epoch
    model, model_loaded = load_model(train_config, model_config, dataset_config)
    no_optimizer = load_optimizer(model.parameters(), train_config)
    no_scheduler = load_lr_scheduler(no_optimizer, train_config)
    training_dataloader, validating_dataloader = load_training_and_validation_datasets(dataset_config, train_config)
    # train prediction nn
    bar = tqdm(list(range(train_config.n_epoch)))
    print('Start training NO')
    if not model_loaded:
        for epoch in bar:
            training_loss_t, _, _ = model_train(model, no_optimizer, no_scheduler, device, training_dataloader,
                                                predict_and_loss)
            bar.set_description(f'NO Epoch [{epoch + 1}/{n_epoch}] || Training loss: {training_loss_t:.6f}')
    # train quantile nn
    print('Start training quantile NN')
    quantile_model = QuantileRegressionModel(
        dataset_config.n_state + dataset_config.n_point_delay,
        dataset_config.n_state, train_config.alpha).to(train_config.device)
    bar = tqdm(list(range(train_config.n_epoch)))
    for epoch in bar:
        training_loss_t, _, _ = model_train(quantile_model, no_optimizer, no_scheduler, device, training_dataloader,
                                            quantile_predict_and_loss)
        bar.set_description(f'Quantile NN Epoch [{epoch + 1}/{n_epoch}] || Training loss: {training_loss_t:.6f}')
    return model


def run_scheduled_sampling_training(dataset_config: DatasetConfig, model_config: ModelConfig,
                                    train_config: TrainConfig):
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
    for epoch in range(train_config.n_epoch):
        train_config.set_scheduled_sampling_p(epoch)
        scheduled_sampling_p_arr.append(train_config.scheduled_sampling_p)

        Z0 = np.random.uniform(low=dataset_config.ic_lower_bound, high=dataset_config.ic_upper_bound,
                               size=(dataset_config.n_state,))
        result = simulation(dataset_config, Z0, 'scheduled_sampling', model, img_save_path=img_save_path)
        predictions = shift(result.P_no, dataset_config.n_point_delay)
        true_values = result.Z[dataset_config.n_point_delay:]

        predictions_array = np.array(predictions)
        true_values_array = np.array(true_values)
        timestamps_array = np.array(dataset_config.ts[dataset_config.n_point_delay:])
        U_array = np.array([result.U[t_i: t_i + dataset_config.n_point_delay] for t_i in range(len(timestamps_array))])

        if np.isnan(predictions_array).any() or np.isnan(true_values_array).any() or np.isnan(
                timestamps_array).any() or np.isnan(U_array).any() or np.isinf(predictions_array).any() or np.isinf(
            true_values_array).any() or np.isinf(timestamps_array).any() or np.isinf(U_array).any():
            training_loss_arr.append(0)
            continue
        if dataset_config.integral_method == 'successive':
            P_batched = dynamic_systems.solve_integral_successive_batched(
                Z_t=true_values_array, n_points=dataset_config.n_point_delay, n_state=dataset_config.n_state,
                dt=dataset_config.dt, U_D=U_array, f=dataset_config.system.dynamic,
                n_iterations=dataset_config.successive_approximation_n_iteration, adaptive=False)
        else:
            P_batched, n_iter = dynamic_systems.solve_integral_successive_batched(
                Z_t=true_values_array, n_points=dataset_config.n_point_delay, n_state=dataset_config.n_state,
                dt=dataset_config.dt, U_D=U_array, f=dataset_config.system.dynamic,
                threshold=dataset_config.successive_approximation_threshold, adaptive=True)

        samples = []
        for t_i, (p, z, t) in enumerate(zip(predictions_array, true_values_array, timestamps_array)):
            u = U_array[t_i]
            P = P_batched[t_i]
            samples.append((sample_to_tensor(z, u, t.reshape(-1)), torch.from_numpy(P)))

        dataloader = DataLoader(PredictionDataset(samples), batch_size=train_config.batch_size, shuffle=False)

        training_loss_t, _, _ = model_train(model, optimizer, scheduler, device, dataloader, predict_and_loss)
        if epoch % 10 == 0:
            desc = f'Epoch [{epoch + 1}/{n_epoch}] ' \
                   f'|| Scheduled Sampling Rate {train_config.scheduled_sampling_p} ' \
                   f'|| Training loss: {training_loss_t:.6f}'
            print(desc)
        # print('end training', get_time_str())
        wandb.log({
            'sampling rate': train_config.scheduled_sampling_p,
            'training loss': training_loss_t,
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


def run_test(m, dataset_config: DatasetConfig, method: str, base_path: str = None, silence: bool = False,
             test_points: List = None, plot: bool = False):
    base_path = f'{base_path}/{method}'
    if test_points is None:
        test_points = dataset_config.test_points
    bar = test_points if silence else tqdm(test_points)
    rl2_list = []
    l2_list = []
    runtime_list = []
    n_iter_list = []
    for test_point in bar:
        if not silence:
            bar.set_description(f'Solving system with initial point {np.round(test_point, decimals=2)}.')

        img_save_path = f'{base_path}/{np.round(test_point, decimals=2)}'
        check_dir(img_save_path)
        result = simulation(dataset_config=dataset_config, model=m, Z0=test_point, method=method,
                            img_save_path=img_save_path)
        plt.close()
        n_point_delay = dataset_config.n_point_delay
        if method == 'no':
            P = result.P_no
        elif method == 'numerical':
            P = result.P_numerical
        elif method == 'numerical_no':
            P = result.P_no
        else:
            raise NotImplementedError()
        rl2, l2 = metric(P[n_point_delay:-n_point_delay], result.Z[2 * n_point_delay:])

        if np.isinf(rl2) or np.isnan(rl2):
            if not silence:
                print(f'[WARNING] Running with initial condition Z = {test_point} with method [{method}] failed.')
            continue
        np.savetxt(f'{img_save_path}/metric.txt', np.array([rl2, l2, result.runtime]))
        rl2_list.append(rl2)
        l2_list.append(l2)
        runtime_list.append(result.runtime)
        n_iter_list.append(result.P_numerical_n_iters)
    rl2 = np.nanmean(rl2_list).item()
    l2 = np.nanmean(l2_list).item()
    runtime = np.nanmean(runtime_list).item()
    to_save = [rl2, l2, runtime]
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
    return rl2, l2, runtime, len(rl2_list)


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
        if dataset_config.data_generation_strategy == 'trajectory':
            training_samples = create_trajectory_dataset(dataset_config)
        elif dataset_config.data_generation_strategy == 'random':
            training_samples = create_random_dataset(dataset_config)
        elif dataset_config.data_generation_strategy == 'nn':
            training_samples = create_nn_dataset(dataset_config)
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

    for i, (feature, label) in enumerate(training_samples[:dataset_config.n_plot_sample]):
        plot_sample(feature, label, dataset_config, f'{str(i)}.png')
    draw_distribution(training_samples, dataset_config.n_state, dataset_config.dataset_base_path)
    training_dataloader, validating_dataloader = prepare_datasets(
        training_samples, train_config.training_ratio, train_config.batch_size, train_config.device)
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
    if not train_config.do_test:
        return None
    if not os.path.exists(dataset_config.testing_dataset_file) or dataset_config.recreate_testing_dataset:
        print('Creating testing dataset')
        testing_samples = create_trajectory_dataset(dataset_config, initial_conditions=dataset_config.test_points)
    else:
        print('Loading testing dataset')
        testing_samples = torch.load(dataset_config.testing_dataset_file)
    testing_dataloader = DataLoader(PredictionDataset(testing_samples), batch_size=train_config.batch_size,
                                    shuffle=False)
    print(f'#Testing sample: {len(testing_samples)}')
    return testing_dataloader


def create_trajectory_dataset(dataset_config: DatasetConfig, initial_conditions: List = None):
    all_samples = []
    if dataset_config.z_u_p_pair:
        print('creating datasets of Z(t), U(t-D~t), P(t) pairs')
    else:
        print('creating datasets of Z(t), U(t-D~t), Z(t+D) pairs')
    if initial_conditions is None:
        bar = tqdm(list(np.random.uniform(dataset_config.ic_lower_bound, dataset_config.ic_upper_bound,
                                          (dataset_config.n_dataset, dataset_config.n_state))))
    else:
        bar = initial_conditions
    for i, Z0 in enumerate(bar):
        img_save_path = f'{dataset_config.dataset_base_path}/example/{str(i)}'
        check_dir(img_save_path)
        if dataset_config.z_u_p_pair:
            result = simulation(method='numerical', Z0=Z0, dataset_config=dataset_config, img_save_path=img_save_path)
            dataset = ZUPDataset(
                torch.tensor(result.Z, dtype=torch.float32), torch.tensor(result.U, dtype=torch.float32),
                torch.tensor(result.P_numerical, dtype=torch.float32), dataset_config.n_point_delay, dataset_config.dt)
        else:
            result = simulation(method='explicit', Z0=Z0, dataset_config=dataset_config, img_save_path=img_save_path)
            dataset = ZUZDataset(
                torch.tensor(result.Z, dtype=torch.float32), torch.tensor(result.U, dtype=torch.float32),
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

    def get_random_z_p(batch_size):
        scale = max(abs(dataset_config.ic_lower_bound), abs(dataset_config.ic_upper_bound))
        z = torch.randn(size=(batch_size, 2)) * scale
        p = torch.randn(size=(batch_size, 2)) * scale
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
        scheduler = load_lr_scheduler(optimizer, dataset_config)
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
    if train_config.training_type == 'offline':
        model = run_offline_training(dataset_config=dataset_config, model_config=model_config,
                                     train_config=train_config)
    elif train_config.training_type == 'scheduled sampling':
        model = run_scheduled_sampling_training(dataset_config=dataset_config, model_config=model_config,
                                                train_config=train_config)
    elif train_config.training_type == 'switching':
        model = run_quantile_training(dataset_config=dataset_config, model_config=model_config,
                                      train_config=train_config)
    else:
        raise NotImplementedError()
    return (
        run_test(m=model, dataset_config=dataset_config, base_path=model_config.base_path, method='no'),
        run_test(m=model, dataset_config=dataset_config, base_path=model_config.base_path, method='numerical'),
        run_test(m=model, dataset_config=dataset_config, base_path=model_config.base_path, method='numerical_no')
    )


if __name__ == '__main__':
    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    wandb.init(
        project="no",
        name=get_time_str()
    )
    set_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, default='s5')
    parser.add_argument('-n', type=int, default=None)
    parser.add_argument('-fl', type=int, default=None)
    args = parser.parse_args()
    dataset_config, model_config, train_config = config.get_config(args.s, args.n)
    assert torch.cuda.is_available()
    # train_config.training_type = 'offline'
    train_config.training_type = 'scheduled sampling'
    print_args(dataset_config)
    print_args(model_config)
    print_args(train_config)
    result_no, result_numerical, result_numerical_no = main(dataset_config, model_config, train_config)
    print('NO Result')
    print_result(result_no, dataset_config)
    print('Numerical Result')
    print_result(result_numerical, dataset_config)
    print('Numerical-NO Result')
    print_result(result_numerical_no, dataset_config)
