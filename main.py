import argparse
import time
import uuid
import warnings
from typing import Literal, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from scipy.stats import norm
from tqdm import tqdm

import config
from config import DatasetConfig, ModelConfig, TrainConfig
from dataset import sample_to_tensor
from dynamic_systems import solve_integral_nn, DynamicSystem, solve_integral
from plot_utils import plot_result, difference
from utils import pad_zeros, l2_p_phat, check_dir, predict_and_loss, load_lr_scheduler, set_everything, print_result, \
    load_model, load_optimizer, print_args, get_time_str, SimulationResult, TestResult, count_params, l2_p_z

warnings.filterwarnings('ignore')


def simulation(dataset_config: DatasetConfig, train_config: TrainConfig, Z0,
               method: Literal[
                   'explicit', 'numerical', 'no', 'numerical_no', 'switching', 'scheduled_sampling', 'baseline'] = None,
               model=None, img_save_path: str = None, silence: bool = True, metric_list: List = None):
    # metric_list may contain l2_p_z, rl2_p_z, l2_p_phat, rl2_p_phat
    if metric_list is None:
        metric_list = ['l2_p_z', 'rl2_p_z']
    system: DynamicSystem = dataset_config.system
    ts = dataset_config.ts
    dt = dataset_config.dt
    Z0 = np.array(Z0)
    n_point = dataset_config.n_point
    n_point_start = dataset_config.n_point_start()
    max_n_point_delay = dataset_config.max_n_point_delay()
    alpha = train_config.uq_alpha
    alpha_t = train_config.uq_alpha
    U = np.zeros((n_point, system.n_input))
    Z = np.zeros((n_point, system.n_state))
    P_numerical_n_iters = np.zeros(n_point)
    subsystem_history = np.zeros(n_point)
    P_explicit = np.zeros((n_point, system.n_state))
    P_numerical = np.zeros((n_point, system.n_state))
    P_baseline = np.zeros((n_point, system.n_state))
    P_no = np.zeros((n_point, system.n_state))
    P_no_ci = np.zeros((n_point, system.n_state, 2))
    P_no_Ri = np.zeros(n_point)
    alpha_ts = np.zeros(n_point)
    q_ts = np.zeros(n_point)
    e_ts = np.zeros(n_point)
    P_switching = np.zeros((n_point, system.n_state))
    # qe = QuantileEstimator()
    # gqe = GaussianQuantileEstimator()

    p_numerical_count = 0
    p_no_count = 0
    Z[:n_point_start + 1, :] = Z0
    runtime = 0.
    if silence:
        bar = range(dataset_config.n_point)
    else:
        bar = tqdm(range(dataset_config.n_point))
    for t_i in bar:
        t = ts[t_i]
        if t_i < n_point_start:
            t_i_delayed = 0
        else:
            t_i_delayed = t_i - dataset_config.n_point_delay(t)

        Z_t = Z[t_i, :] + dataset_config.noise()

        # set the ground truth (or not for fast evaluation)
        if method != 'numerical' and 'l2_p_z' in metric_list or 'rl2_p_z' in metric_list:
            solution = solve_integral(Z_t=Z_t, P_D=P_numerical[t_i_delayed:t_i], U_D=U[t_i_delayed:t_i], t=t,
                                      dataset_config=dataset_config, delay=dataset_config.delay)
            P_numerical[t_i, :] = solution.solution
            P_numerical_n_iters[t_i] = solution.n_iter

        if method == 'numerical':
            begin = time.time()
            solution = solve_integral(Z_t=Z_t, P_D=P_numerical[t_i_delayed:t_i], U_D=U[t_i_delayed:t_i], t=t,
                                      dataset_config=dataset_config, delay=dataset_config.delay)
            P_numerical[t_i, :] = solution.solution
            P_numerical_n_iters[t_i] = solution.n_iter
            end = time.time()
            runtime += end - begin
            U[t_i] = system.kappa(P_numerical[t_i, :], t)
        elif method == 'baseline':
            P_baseline[t_i, :] = Z_t
            U[t_i] = system.kappa(P_baseline[t_i, :], t)
        elif method == 'no':
            U_D = pad_zeros(segment=U[t_i_delayed:t_i], length=max_n_point_delay)
            begin = time.time()
            P_no[t_i, :] = solve_integral_nn(model=model, U_D=U_D, Z_t=Z_t, t=t)
            end = time.time()
            runtime += end - begin
            U[t_i] = system.kappa(P_no[t_i, :], t)
        elif method == 'numerical_no':
            begin = time.time()
            solution = solve_integral(
                Z_t=Z_t, P_D=P_numerical[t_i_delayed:t_i], U_D=U[t_i_delayed:t_i], t=t,
                dataset_config=dataset_config, delay=dataset_config.delay)
            P_numerical[t_i, :] = solution.solution
            P_numerical_n_iters[t_i] = solution.n_iter

            U_D = pad_zeros(segment=U[t_i_delayed:t_i], length=n_point_start)
            P_no[t_i, :] = solve_integral_nn(model=model, U_D=U_D, Z_t=Z_t, t=t)
            end = time.time()
            runtime += end - begin
            U[t_i] = system.kappa(P_numerical[t_i, :], t)
        elif method == 'switching':
            U_D = pad_zeros(segment=U[t_i_delayed:t_i], length=max_n_point_delay)
            begin = time.time()
            P_no[t_i, :] = solve_integral_nn(model=model, U_D=U_D, Z_t=Z_t, t=t)
            # actuate the controller
            start_point = n_point_start
            # start_point = 0
            if t_i >= start_point:
                # System switching
                # (1) Get the uncertainty of no model
                P_no_Ri[t_i] = np.linalg.norm(P_no[t_i - start_point, :] - Z_t)

                if train_config.uq_adaptive:
                    quantile = (1 - min(1, max(alpha_t, 0))) * 100
                else:
                    quantile = (1 - alpha) * 100

                Ris = P_no_Ri[start_point:t_i + 1]
                if train_config.uq_type == 'conformal prediction':
                    Q = np.percentile(Ris, quantile)
                    # qe.update(P_no_Ri[t_i])
                    # Q = qe.query(quantile / 100)
                elif train_config.uq_type == 'gaussian process':
                    Q = norm.ppf(quantile / 100, loc=np.mean(Ris), scale=np.std(Ris) + 1e-7)
                    # gqe.update(P_no_Ri[t_i])
                    # Q = gqe.query(quantile / 100)
                else:
                    raise NotImplementedError()

                e_t = 0 if P_no_Ri[t_i] <= Q else 1
                # (2) Assign the indicator
                if train_config.uq_switching_type == 'switching':
                    if e_t == 0:
                        # switch to no scheme if possible
                        if subsystem_history[t_i - 1] == 1:
                            a = np.where(subsystem_history[:t_i] == 0)[0]
                            if len(a) == 0:
                                t_last_no = -1
                            else:
                                t_last_no = a[-1]

                            if t_i - t_last_no > n_point_start:
                                # switch back
                                subsystem_history[t_i] = 0
                            else:
                                # cannot switch
                                subsystem_history[t_i] = 1
                        else:
                            # keeping using no
                            subsystem_history[t_i] = 0
                    else:
                        # use to numerical scheme
                        subsystem_history[t_i] = 1
                elif train_config.uq_switching_type == 'alternating':
                    # alternating between no and numerical arbitrarily
                    # if indicator is 1, then use numerical scheme, otherwise no scheme
                    subsystem_history[t_i] = e_t

                # (3) Select the sub-controller
                if subsystem_history[t_i] == 0:
                    P_switching[t_i, :] = P_no[t_i, :]
                    p_no_count += 1
                else:
                    P_numerical[t_i, :] = solve_integral(
                        Z_t=Z_t, P_D=P_numerical[t_i_delayed:t_i], U_D=U[t_i_delayed:t_i], t=t,
                        dataset_config=dataset_config, delay=dataset_config.delay).solution
                    P_switching[t_i, :] = P_numerical[t_i, :]
                    p_numerical_count += 1
                alpha_t += train_config.uq_gamma * (train_config.uq_alpha - e_t)
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
                subsystem_history[t_i] = 1
            end = time.time()
            runtime += end - begin

            U[t_i] = system.kappa(P_switching[t_i, :], t)
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
                P_no[t_i, :] = solve_integral_nn(model=model, U_D=pad_zeros(
                    segment=U[t_i_delayed:t_i], length=max_n_point_delay), Z_t=Z_t, t=t)
            U[t_i] = system.kappa(P_no[t_i, :], t)
        else:
            raise NotImplementedError()

        if t_i < n_point_start:
            U[t_i] = 0

        if n_point_start <= t_i < len(Z) - 1:
            f_t = system.dynamic(Z[t_i], ts[t_i], U[t_i_delayed])
            Z[t_i + 1] = Z[t_i] + dt * f_t

    plot_result(dataset_config, img_save_path, P_no, P_numerical, P_explicit, P_switching, Z, U, method)

    D_explicit = difference(Z, P_explicit, n_point_start, dataset_config.n_point_delay, ts)
    D_no = difference(Z, P_no, n_point_start, dataset_config.n_point_delay, ts)
    D_numerical = difference(Z, P_numerical, n_point_start, dataset_config.n_point_delay, ts)
    D_switching = difference(Z, P_switching, n_point_start, dataset_config.n_point_delay, ts)
    if method == 'no':
        P = P_no
    elif method == 'numerical':
        P = P_numerical
    elif method == 'numerical_no':
        P = P_no
    elif method == 'switching':
        P = P_switching
    else:
        raise NotImplementedError()
    if 'l2_p_z' in metric_list:
        l2_p_z_value, rl2_p_z_value = l2_p_z(P, Z, dataset_config.n_point_delay, ts)
    else:
        l2_p_z_value, rl2_p_z_value = None, None

    if 'l2_p_phat' in metric_list:
        l2_p_phat_value, rl2_p_phat_value = l2_p_phat(P, P_numerical, dataset_config.n_point_start())
    else:
        l2_p_phat_value, rl2_p_phat_value = None, None
    success = not (np.any(np.isnan(Z)) or np.any(np.isinf(Z)))
    return SimulationResult(
        U=U, Z=Z, D_explicit=D_explicit, D_no=D_no, D_numerical=D_numerical, D_switching=D_switching,
        P_explicit=P_explicit, P_no=P_no, P_no_ci=P_no_ci, P_numerical=P_numerical,
        P_switching=P_switching, runtime=runtime, P_numerical_n_iters=P_numerical_n_iters,
        p_numerical_count=p_numerical_count, p_no_count=p_no_count, P_no_Ri=P_no_Ri, alpha_ts=alpha_ts, q_ts=q_ts,
        e_ts=e_ts, switching_indicator=subsystem_history, avg_prediction_time=runtime / n_point,
        l2_p_z=l2_p_z_value, rl2_p_z=rl2_p_z_value, l2_p_phat=l2_p_phat_value,
        rl2_p_phat=rl2_p_phat_value, success=success,
        n_parameter=count_params(model) if model is not None else 'N/A')


def result_to_samples(result: SimulationResult, dataset_config):
    max_n_point_delay = dataset_config.max_n_point_delay()
    n_point_start = dataset_config.n_point_start()
    n_point_delay = dataset_config.n_point_delay
    # Us = [pad_zeros(result.U[idx + n_point_start - n_point_delay(t): idx + n_point_start], max_n_point_delay,
    #                 leading=True) for idx, t in enumerate(horizon)]

    # Zs = np.array(result.Z[n_point_start:])

    horizon = np.array(dataset_config.ts[n_point_start:-n_point_start])
    Zs = np.array(result.Z[n_point_start:-n_point_start])
    # Ps = np.zeros_like(result.P_numerical[n_point_start:-n_point_start])
    Zs_prediction = np.zeros_like(result.Z[n_point_start:-n_point_start])
    # Us = np.zeros_like(result.U[n_point_start:])
    Us = []
    for ti, t in enumerate(horizon):
        ti_n_point_start = ti + n_point_start
        Zs[ti] = result.Z[ti_n_point_start]
        Zs_prediction[ti] = result.Z[n_point_delay(t) + ti_n_point_start]
        Us.append(pad_zeros(result.U[ti_n_point_start - n_point_delay(t): ti_n_point_start], max_n_point_delay,
                            leading=True))

    samples = []
    for z_pred, z, t, u in zip(Zs_prediction, Zs, horizon, Us):
        samples.append((sample_to_tensor(z, u, t.reshape(-1)), torch.from_numpy(z_pred)))
    return samples


def create_simulation_result(dataset_config: DatasetConfig, train_config: TrainConfig, n_dataset: int = None,
                             test_points=None):
    assert not (n_dataset is None and test_points is None)
    results = []
    state = np.random.RandomState()
    if test_points is None:
        test_points = []
        for dataset_idx in range(n_dataset):
            Z0 = state.uniform(low=dataset_config.ic_lower_bound, high=dataset_config.ic_upper_bound,
                               size=(dataset_config.n_state,))
            test_points.append(Z0)
    else:
        n_dataset = len(test_points)

    for dataset_idx, Z0 in enumerate(test_points):
        result = simulation(dataset_config, train_config, Z0, 'numerical')
        results.append(result)
        wandb.log(
            {
                'dataset progress': dataset_idx / n_dataset
            }
        )

    return results


def run_training(model_config: ModelConfig, train_config: TrainConfig, training_dataset, validation_dataset, model):
    device = train_config.device
    batch_size = train_config.batch_size
    img_save_path = model_config.base_path
    print(f'Train all parameters in {model.__class__.__name__}')
    optimizer = load_optimizer(model.parameters(), train_config)
    scheduler = load_lr_scheduler(optimizer, train_config)
    check_dir(img_save_path)
    model.train()
    print('Begin Training...')
    print(f'Training size: {len(training_dataset)}, Validating size: {len(validation_dataset)}')
    for epoch in range(train_config.n_epoch):
        np.random.shuffle(training_dataset)
        # seq index; time index; input-output pair; data
        n_epoch = 0
        training_loss = 0.0
        for dataset_idx in range(0, len(training_dataset), batch_size):
            sequences = training_dataset[dataset_idx:dataset_idx + batch_size]
            current_batch_size = len(sequences)
            losses = []
            for batch in tqdm(list(zip(*sequences))):
                optimizer.zero_grad()
                inputs, labels = torch.vstack([batch[i][0] for i in range(current_batch_size)]), torch.vstack(
                    [batch[i][1] for i in range(current_batch_size)])
                inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
                _, loss = predict_and_loss(inputs, labels, model)
                losses.append(loss.detach())
                loss.backward()
                optimizer.step()

            training_loss += (sum(losses) / len(losses)).item()

            n_epoch += 1
        training_loss /= n_epoch
        with torch.no_grad():
            n_epoch = 0
            validating_loss = 0.0
            for dataset_idx in range(0, len(validation_dataset), batch_size):
                sequences = validation_dataset[dataset_idx:dataset_idx + batch_size]
                current_batch_size = len(sequences)
                losses = []
                for batch in tqdm(list(zip(*sequences))):
                    inputs, labels = torch.vstack([batch[i][0] for i in range(current_batch_size)]), torch.vstack(
                        [batch[i][1] for i in range(current_batch_size)])
                    inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
                    _, loss = predict_and_loss(inputs, labels, model)
                    losses.append(loss.detach())
                validating_loss += (sum(losses) / len(losses)).item()
                n_epoch += 1
            validating_loss /= n_epoch
        scheduler.step()
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], train_config.scheduler_min_lr)

        wandb.log({
            f'training loss': training_loss,
            f'validating loss': validating_loss,
            f'learning rate': optimizer.param_groups[0]['lr'],
            f'epoch': epoch
        })

    return model


def run_test(m, dataset_config: DatasetConfig, train_config: TrainConfig, method: str, base_path: str = None,
             silence: bool = False, test_points: List = None):
    if base_path is not None:
        base_path = f'{base_path}/{method}'
    assert test_points is not None

    bar = test_points if silence else tqdm(test_points)
    l2_list = []
    prediction_time = []
    n_iter_list = []
    results = []
    no_pred_ratio = []
    for i, test_point in enumerate(bar):
        if isinstance(test_point, tuple) and len(test_point) == 2:
            test_point, name = test_point
        else:
            name = None
        if not silence:
            bar.set_description(f'Solving system with initial point {np.round(test_point, decimals=3)}.')
            bar.set_description(f'Save to {name}.')
        if base_path is not None:
            img_save_path = f'{base_path}/{name}'
            check_dir(img_save_path)
        else:
            img_save_path = None
        result = simulation(dataset_config=dataset_config, train_config=train_config, model=m, Z0=test_point,
                            method=method, img_save_path=img_save_path)
        results.append(result)
        if i == 0 and img_save_path is not None:
            wandb.log({f'{method}-comparison': wandb.Image(f"{img_save_path}/{method}_comp_fit.png")})
            wandb.log({f'{method}-difference': wandb.Image(f"{img_save_path}/{method}_diff_fit.png")})
            wandb.log({f'{method}-u': wandb.Image(f"{img_save_path}/{method}_u.png")})
        plt.close()

        if method == 'switching':
            # plot_switch_system(train_config, dataset_config, result, n_point_start, img_save_path)
            no_pred_ratio.append(result.p_no_count / (result.p_no_count + result.p_numerical_count))

        l2_list.append(result.l2_p_z)
        prediction_time.append(result.avg_prediction_time)
        n_iter_list.append(result.P_numerical_n_iters)

        if not result.success:
            if not silence:
                print(f'[WARNING] Running with initial condition Z = {test_point} with method [{method}] failed.')
            continue
    # l2 = np.nanmean(l2_list).item()
    l2 = np.mean(l2_list).item()
    runtime = np.nanmean(prediction_time).item()
    if method == 'numerical':
        n_iter = np.concatenate(n_iter_list).mean()
        print(f'Numerical method uses {n_iter} iterations on average.')

    return TestResult(runtime=runtime, l2=l2, success_cases=len(l2_list), results=results, no_pred_ratio=no_pred_ratio)


def run_tests(model, train_config, dataset_config, model_config, test_points, only_no_out: bool = False):
    if only_no_out:
        return {
            'no': run_test(m=model, dataset_config=dataset_config, train_config=train_config,
                           base_path=model_config.base_path, test_points=test_points, method='no')
        }
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


def load_dataset(dataset_config, train_config, test_points, run):
    if dataset_config.recreate_dataset:
        print('Begin generating dataset...')
        training_results = create_simulation_result(
            dataset_config, train_config, n_dataset=dataset_config.n_training_dataset)
        validation_results = create_simulation_result(
            dataset_config, train_config, n_dataset=dataset_config.n_validation_dataset)
        print(f'{len(training_results)} generated')
        try:
            training_results_, validation_results_ = dataset_config.load_dataset(run, resize=False)
            print(f'{len(training_results_)} loaded')
            training_results += training_results_
            validation_results += validation_results_
        except Exception:
            print(f'Running results of system {dataset_config.system} do not exist. Create for the first time')

        print(f'{len(training_results)} saved')
        dataset_config.save_dataset(run, training_results, validation_results)
        print(f'Dataset created and saved')
    else:
        training_results, validation_results = dataset_config.load_dataset(run)
        print(f'Dataset loaded')

    # the sample to visualize
    validation_results = create_simulation_result(
        dataset_config, train_config, test_points=test_points)

    training_dataset = []
    for result in training_results:
        training_dataset.append(result_to_samples(result, dataset_config))
    validation_dataset = []
    for result in validation_results:
        validation_dataset.append(result_to_samples(result, dataset_config))

    return training_dataset, validation_dataset


def main(dataset_config: DatasetConfig, model_config: ModelConfig, train_config: TrainConfig, run,
         only_no_out: bool = True, save_model: bool = True, training_dataset=None, validation_dataset=None,
         test_points=None):
    assert train_config.training_type == 'sequence'
    set_everything(0)
    if test_points is None:
        test_points = dataset_config.test_points
    else:
        print('Test points loaded', test_points)

    test_point_pairs = [(tp, uuid.uuid4()) for tp in test_points]
    print('All test points:')
    print(test_point_pairs)

    if training_dataset is None or validation_dataset is None:
        training_dataset, validation_dataset = load_dataset(dataset_config, train_config, test_points, run)
        print('Load dataset in this run')
    else:
        print('Dataset already set, skip loading dataset')

    model = load_model(train_config, model_config, dataset_config)
    wandb.log({'n params': count_params(model)})

    run_training(model_config=model_config, train_config=train_config, training_dataset=training_dataset,
                 validation_dataset=validation_dataset, model=model)
    if save_model:
        model_config.save_model(run, model)
    test_results = run_tests(model, train_config, dataset_config, model_config, test_point_pairs, only_no_out)
    wandb.log({'l2': test_results['no'].l2})
    wandb.log({'l2': test_results['no'].runtime})
    return test_results, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, default='s9')
    parser.add_argument('-delay', type=float, default=None)
    parser.add_argument('-training_type', type=str, default='sequence')
    parser.add_argument('-model_name', type=str, default='FNO-GRU')
    parser.add_argument('-tlb', type=float, default=0.)
    parser.add_argument('-tub', type=float, default=1.)
    parser.add_argument('-cp_gamma', type=float, default=0.01)
    parser.add_argument('-cp_alpha', type=float, default=0.1)

    args = parser.parse_args()
    dataset_config_, model_config_, train_config_ = config.get_config(system_=args.s, delay=args.delay,
                                                                      model_name=args.model_name)
    assert torch.cuda.is_available()
    train_config_.training_type = args.training_type

    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    run = wandb.init(
        project="no",
        name=f'{train_config_.system} {model_config_.model_name}'
             f' {dataset_config_.delay.__class__.__name__} {get_time_str()}',
        config={
            'system': train_config_.system,
            'model': model_config_.model_name
        }
    )
    print_args(dataset_config_)
    print_args(model_config_)
    print_args(train_config_)

    results_, model = main(dataset_config_, model_config_, train_config_, run)
    for method_, result_ in results_.items():
        print(method_)
        print_result(result_, dataset_config_)
        try:
            speedup = results_["numerical"].runtime / result_.runtime
            print(f'Speedup w.r.t numerical: {speedup :.3f}; $\\times {speedup:.3f}$')
        except:
            ...
    wandb.finish()
