import uuid
from typing import List

import numpy as np
from matplotlib import pyplot as plt

import config
from main import simulation, run_test
from plot_utils import plot_comparison, plot_difference, plot_control, set_size, fig_width, plot_switch_segments, \
    plot_q, plot_quantile
from utils import set_everything, load_cp_hyperparameters, load_model, get_time_str, TestResult, check_dir, \
    SimulationResult


def interval(min_, max_):
    interval = max_ - min_
    expanded_interval = interval * 1.1

    new_min = min_ - (expanded_interval - interval) / 2
    new_max = max_ + (expanded_interval - interval) / 2

    # return max(new_min, -10), min(new_max, 10)
    return new_min, new_max


def plot_no_numerical_comparison(test_points, plot_name, dataset_config, train_config, model_config, model, n_row):
    print(f'Begin simulation {plot_name}')
    result_numerical = run_test(dataset_config=dataset_config, train_config=train_config, m=model,
                                test_points=test_points, method='numerical')
    result_no = run_test(dataset_config=dataset_config, train_config=train_config, m=model, test_points=test_points,
                         method='no')
    print_results([result_no, result_numerical], result_numerical)
    print(f'End simulation {plot_name}')

    for i, (test_point, numerical, no) in enumerate(zip(test_points, result_numerical.results, result_no.results)):
        ts = dataset_config.ts
        delay = dataset_config.delay
        n_state = dataset_config.n_state
        n_point_delay = dataset_config.n_point_delay
        fig = plt.figure(figsize=set_size(width=fig_width, fraction=1.4, subplots=(n_row, 2), height_add=0.6))
        subfigs = fig.subfigures(nrows=1, ncols=2)
        numerical_fig, no_fig = subfigs
        numerical_fig.suptitle('Successive \n Approximation')
        no_fig.suptitle(model_config.model_name)
        numerical_axes = numerical_fig.subplots(nrows=n_row, ncols=1, gridspec_kw={'hspace': 0.5})
        no_axes = no_fig.subplots(nrows=n_row, ncols=1, gridspec_kw={'hspace': 0.5})

        min_p, max_p = interval(min(numerical.P_numerical.min(), no.P_no.min()),
                                max(numerical.P_numerical.max(), no.P_no.max()))

        plot_comparison(ts, [numerical.P_numerical], numerical.Z, delay, n_point_delay, None, n_state,
                        ylim=[min_p, max_p],
                        ax=numerical_axes[0], comment=False)
        plot_comparison(ts, [no.P_no], no.Z, delay, n_point_delay, None, n_state, ylim=[min_p, max_p], ax=no_axes[0],
                        comment=False)
        min_d, max_d = interval(min(numerical.D_numerical.min(), no.D_no.min()),
                                max(numerical.D_numerical.max(), no.D_no.max()))

        plot_difference(ts, [numerical.P_numerical], numerical.Z, delay, n_point_delay, None, n_state,
                        ylim=[min_d, max_d],
                        ax=numerical_axes[1], comment=False, differences=[numerical.D_numerical])
        plot_difference(ts, [no.P_no], no.Z, delay, n_point_delay, None, n_state, ylim=[min_d, max_d], ax=no_axes[1],
                        comment=False, differences=[no.D_no])

        min_u, max_u = interval(min(numerical.U.min(), no.U.min()),
                                max(numerical.U.max(), no.U.max()))
        plot_control(ts, numerical.U, None, n_point_delay, ax=numerical_axes[2], comment=False, ylim=[min_u, max_u],
                     linestyle='--')
        plot_control(ts, no.U, None, n_point_delay, ax=no_axes[2], comment=False, ylim=[min_u, max_u])

        if n_row == 4:
            q_des = np.array([dataset_config.system.q_des(t) for t in ts])
            q_numerical = q_des - numerical.Z[:, :2]
            q_no = q_des - no.Z[:, :2]
            n_point_start = n_point_delay(0)
            q_des = q_des[n_point_start:]
            q_numerical = q_numerical[n_point_start:]
            q_no = q_no[n_point_start:]
            plot_q(ts[n_point_start:], [q_numerical], q_des, None, dataset_config.system.n_input, ax=numerical_axes[3],
                   comment=False)
            plot_q(ts[n_point_start:], [q_no], q_des, None, dataset_config.system.n_input, ax=no_axes[3], comment=False)

        check_dir(f'./misc/plots/{plot_name}')
        plt.savefig(f"./misc/plots/{plot_name}/{i}.pdf")


def plot_sw_numerical_comparison(test_points, plot_name, dataset_config, train_config, model_config, model, n_row):
    print(f'Begin simulation {plot_name}')
    result_numerical = run_test(dataset_config=dataset_config, train_config=train_config, m=model,
                                test_points=test_points, method='numerical')
    train_config.uq_type = 'conformal prediction'
    result_cp = run_test(dataset_config=dataset_config, train_config=train_config, m=model, test_points=test_points,
                         method='switching')
    print_results([result_cp, result_numerical], result_numerical)
    print(f'End simulation {plot_name}')

    for i, (test_point, numerical, cp) in enumerate(zip(test_points, result_numerical.results, result_cp.results)):
        ts = dataset_config.ts
        delay = dataset_config.delay
        n_state = dataset_config.n_state
        n_point_delay = dataset_config.n_point_delay
        fig = plt.figure(figsize=set_size(width=fig_width, fraction=1.4, subplots=(n_row, 2), height_add=0.6))
        subfigs = fig.subfigures(nrows=1, ncols=2)
        numerical_fig, cp_fig = subfigs
        numerical_fig.suptitle('Successive \n Approximation')
        cp_fig.suptitle(model_config.model_name + '-CP')
        numerical_axes = numerical_fig.subplots(nrows=n_row, ncols=1, gridspec_kw={'hspace': 0.5})
        cp_axes = cp_fig.subplots(nrows=n_row, ncols=1, gridspec_kw={'hspace': 0.5})

        min_p, max_p = interval(min(numerical.P_numerical.min(), cp.P_switching.min()),
                                max(numerical.P_numerical.max(), cp.P_switching.max()))

        plot_comparison(ts, [numerical.P_numerical], numerical.Z, delay, n_point_delay, None, n_state,
                        ylim=[min_p, max_p],
                        ax=numerical_axes[0], comment=False)
        plot_comparison(ts, [cp.P_switching], cp.Z, delay, n_point_delay, None, n_state, ylim=[min_p, max_p],
                        ax=cp_axes[0], comment=True)
        min_d, max_d = interval(min(numerical.D_numerical.min(), cp.D_switching.min()),
                                max(numerical.D_numerical.max(), cp.D_switching.max()))

        plot_difference(ts, [numerical.P_numerical], numerical.Z, delay, n_point_delay, None, n_state,
                        ylim=[min_d, max_d],
                        ax=numerical_axes[1],
                        comment=False, differences=[numerical.D_numerical])
        plot_difference(ts, [cp.P_switching], cp.Z, delay, n_point_delay, None, n_state, ylim=[min_d, max_d],
                        ax=cp_axes[1], comment=True, differences=[cp.D_switching])

        min_u, max_u = interval(min(numerical.U.min(), cp.U.min()), max(numerical.U.max(), cp.U.max()))
        plot_control(ts, numerical.U, None, n_point_delay, ax=numerical_axes[2], comment=False, ylim=[min_u, max_u])
        plot_switch_segments(ts, cp, n_point_delay(0), ax=cp_axes[2], legend=True, ylim=[min_u, max_u])

        if n_row == 4:
            q_des = np.array([dataset_config.system.q_des(t) for t in ts])
            q_numerical = q_des - numerical.Z[:, :2]
            q_switching = q_des - cp.Z[:, :2]
            n_point_start = n_point_delay(0)
            q_des = q_des[n_point_start:]
            q_numerical = q_numerical[n_point_start:]
            q_switching = q_switching[n_point_start:]
            plot_q(ts[n_point_start:], [q_numerical], q_des, None, dataset_config.system.n_input, ax=numerical_axes[3],
                   comment=False)
            plot_q(ts[n_point_start:], [q_switching], q_des, None, dataset_config.system.n_input, ax=cp_axes[3],
                   comment=True)

        check_dir(f'./misc/plots/{plot_name}')
        plt.savefig(f"./misc/plots/{plot_name}/{i}.pdf")


def plot_uq_ablation(test_points, plot_name, dataset_config, train_config, model_config, model, n_row):
    print(f'Begin simulation {plot_name}')
    result_numerical = run_test(dataset_config=dataset_config, train_config=train_config, m=model,
                                test_points=test_points, method='numerical')
    result_no = run_test(dataset_config=dataset_config, train_config=train_config, m=model, test_points=test_points,
                         method='no')
    train_config.uq_type = 'conformal prediction'
    result_cp = run_test(dataset_config=dataset_config, train_config=train_config, m=model, test_points=test_points,
                         method='switching')
    train_config.uq_type = 'gaussian process'
    result_gp = run_test(dataset_config=dataset_config, train_config=train_config, m=model, test_points=test_points,
                         method='switching')
    print_results([result_no, result_cp, result_gp], result_numerical)
    print(f'End simulation {plot_name}')

    for i, (test_point, no, cp, gp) in enumerate(
            zip(test_points, result_no.results, result_cp.results, result_gp.results)):
        ts = dataset_config.ts
        delay = dataset_config.delay
        n_state = dataset_config.n_state
        n_point_delay = dataset_config.n_point_delay
        fig = plt.figure(figsize=set_size(width=fig_width, fraction=1.4, subplots=(n_row, 3), height_add=0.6))
        subfigs = fig.subfigures(nrows=1, ncols=3)
        no_fig, cp_fig, gp_fig = subfigs
        no_fig.suptitle(model_config.model_name)
        cp_fig.suptitle(model_config.model_name + '-CP')
        gp_fig.suptitle(model_config.model_name + '-GP')
        no_axes = no_fig.subplots(nrows=n_row, ncols=1, gridspec_kw={'hspace': 0.5})
        cp_axes = cp_fig.subplots(nrows=n_row, ncols=1, gridspec_kw={'hspace': 0.5})
        gp_axes = gp_fig.subplots(nrows=n_row, ncols=1, gridspec_kw={'hspace': 0.5})

        min_p, max_p = interval(min(no.P_no.min(), cp.P_switching.min(), gp.P_switching.min()),
                                max(no.P_no.max(), cp.P_switching.max(), gp.P_switching.max()))

        plot_comparison(ts, [no.P_no], no.Z, delay, n_point_delay, None, n_state, ylim=[min_p, max_p], ax=no_axes[0],
                        comment=False)
        plot_comparison(ts, [cp.P_switching], cp.Z, delay, n_point_delay, None, n_state, ylim=[min_p, max_p],
                        ax=gp_axes[0], comment=False)
        plot_comparison(ts, [gp.P_switching], gp.Z, delay, n_point_delay, None, n_state, ylim=[min_p, max_p],
                        ax=cp_axes[0], comment=True)
        min_d, max_d = interval(min(no.D_no.min(), cp.D_switching.min(), gp.D_switching.min()),
                                max(no.D_no.max(), cp.D_switching.max(), gp.D_switching.max()))

        plot_difference(ts, [no.P_no], no.Z, delay, n_point_delay, None, n_state, ylim=[min_d, max_d], ax=no_axes[1],
                        comment=False, differences=[no.D_no])
        plot_difference(ts, [cp.P_switching], cp.Z, delay, n_point_delay, None, n_state, ylim=[min_d, max_d],
                        ax=gp_axes[1], comment=False, differences=[cp.D_switching])
        plot_difference(ts, [gp.P_switching], gp.Z, delay, n_point_delay, None, n_state, ylim=[min_d, max_d],
                        ax=cp_axes[1], comment=True, differences=[gp.D_switching])
        min_u, max_u = interval(min(no.U.min(), cp.U.min(), gp.U.min()),
                                max(no.U.max(), cp.U.max(), gp.U.max()))
        plot_control(ts, no.U, None, n_point_delay, ax=no_axes[2], comment=False, ylim=[min_u, max_u])
        plot_switch_segments(ts, cp, n_point_delay(0), ax=cp_axes[2], legend=False, ylim=[min_u, max_u])
        plot_switch_segments(ts, gp, n_point_delay(0), ax=gp_axes[2], legend=True, ylim=[min_u, max_u])

        if n_row == 4:
            q_des = np.array([dataset_config.system.q_des(t) for t in ts])
            q_no = q_des - no.Z[:, :2]
            q_cp = q_des - cp.Z[:, :2]
            q_gp = q_des - gp.Z[:, :2]
            n_point_start = n_point_delay(0)
            q_des = q_des[n_point_start:]
            q_no = q_no[n_point_start:]
            q_cp = q_cp[n_point_start:]
            q_gp = q_gp[n_point_start:]
            plot_q(ts[n_point_start:], [q_no], q_des, None, dataset_config.system.n_input, ax=no_axes[3], comment=False)
            plot_q(ts[n_point_start:], [q_cp], q_des, None, dataset_config.system.n_input, ax=cp_axes[3], comment=True)
            plot_q(ts[n_point_start:], [q_gp], q_des, None, dataset_config.system.n_input, ax=gp_axes[3], comment=True)

        check_dir(f'./misc/plots/{plot_name}')
        plt.savefig(f"./misc/plots/{plot_name}/{i}.pdf")


def plot_rnn_ablation(test_points, plot_name):
    print(f'Begin simulation {plot_name}')

    dataset_config, model_config, train_config = config.get_config(system_='s5', model_name='FNO')
    model, model_loaded = load_model(train_config, model_config, dataset_config)
    model_config.load_model(run, model)

    result_no = run_test(dataset_config=dataset_config, train_config=train_config, m=model,
                         test_points=test_points, method='no')

    dataset_config, model_config, train_config = config.get_config(system_='s5', model_name='FNO-GRU')
    model, model_loaded = load_model(train_config, model_config, dataset_config)
    model_config.load_model(run, model)
    result_gru = run_test(dataset_config=dataset_config, train_config=train_config, m=model,
                          test_points=test_points, method='no')

    dataset_config, model_config, train_config = config.get_config(system_='s5', model_name='FNO-LSTM')
    model, model_loaded = load_model(train_config, model_config, dataset_config)
    model_config.load_model(run, model)
    result_lstm = run_test(dataset_config=dataset_config, train_config=train_config, m=model,
                           test_points=test_points, method='no')

    result_numerical = run_test(dataset_config=dataset_config, train_config=train_config, m=model,
                                test_points=test_points, method='numerical')
    print_results([result_no, result_gru, result_lstm], result_numerical)

    print(f'End simulation {plot_name}')

    for i, (test_point, no, gru, lstm) in enumerate(
            zip(test_points, result_no.results, result_gru.results, result_lstm.results)):
        ts = dataset_config.ts
        delay = dataset_config.delay
        n_state = dataset_config.n_state
        n_point_delay = dataset_config.n_point_delay
        fig = plt.figure(figsize=set_size(width=fig_width, fraction=1.4, subplots=(4, 3), height_add=0.6))
        subfigs = fig.subfigures(nrows=1, ncols=3)
        no_fig, gru_fig, lstm_fig = subfigs
        no_fig.suptitle('FNO')
        gru_fig.suptitle('FNO-GRU')
        lstm_fig.suptitle('FNO-LSTM')
        no_axes = no_fig.subplots(nrows=4, ncols=1, gridspec_kw={'hspace': 0.5})
        gru_axes = gru_fig.subplots(nrows=4, ncols=1, gridspec_kw={'hspace': 0.5})
        lstm_axes = lstm_fig.subplots(nrows=4, ncols=1, gridspec_kw={'hspace': 0.5})

        min_p, max_p = interval(min(no.P_no.min(), gru.P_no.min(), lstm.P_no.min()),
                                max(no.P_no.max(), gru.P_no.max(), lstm.P_no.max()))

        plot_comparison(ts, [no.P_no], no.Z, delay, n_point_delay, None, n_state, ylim=[min_p, max_p], ax=no_axes[0],
                        comment=False)
        plot_comparison(ts, [gru.P_no], gru.Z, delay, n_point_delay, None, n_state, ylim=[min_p, max_p], ax=gru_axes[0],
                        comment=False)
        plot_comparison(ts, [lstm.P_no], lstm.Z, delay, n_point_delay, None, n_state, ylim=[min_p, max_p],
                        ax=lstm_axes[0],
                        comment=True)
        min_d, max_d = interval(min(no.D_no.min(), gru.D_no.min(), lstm.D_no.min()),
                                max(no.D_no.max(), gru.D_no.max(), lstm.D_no.max()))

        plot_difference(ts, [no.P_no], no.Z, delay, n_point_delay, None, n_state, ylim=[min_d, max_d], ax=no_axes[1],
                        comment=False, differences=[no.D_no])
        plot_difference(ts, [gru.P_no], gru.Z, delay, n_point_delay, None, n_state, ylim=[min_d, max_d], ax=gru_axes[1],
                        comment=False, differences=[gru.D_no])
        plot_difference(ts, [lstm.P_no], lstm.Z, delay, n_point_delay, None, n_state, ylim=[min_d, max_d],
                        ax=lstm_axes[1], comment=True, differences=[lstm.D_no])

        min_u, max_u = interval(min(no.U.min(), gru.U.min(), lstm.U.min()), max(no.U.max(), gru.U.max(), lstm.U.max()))
        plot_control(ts, no.U, None, n_point_delay, ax=no_axes[2], comment=False, ylim=[min_u, max_u])
        plot_control(ts, gru.U, None, n_point_delay, ax=gru_axes[2], comment=False, ylim=[min_u, max_u])
        plot_control(ts, lstm.U, None, n_point_delay, ax=lstm_axes[2], comment=False, ylim=[min_u, max_u])

        q_des = np.array([dataset_config.system.q_des(t) for t in ts])
        q_no = q_des - no.Z[:, :2]
        q_gru = q_des - gru.Z[:, :2]
        q_lstm = q_des - lstm.Z[:, :2]
        n_point_start = n_point_delay(0)
        q_des = q_des[n_point_start:]
        q_no = q_no[n_point_start:]
        q_gru = q_gru[n_point_start:]
        q_lstm = q_lstm[n_point_start:]
        plot_q(ts[n_point_start:], [q_no], q_des, None, dataset_config.system.n_input, ax=no_axes[3], comment=False)
        plot_q(ts[n_point_start:], [q_gru], q_des, None, dataset_config.system.n_input, ax=gru_axes[3], comment=False)
        plot_q(ts[n_point_start:], [q_lstm], q_des, None, dataset_config.system.n_input, ax=lstm_axes[3], comment=True)

        check_dir(f'./misc/plots/{plot_name}')
        plt.savefig(f"./misc/plots/{plot_name}/{i}.pdf")


def plot_alpha(test_points, plot_name, dataset_config, train_config, model, alphas):
    n_col = len(alphas)
    test_results = []
    ts = dataset_config.ts
    delay = dataset_config.delay
    n_point_delay = dataset_config.n_point_delay
    n_state = dataset_config.n_state
    print('Begin simulation')
    for alpha in alphas:
        train_config.uq_alpha = alpha
        switching_result = run_test(dataset_config=dataset_config, train_config=train_config, m=model,
                                    test_points=test_points, method='switching')
        test_results.append(switching_result.results)
    print('End simulation')
    test_results = zip(*test_results)

    for i, (test_point, test_result) in enumerate(
            zip(test_points, test_results)):
        fig = plt.figure(figsize=set_size(width=fig_width, fraction=1.4, subplots=(4, n_col), height_add=0.6))
        subfigs = fig.subfigures(nrows=1, ncols=n_col)
        switching_alpha_axes = []
        for subfig, alpha in zip(subfigs, alphas):
            subfig.suptitle(rf'$\alpha_0 = {alpha}$')

            switching_alpha_axes.append(subfig.subplots(nrows=4, ncols=1, gridspec_kw={'hspace': 0.5}))
        test_result: List[SimulationResult]

        min_p, max_p = interval(
            min([switching_result.P_switching.min() for switching_result in test_result]),
            max([switching_result.P_switching.max() for switching_result in test_result])
        )
        for switching_result, switching_alpha_ax in zip(test_result, switching_alpha_axes):
            plot_comparison(ts, [switching_result.P_switching], switching_result.Z, delay, n_point_delay, None, n_state,
                            ylim=[min_p, max_p], ax=switching_alpha_ax[0], comment=False)

        min_d, max_d = interval(
            min([switching_result.D_switching.min() for switching_result in test_result]),
            max([switching_result.D_switching.max() for switching_result in test_result])
        )
        for switching_result, switching_alpha_ax in zip(test_result, switching_alpha_axes):
            plot_difference(ts, [switching_result.D_switching], switching_result.Z, delay, n_point_delay, None, n_state,
                            ylim=[min_d, max_d],
                            ax=switching_alpha_ax[1], comment=False, differences=[switching_result.D_numerical])

        min_u, max_u = interval(
            min([switching_result.U.min() for switching_result in test_result]),
            max([switching_result.U.max() for switching_result in test_result])
        )
        for switching_result, switching_alpha_ax in zip(test_result, switching_alpha_axes):
            plot_switch_segments(ts, switching_result, n_point_delay(0), ax=switching_alpha_ax[2], legend=False,
                                 ylim=[min_u, max_u])

        n_point_start = n_point_delay(0)
        for switching_result, switching_alpha_ax, alpha in zip(test_result, switching_alpha_axes, alphas):
            plot_switch_segments(ts, switching_result, n_point_delay(0), ax=switching_alpha_ax[2], legend=False,
                                 ylim=[min_u, max_u])
            plot_quantile(n_point_start, switching_result.P_no_Ri, alpha, switching_alpha_ax[3], ylim=[0, 30],
                          comment=False, legend_loc='upper right')

        check_dir(f'./misc/plots/{plot_name}')
        plt.savefig(f"./misc/plots/{plot_name}/{i}.pdf")


def plot_figure(n_test=10):
    # train_config, dataset_config, model_config, model = load_config('s5', 'FNO-GRU', None)
    # test_points = [
    #     (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)) for _
    #     in range(n_test)
    # ]
    # plot_no_numerical_comparison(test_points, 'baxter-id-fno-gru', dataset_config, train_config, model_config,
    #                              model, n_row=4)
    #
    # train_config, dataset_config, model_config, model = load_config('s7', 'FNO-GRU', None)
    # test_points = [
    #     (np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)) for _ in
    #     range(n_test)
    # ]
    # plot_no_numerical_comparison(test_points, 'unicycle-id-fno-gru', dataset_config, train_config, model_config,
    #                              model, n_row=3)
    #
    # train_config, dataset_config, model_config, model = load_config('s5', 'FNO-LSTM', None)
    # test_points = [
    #     (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)) for _
    #     in range(n_test)
    # ]
    # plot_no_numerical_comparison(test_points, 'baxter-id-fno-lstm', dataset_config, train_config, model_config,
    #                              model, n_row=4)
    #
    # train_config, dataset_config, model_config, model = load_config('s5', 'FNO-GRU', cp_alpha=0.1)
    # test_points = [
    #     (np.random.uniform(1, 2), np.random.uniform(1, 2), np.random.uniform(0, 1), np.random.uniform(0, 1)) for _
    #     in range(n_test)
    # ]
    # plot_uq_ablation(test_points, 'baxter-ood-fno-gru', dataset_config, train_config, model_config, model, n_row=4)
    #
    # test_points = [
    #     (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)) for _
    #     in range(n_test)
    # ]
    # plot_rnn_ablation(test_points, 'rnn-ablation')

    train_config, dataset_config, model_config, model = load_config('s5', 'FNO-GRU', None)
    test_points = [
        (np.random.uniform(1, 1.5), np.random.uniform(1, 1.5), np.random.uniform(1, 1.5), np.random.uniform(1, 1.5)) for _
        in range(n_test)
    ]
    print(test_points)
    plot_alpha(test_points, 'alpha-ablation', dataset_config, train_config, model, [0.01, 0.1, 0.5])


def load_config(system, model_name, cp_alpha):
    dataset_config, model_config, train_config = config.get_config(system_=system, model_name=model_name)
    model_config.model_name = model_name
    train_config.uq_alpha = cp_alpha
    model, model_loaded = load_model(train_config, model_config, dataset_config)
    model_config.load_model(run, model)
    return train_config, dataset_config, model_config, model


def print_results(results, result_baseline=None):
    raw_prediction_times = [result.runtime for result in results]
    print('raw prediction time', '&'.join([f'${t * 1000:.2f}$' for t in raw_prediction_times]))
    if result_baseline is not None:
        speedups = [result_baseline.runtime / result.runtime for result in results]
        print('speedup', '&'.join([f'$\\times {t:.3f}$' for t in speedups]))
    l2s = [result.l2 for result in results]
    print('l2 error', '&'.join([f'${t:.3f}$' for t in l2s]))
    success_cases = [result.success_cases for result in results]
    print('success case', success_cases)


if __name__ == '__main__':
    import wandb

    set_everything(0)
    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    run = wandb.init(
        project="no",
        name=f'result-plotting {get_time_str()}'
    )
    plot_figure(n_test=1)
