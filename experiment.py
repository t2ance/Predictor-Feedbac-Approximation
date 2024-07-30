import numpy as np
from matplotlib import pyplot as plt

import config
from main import simulation
from plot_utils import plot_comparison, plot_difference, plot_control, set_size, fig_width, plot_switch_segments, plot_q
from utils import set_everything, load_cp_hyperparameters, load_model


def interval(min_, max_):
    interval = max_ - min_
    expanded_interval = interval * 1.1

    new_min = min_ - (expanded_interval - interval) / 2
    new_max = max_ + (expanded_interval - interval) / 2

    # return max(new_min, -10), min(new_max, 10)
    return new_min, new_max


def plot(test_point, plot_name):
    ts = dataset_config.ts
    delay = dataset_config.delay
    n_state = dataset_config.n_state
    n_point_delay = dataset_config.n_point_delay
    fig = plt.figure(figsize=set_size(width=fig_width, fraction=1.4, subplots=(3, 3), height_add=0.6))
    subfigs = fig.subfigures(nrows=1, ncols=3)
    numerical_fig, no_fig, switching_fig = subfigs
    numerical_fig.suptitle('Successive \n Approximation')
    no_fig.suptitle('FNO')
    switching_fig.suptitle('FNO$_{ACP}$')
    numerical_axes = numerical_fig.subplots(nrows=3, ncols=1, gridspec_kw={'hspace': 0.5})
    no_axes = no_fig.subplots(nrows=3, ncols=1, gridspec_kw={'hspace': 0.5})
    switching_axes = switching_fig.subplots(nrows=3, ncols=1, gridspec_kw={'hspace': 0.5})

    print('Begin simulation')
    numerical = simulation(dataset_config=dataset_config, train_config=train_config, model=model, Z0=test_point,
                           method='numerical')
    no = simulation(dataset_config=dataset_config, train_config=train_config, model=model, Z0=test_point, method='no')
    switching = simulation(dataset_config=dataset_config, train_config=train_config, model=model, Z0=test_point,
                           method='switching')
    print('End simulation')

    min_p, max_p = interval(min(numerical.P_numerical.min(), no.P_no.min(), switching.P_switching.min()),
                            max(numerical.P_numerical.max(), no.P_no.max(), switching.P_switching.max()))

    plot_comparison(ts, [numerical.P_numerical], numerical.Z, delay, n_point_delay, None, n_state, ylim=[min_p, max_p],
                    ax=numerical_axes[0], comment=False)
    plot_comparison(ts, [no.P_no], no.Z, delay, n_point_delay, None, n_state, ylim=[min_p, max_p], ax=no_axes[0],
                    comment=False)
    plot_comparison(ts, [switching.P_switching], switching.Z, delay, n_point_delay, None, n_state, ylim=[min_p, max_p],
                    ax=switching_axes[0], comment=True)
    min_d, max_d = interval(min(numerical.D_numerical.min(), no.D_no.min(), switching.D_switching.min()),
                            max(numerical.D_numerical.max(), no.D_no.max(), switching.D_switching.max()))

    plot_difference(ts, [numerical.P_numerical], numerical.Z, delay, n_point_delay, None, n_state, ylim=[min_d, max_d],
                    ax=numerical_axes[1], comment=False, differences=[numerical.D_numerical])
    plot_difference(ts, [no.P_no], no.Z, delay, n_point_delay, None, n_state, ylim=[min_d, max_d], ax=no_axes[1],
                    comment=False, differences=[no.D_no])
    plot_difference(ts, [switching.P_switching], switching.Z, delay, n_point_delay, None, n_state, ylim=[min_d, max_d],
                    ax=switching_axes[1], comment=True, differences=[switching.D_switching])

    min_u, max_u = interval(min(numerical.U.min(), no.U.min(), switching.U.min()),
                            max(numerical.U.max(), no.U.max(), switching.U.max()))
    plot_control(ts, numerical.U, None, n_point_delay, ax=numerical_axes[2], comment=False, ylim=[min_u, max_u])
    plot_control(ts, no.U, None, n_point_delay, ax=no_axes[2], comment=False, ylim=[min_u, max_u])
    plot_switch_segments(ts, switching, None, n_point_delay(0), ax=switching_axes[2], comment=True, ylim=[min_u, max_u])

    plt.savefig(f"./misc/plots/{plot_name}.pdf")


def plot_alpha():
    dataset_config, model_config, train_config = config.get_config('s1')
    dataset_config.recreate_training_dataset = False
    dataset_config.duration = 9
    train_config.do_training = False
    train_config.load_model = True
    model, model_loaded = load_model(train_config, model_config, dataset_config)
    ts = dataset_config.ts
    delay = dataset_config.delay
    n_state = dataset_config.n_state
    n_point_delay = dataset_config.n_point_delay
    dataset_config.random_test_lower_bound = 1.55
    dataset_config.random_test_upper_bound = 1.6
    test_point = dataset_config.test_points[0]
    fig = plt.figure(figsize=set_size(width=fig_width, fraction=1.4, subplots=(3, 3), height_add=0.6))
    subfigs = fig.subfigures(nrows=1, ncols=3)
    switching_alpha01_fig, switching_alpha02_fig, switching_alpha05_fig = subfigs
    switching_alpha01_fig.suptitle('$alpha_0 = 0.1$')
    switching_alpha02_fig.suptitle('$alpha_0 = 0.2$')
    switching_alpha05_fig.suptitle('$alpha_0 = 0.5$')
    switching_alpha01_axes = switching_alpha01_fig.subplots(nrows=3, ncols=1, gridspec_kw={'hspace': 0.5})
    switching_alpha02_axes = switching_alpha02_fig.subplots(nrows=3, ncols=1, gridspec_kw={'hspace': 0.5})
    switching_alpha05_axes = switching_alpha05_fig.subplots(nrows=3, ncols=1, gridspec_kw={'hspace': 0.5})

    print('Begin simulation')
    tlb, tub, cp_gamma, cp_alpha, system = load_cp_hyperparameters('toy_alpha_0.1')
    train_config.cp_gamma = cp_gamma
    train_config.cp_alpha = cp_alpha
    dataset_config.random_test_lower_bound = tlb
    dataset_config.random_test_upper_bound = tub
    switching_alpha01 = simulation(dataset_config=dataset_config, train_config=train_config, model=model, Z0=test_point,
                                   method='switching')
    tlb, tub, cp_gamma, cp_alpha, system = load_cp_hyperparameters('toy_alpha_0.2')
    train_config.cp_gamma = cp_gamma
    train_config.cp_alpha = cp_alpha
    dataset_config.random_test_lower_bound = tlb
    dataset_config.random_test_upper_bound = tub
    switching_alpha02 = simulation(dataset_config=dataset_config, train_config=train_config, model=model, Z0=test_point,
                                   method='switching')
    tlb, tub, cp_gamma, cp_alpha, system = load_cp_hyperparameters('toy_alpha_0.5')
    train_config.cp_gamma = cp_gamma
    train_config.cp_alpha = cp_alpha
    dataset_config.random_test_lower_bound = tlb
    dataset_config.random_test_upper_bound = tub
    switching_alpha05 = simulation(dataset_config=dataset_config, train_config=train_config, model=model, Z0=test_point,
                                   method='switching')
    print('End simulation')

    min_p, max_p = interval(
        min(switching_alpha01.P_numerical.min(), switching_alpha02.P_no.min(), switching_alpha05.P_switching.min()),
        max(switching_alpha01.P_numerical.max(), switching_alpha02.P_no.max(), switching_alpha05.P_switching.max()))

    plot_comparison(ts, [switching_alpha01.P_numerical], switching_alpha01.Z, delay, n_point_delay, None, n_state,
                    ylim=[min_p, max_p], ax=switching_alpha01_axes[0], comment=False)
    plot_comparison(ts, [switching_alpha02.P_no], switching_alpha02.Z, delay, n_point_delay, None, n_state,
                    ylim=[min_p, max_p], ax=switching_alpha02_axes[0],
                    comment=False)
    plot_comparison(ts, [switching_alpha05.P_switching], switching_alpha05.Z, delay, n_point_delay, None, n_state,
                    ylim=[min_p, max_p],
                    ax=switching_alpha05_axes[0], comment=True)
    min_d, max_d = interval(
        min(switching_alpha01.D_numerical.min(), switching_alpha02.D_no.min(), switching_alpha05.D_switching.min()),
        max(switching_alpha01.D_numerical.max(), switching_alpha02.D_no.max(), switching_alpha05.D_switching.max()))

    plot_difference(ts, [switching_alpha01.P_numerical], switching_alpha01.Z, delay, n_point_delay, None, n_state,
                    ylim=[min_d, max_d],
                    ax=switching_alpha01_axes[1], comment=False, differences=[switching_alpha01.D_numerical])
    plot_difference(ts, [switching_alpha02.P_no], switching_alpha02.Z, delay, n_point_delay, None, n_state,
                    ylim=[min_d, max_d], ax=switching_alpha02_axes[1],
                    comment=False, differences=[switching_alpha02.D_no])
    plot_difference(ts, [switching_alpha05.P_switching], switching_alpha05.Z, delay, n_point_delay, None, n_state,
                    ylim=[min_d, max_d],
                    ax=switching_alpha05_axes[1], comment=True, differences=[switching_alpha05.D_switching])

    min_u, max_u = interval(min(switching_alpha01.U.min(), switching_alpha02.U.min(), switching_alpha05.U.min()),
                            max(switching_alpha01.U.max(), switching_alpha02.U.max(), switching_alpha05.U.max()))
    plot_switch_segments(ts, switching_alpha01, None, n_point_delay(0), ax=switching_alpha01_axes[2], comment=False,
                         ylim=[min_u, max_u])
    plot_switch_segments(ts, switching_alpha02, None, n_point_delay(0), ax=switching_alpha02_axes[2], comment=False,
                         ylim=[min_u, max_u])
    plot_switch_segments(ts, switching_alpha05, None, n_point_delay(0), ax=switching_alpha05_axes[2], comment=True,
                         ylim=[min_u, max_u])
    plt.savefig(f"./misc/plots/alpha.pdf")


def plot_tracking(test_point, plot_name):
    ts = dataset_config.ts
    delay = dataset_config.delay
    n_state = dataset_config.n_state
    n_point_delay = dataset_config.n_point_delay
    fig = plt.figure(figsize=set_size(width=fig_width, fraction=1.4, subplots=(3, 3), height_add=0.6))
    subfigs = fig.subfigures(nrows=1, ncols=3)
    numerical_fig, no_fig, switching_fig = subfigs
    numerical_fig.suptitle('Successive \n Approximation')
    no_fig.suptitle('FNO')
    switching_fig.suptitle('FNO$_{ACP}$')
    numerical_axes = numerical_fig.subplots(nrows=3, ncols=1, gridspec_kw={'hspace': 0.5})
    no_axes = no_fig.subplots(nrows=3, ncols=1, gridspec_kw={'hspace': 0.5})
    switching_axes = switching_fig.subplots(nrows=3, ncols=1, gridspec_kw={'hspace': 0.5})

    print('Begin simulation')
    numerical = simulation(dataset_config=dataset_config, train_config=train_config, model=model, Z0=test_point,
                           method='numerical')
    no = simulation(dataset_config=dataset_config, train_config=train_config, model=model, Z0=test_point, method='no')
    switching = simulation(dataset_config=dataset_config, train_config=train_config, model=model, Z0=test_point,
                           method='switching')
    print('End simulation')

    min_p, max_p = interval(min(numerical.P_numerical.min(), no.P_no.min(), switching.P_switching.min()),
                            max(numerical.P_numerical.max(), no.P_no.max(), switching.P_switching.max()))

    plot_comparison(ts, [numerical.P_numerical], numerical.Z, delay, n_point_delay, None, n_state, ylim=[min_p, max_p],
                    ax=numerical_axes[0], comment=False)
    plot_comparison(ts, [no.P_no], no.Z, delay, n_point_delay, None, n_state, ylim=[min_p, max_p], ax=no_axes[0],
                    comment=False)
    plot_comparison(ts, [switching.P_switching], switching.Z, delay, n_point_delay, None, n_state, ylim=[min_p, max_p],
                    ax=switching_axes[0], comment=True)
    min_d, max_d = interval(min(numerical.D_numerical.min(), no.D_no.min(), switching.D_switching.min()),
                            max(numerical.D_numerical.max(), no.D_no.max(), switching.D_switching.max()))

    plot_difference(ts, [numerical.P_numerical], numerical.Z, delay, n_point_delay, None, n_state, ylim=[min_d, max_d],
                    ax=numerical_axes[1], comment=False, differences=[numerical.D_numerical])
    plot_difference(ts, [no.P_no], no.Z, delay, n_point_delay, None, n_state, ylim=[min_d, max_d], ax=no_axes[1],
                    comment=False, differences=[no.D_no])
    plot_difference(ts, [switching.P_switching], switching.Z, delay, n_point_delay, None, n_state, ylim=[min_d, max_d],
                    ax=switching_axes[1], comment=True, differences=[switching.D_switching])

    # min_u, max_u = interval(min(numerical.U.min(), no.U.min(), switching.U.min()),
    #                         max(numerical.U.max(), no.U.max(), switching.U.max()))
    # plot_control(ts, numerical.U, None, n_point_delay, ax=numerical_axes[2], comment=False, ylim=[min_u, max_u])
    # plot_control(ts, no.U, None, n_point_delay, ax=no_axes[2], comment=False, ylim=[min_u, max_u])
    # plot_switch_segments(ts, switching, None, n_point_delay(0), ax=switching_axes[2], comment=True, ylim=[min_u, max_u])

    q_des = np.array([dataset_config.system.q_des(t) for t in ts])
    q_numerical = q_des - numerical.Z[:, :2]
    q_no = q_des - no.Z[:, :2]
    q_switching = q_des - switching.Z[:, :2]

    n_point_start = n_point_delay(0)
    q_des = q_des[n_point_start:]
    q_numerical = q_numerical[n_point_start:]
    q_no = q_no[n_point_start:]
    q_switching = q_switching[n_point_start:]
    plot_q(ts, [q_numerical], q_des, None, dataset_config.system.n_input, ax=numerical_axes[2], comment=False)
    plot_q(ts, [q_no], q_des, None, dataset_config.system.n_input, ax=no_axes[2], comment=False)
    plot_q(ts, [q_switching], q_des, None, dataset_config.system.n_input, ax=switching_axes[2], comment=True)

    plt.savefig(f"./misc/plots/{plot_name}.pdf")


if __name__ == '__main__':
    set_everything(0)
    hyperparameters = [
        # 'toy_id', 'toy_ood',
        'baxter_id',
        # 'baxter_ood1', 'baxter_ood2'
    ]
    for hyperparameter in hyperparameters:
        print(f'Running with {hyperparameter}')
        tlb, tub, cp_gamma, cp_alpha, system = load_cp_hyperparameters(hyperparameter)
        dataset_config, model_config, train_config = config.get_config(system)
        dataset_config.recreate_training_dataset = False
        train_config.do_training = False
        train_config.load_model = True
        train_config.cp_gamma = cp_gamma
        train_config.cp_alpha = cp_alpha
        dataset_config.random_test_lower_bound = tlb
        dataset_config.random_test_upper_bound = tub
        model, model_loaded = load_model(train_config, model_config, dataset_config)
        if hyperparameter.startswith('toy'):
            plot(dataset_config.test_points[0], hyperparameter)
        else:
            plot_tracking(dataset_config.test_points[0], hyperparameter)

    plot_alpha()
