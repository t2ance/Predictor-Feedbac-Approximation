import numpy as np
from matplotlib import pyplot as plt

import config
from main import simulation
from plot_utils import plot_comparison, plot_difference, plot_control, set_size, fig_width, plot_switched_control, \
    plot_q, plot_quantile
from utils import set_everything, get_time_str, check_dir


def interval(min_, max_):
    interval = max_ - min_
    expanded_interval = interval * 1.1

    new_min = min_ - (expanded_interval - interval) / 2
    new_max = max_ + (expanded_interval - interval) / 2

    # return max(new_min, -10), min(new_max, 10)
    return new_min, new_max


def plot_base(plot_name, dataset_config, system, Ps, Zs, Ds, Us, labels, captions, results, plot_alpha: bool = False):
    if system == 's8':
        Ps = [P[:, 4:5] for P in Ps]
        Zs = [Z[:, 4:5] for Z in Zs]
        Ds = [D[:, 4:5] for D in Ds]
        n_row = 4
    elif system == 's9':
        n_row = 3
    else:
        raise NotImplementedError()

    n_col = len(labels)
    ts = dataset_config.ts
    delay = dataset_config.delay
    n_point_delay = dataset_config.n_point_delay
    fig = plt.figure(figsize=set_size(width=fig_width, subplots=(n_row, n_col), fraction=1.4))
    subfigs = fig.subfigures(nrows=1, ncols=n_col)
    method_axes = []

    for subfig, caption in zip(subfigs, captions):
        method_axes.append(subfig.subplots(nrows=n_row, ncols=1, gridspec_kw={'hspace': 0.5}))
        subfig.suptitle(caption)

    P_mins, P_maxs = [], []
    for P in Ps:
        P_mins.append(P.min())
        P_maxs.append(P.max())
    min_p, max_p = interval(min(*P_mins), max(*P_maxs))

    D_mins, D_maxs = [], []
    for D in Ds:
        D_mins.append(D.min())
        D_maxs.append(D.max())
    min_d, max_d = interval(min(*D_mins), max(*D_maxs))

    U_mins, U_maxs = [], []
    for U in Us:
        U_mins.append(U.min())
        U_maxs.append(U.max())
    min_u, max_u = interval(min(*U_mins), max(*U_maxs))

    Z_mins, Z_maxs = [], []
    for Z in Zs:
        Z_mins.append(Z.min())
        Z_maxs.append(Z.max())
    min_z, max_z = interval(min(*Z_mins), max(*Z_maxs))

    for i, (axes, P, Z) in enumerate(zip(method_axes, Ps, Zs)):
        comment = i == n_col - 1
        plot_comparison(ts, [P], Z, delay, n_point_delay, None, ylim=[min(min_p, min_z), max(max_p, max_z)],
                        ax=axes[0], comment=comment)

    for i, (axes, P, Z, D) in enumerate(zip(method_axes, Ps, Zs, Ds)):
        comment = i == n_col - 1
        plot_difference(ts, [P], Z, n_point_delay, None, ylim=[min_d, max_d], ax=axes[1], comment=comment,
                        differences=[D], xlim=[0, dataset_config.duration])

    for i, (axes, P, Z, D, U, result, label) in enumerate(zip(method_axes, Ps, Zs, Ds, Us, results, labels)):
        comment = i == n_col - 1
        if 'CP' in label or 'GP' in label or 'alpha' in label:
            plot_switched_control(ts, result, n_point_delay(0), ax=axes[2], comment=comment, ylim=[min_u, max_u])
        else:
            if 'Successive' in label and plot_alpha:
                linestyle = ':'
            else:
                linestyle = '-'
            plot_control(ts, U, None, n_point_delay, ax=axes[2], comment=comment, ylim=[min_u, max_u],
                         linestyle=linestyle)

    if n_row == 4:
        if not plot_alpha:
            q_des = np.array([dataset_config.system.q_des(t) for t in ts])
            n_point_start = n_point_delay(0)
            for i, (axes, P, Z, D, U) in enumerate(zip(method_axes, Ps, Zs, Ds, Us)):
                comment = i == n_col - 1
                q = q_des - Z[:, :dataset_config.n_state // 2]
                if system == 's8':
                    q = q[:, 4:5]
                    q_des = q_des[:, 4:5]
                elif system == 's9':
                    ...
                else:
                    raise NotImplementedError()

                q = q[n_point_start:]
                plot_q(ts[n_point_start:], [q], q_des[n_point_start:], None, ax=axes[3], comment=comment)
        else:
            n_point_start = n_point_delay(0)
            for i, (result, axes, alpha) in enumerate(zip(results[1:], method_axes[1:], alphas)):
                comment = i == n_col - 2
                plot_quantile(n_point_start, result.P_no_Ri, alpha, axes[3], ylim=[0, 100],
                              comment=comment, legend_loc='upper right')
    check_dir(f'./misc/plots')
    plt.savefig(f'./misc/plots/{plot_name}.png')
    plt.savefig(f'./misc/plots/{plot_name}.pdf')
    try:
        wandb.save(f'./misc/plots/{plot_name}.png')
        wandb.save(f'./misc/plots/{plot_name}.pdf')
        wandb.log({f'comparison {plot_name}': wandb.Image(f"./misc/plots/{plot_name}.png")})
    except:
        print('Logging figures to wandb server failed.')
    results_dict = {k: v for k, v in zip(labels, results)}
    return results_dict


def plot_comparisons(test_point, plot_name, dataset_config, train_config, system, fno=None, deeponet=None, gru=None,
                     lstm=None, fno_gru=None, fno_lstm=None, deeponet_gru=None, deeponet_lstm=None, fno_cp=None,
                     deeponet_cp=None, gru_cp=None, lstm_cp=None, fno_gru_cp=None, fno_lstm_cp=None,
                     deeponet_gru_cp=None, deeponet_lstm_cp=None, fno_gp=None, deeponet_gp=None, gru_gp=None,
                     lstm_gp=None, fno_gru_gp=None, fno_lstm_gp=None, deeponet_gru_gp=None, deeponet_lstm_gp=None,
                     metric_list=None):
    def simulate_ml_methods(model, model_name):
        if model is None:
            print(f'Model {model_name} excluded')
            return
        if model_name.endswith(r'$_{CP}$'):
            prediction_method = 'switching'
            train_config.uq_type = 'conformal prediction'
        elif model_name.endswith(r'$_{GP}$'):
            prediction_method = 'switching'
            train_config.uq_type = 'gaussian process'
        else:
            prediction_method = 'no'
        m_result = simulation(dataset_config=dataset_config, train_config=train_config, Z0=test_point, model=model,
                              method=prediction_method, silence=False, metric_list=metric_list)
        Ps.append(m_result.P_no)
        Zs.append(m_result.Z)
        Ds.append(m_result.D_no)
        Us.append(m_result.U)
        labels.append(model_name)
        results.append(m_result)
        if not m_result.success:
            print(f'{model_name} failed')

    Ps = []
    Zs = []
    Ds = []
    Us = []
    labels = []
    results = []
    print(f'Begin simulation {plot_name}, with initial point {test_point}')
    result = simulation(dataset_config=dataset_config, train_config=train_config, Z0=test_point, method='numerical',
                        silence=False, metric_list=metric_list)
    Ps.append(result.P_numerical)
    Zs.append(result.Z)
    Ds.append(result.D_numerical)
    Us.append(result.U)
    labels.append('Successive \n Approximation')
    results.append(result)
    print('Numerical approximation iteration', result.P_numerical_n_iters.mean())

    simulate_ml_methods(fno, model_name='FNO')
    simulate_ml_methods(deeponet, model_name='DeepONet')
    simulate_ml_methods(gru, model_name='GRU')
    simulate_ml_methods(lstm, model_name='LSTM')
    simulate_ml_methods(fno_gru, model_name='FNO-GRU')
    simulate_ml_methods(fno_lstm, model_name='FNO-LSTM')
    simulate_ml_methods(deeponet_gru, model_name='DeepONet-GRU')
    simulate_ml_methods(deeponet_lstm, model_name='DeepONet-LSTM')

    simulate_ml_methods(fno_cp, model_name=r'FNO$_{CP}$')
    simulate_ml_methods(deeponet_cp, model_name=r'DeepONet$_{CP}$')
    simulate_ml_methods(gru_cp, model_name=r'GRU$_{CP}$')
    simulate_ml_methods(lstm_cp, model_name=r'LSTM$_{CP}$')
    simulate_ml_methods(fno_gru_cp, model_name=r'FNO-GRU$_{CP}$')
    simulate_ml_methods(fno_lstm_cp, model_name=r'FNO-LSTM$_{CP}$')
    simulate_ml_methods(deeponet_gru_cp, model_name=r'DeepONet-GRU$_{CP}$')
    simulate_ml_methods(deeponet_lstm_cp, model_name=r'DeepONet-LSTM$_{CP}$')

    simulate_ml_methods(fno_gp, model_name=r'FNO$_{GP}$')
    simulate_ml_methods(deeponet_gp, model_name=r'DeepONet$_{GP}$')
    simulate_ml_methods(gru_gp, model_name=r'GRU$_{GP}$')
    simulate_ml_methods(lstm_gp, model_name=r'LSTM$_{GP}$')
    simulate_ml_methods(fno_gru_gp, model_name=r'FNO-GRU$_{GP}$')
    simulate_ml_methods(fno_lstm_gp, model_name=r'FNO-LSTM$_{GP}$')
    simulate_ml_methods(deeponet_gru_gp, model_name=r'DeepONet-GRU$_{GP}$')
    simulate_ml_methods(deeponet_lstm_gp, model_name=r'DeepONet-LSTM$_{GP}$')
    captions = []
    for label, result in zip(labels, results):
        caption = label
        captions.append(caption)
    print(f'End simulation {plot_name}')
    print(labels)

    return plot_base(plot_name, dataset_config, system, Ps, Zs, Ds, Us, labels, captions, results)


def plot_alpha(test_point, plot_name, dataset_config, train_config, model, alphas, system, metric_list):
    Ps = []
    Zs = []
    Ds = []
    Us = []
    labels = []
    results = []
    print('Begin simulation')

    result = simulation(dataset_config=dataset_config, train_config=train_config, Z0=test_point, method='numerical',
                        silence=False, metric_list=metric_list)
    Ps.append(result.P_numerical)
    Zs.append(result.Z)
    Ds.append(result.D_numerical)
    Us.append(result.U)
    labels.append('Successive \n Approximation')
    results.append(result)
    print('Numerical approximation iteration', result.P_numerical_n_iters.mean())

    for alpha in alphas:
        train_config.uq_alpha = alpha
        m_result = simulation(dataset_config=dataset_config, train_config=train_config, Z0=test_point, model=model,
                              method='switching', silence=False, metric_list=metric_list)
        Ps.append(m_result.P_no)
        Zs.append(m_result.Z)
        Ds.append(m_result.D_no)
        Us.append(m_result.U)
        labels.append(rf'$\alpha = {alpha}$')
        results.append(m_result)

    print('End simulation')

    return plot_base(plot_name, dataset_config, system, Ps, Zs, Ds, Us, labels, labels, results, plot_alpha=True)


if __name__ == '__main__':
    set_everything(0)
    import wandb
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, default='s9')
    parser.add_argument('-n', type=int, default=5)
    parser.add_argument('-m', type=str, default='cp-ood')
    args = parser.parse_args()

    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    run = wandb.init(
        project="no",
        name=f'experiment {args.s} {args.n} {args.m} {get_time_str()}'
    )

    metric_list = ['l2_p_z', 'rl2_p_z', 'l2_p_phat', 'rl2_p_phat']

    dataset_config, model_config, train_config = config.get_config(system_=args.s, model_name='FNO')
    fno, n_params = model_config.get_model(run, train_config, dataset_config, 'latest')
    fno_cp, fno_gp = fno, fno

    dataset_config, model_config, train_config = config.get_config(system_=args.s, model_name='DeepONet')
    deeponet, n_params = model_config.get_model(run, train_config, dataset_config, 'latest')
    deeponet_cp, deeponet_gp = deeponet, deeponet

    dataset_config, model_config, train_config = config.get_config(system_=args.s, model_name='GRU')
    gru, n_params = model_config.get_model(run, train_config, dataset_config, 'latest')
    gru_cp, gru_gp = gru, gru

    dataset_config, model_config, train_config = config.get_config(system_=args.s, model_name='LSTM')
    lstm, n_params = model_config.get_model(run, train_config, dataset_config, 'latest')
    lstm_cp, lstm_gp = lstm, lstm

    dataset_config, model_config, train_config = config.get_config(system_=args.s, model_name='FNO-GRU')
    train_config.zero_init = False
    fno_gru, n_params = model_config.get_model(run, train_config, dataset_config, 'best')
    fno_gru_cp, fno_gru_gp = fno_gru, fno_gru

    dataset_config, model_config, train_config = config.get_config(system_=args.s, model_name='FNO-LSTM')
    train_config.zero_init = False
    fno_lstm, n_params = model_config.get_model(run, train_config, dataset_config, 'best')
    fno_lstm_cp, fno_lstm_gp = fno_lstm, fno_lstm

    dataset_config, model_config, train_config = config.get_config(system_=args.s, model_name='DeepONet-GRU')
    train_config.zero_init = False
    deeponet_gru, n_params = model_config.get_model(run, train_config, dataset_config, 'best')
    deeponet_gru_cp, deeponet_gru_gp = deeponet_gru, deeponet_gru

    dataset_config, model_config, train_config = config.get_config(system_=args.s, model_name='DeepONet-LSTM')
    train_config.zero_init = False
    deeponet_lstm, n_params = model_config.get_model(run, train_config, dataset_config, 'best')
    deeponet_lstm_cp, deeponet_lstm_gp = deeponet_lstm, deeponet_lstm

    if args.m == 'table':
        fno_cp = None
        deeponet_cp = None
        gru_cp = None
        lstm_cp = None
        fno_gru_cp = None
        fno_lstm_cp = None
        deeponet_gru_cp = None
        deeponet_lstm_cp = None

        fno_gp = None
        deeponet_gp = None
        gru_gp = None
        lstm_gp = None
        fno_gru_gp = None
        fno_lstm_gp = None
        deeponet_gru_gp = None
        deeponet_lstm_gp = None
    elif args.m == 'figure':
        metric_list = ['l2_p_z', 'rl2_p_z']
        if args.s == 's8':
            # fno = None
            deeponet = None
            gru = None
            # lstm = None
            fno_gru = None
            # fno_lstm = None
            deeponet_gru = None
            deeponet_lstm = None

            fno_cp = None
            deeponet_cp = None
            gru_cp = None
            lstm_cp = None
            fno_gru_cp = None
            fno_lstm_cp = None
            deeponet_gru_cp = None
            deeponet_lstm_cp = None

            fno_gp = None
            deeponet_gp = None
            gru_gp = None
            lstm_gp = None
            fno_gru_gp = None
            fno_lstm_gp = None
            deeponet_gru_gp = None
            deeponet_lstm_gp = None
        elif args.s == 's9':
            fno = None
            # deeponet = None
            # gru = None
            lstm = None
            fno_gru = None
            fno_lstm = None
            # deeponet_gru = None
            deeponet_lstm = None

            fno_cp = None
            deeponet_cp = None
            gru_cp = None
            lstm_cp = None
            fno_gru_cp = None
            fno_lstm_cp = None
            deeponet_gru_cp = None
            deeponet_lstm_cp = None

            fno_gp = None
            deeponet_gp = None
            gru_gp = None
            lstm_gp = None
            fno_gru_gp = None
            fno_lstm_gp = None
            deeponet_gru_gp = None
            deeponet_lstm_gp = None
        else:
            raise NotImplementedError()
    elif args.m == 'cp-ood':
        if args.s == 's8':
            fno = None
            deeponet = None
            gru = None
            lstm = None
            fno_gru = None
            # fno_lstm = None
            deeponet_gru = None
            deeponet_lstm = None

            fno_cp = None
            deeponet_cp = None
            gru_cp = None
            lstm_cp = None
            fno_gru_cp = None
            # fno_lstm_cp = None
            deeponet_gru_cp = None
            deeponet_lstm_cp = None

            fno_gp = None
            deeponet_gp = None
            gru_gp = None
            lstm_gp = None
            fno_gru_gp = None
            # fno_lstm_gp = None
            deeponet_gru_gp = None
            deeponet_lstm_gp = None

            dataset_config.random_test_lower_bound = -2
            dataset_config.random_test_upper_bound = 2
            train_config.uq_gamma = 0.01
            train_config.uq_alpha = 0.1
        elif args.s == 's9':
            fno = None
            deeponet = None
            gru = None
            lstm = None
            fno_gru = None
            fno_lstm = None
            # deeponet_gru = None
            deeponet_lstm = None

            fno_cp = None
            deeponet_cp = None
            gru_cp = None
            lstm_cp = None
            fno_gru_cp = None
            fno_lstm_cp = None
            # deeponet_gru_cp = None
            deeponet_lstm_cp = None

            fno_gp = None
            deeponet_gp = None
            gru_gp = None
            lstm_gp = None
            fno_gru_gp = None
            fno_lstm_gp = None
            # deeponet_gru_gp = None
            deeponet_lstm_gp = None

            # dataset_config.random_test_lower_bound = -2
            # dataset_config.random_test_upper_bound = 2
            dataset_config.random_test_lower_bound = -5
            dataset_config.random_test_upper_bound = 5
            train_config.uq_gamma = 0.01
            train_config.uq_alpha = 0.1
        else:
            raise NotImplementedError()
    elif args.m == 'alpha':
        if args.s == 's8':
            model = fno_lstm
            dataset_config.random_test_lower_bound = -2
            dataset_config.random_test_upper_bound = 2
            train_config.uq_gamma = 0.01
            alphas = [0.02, 0.1, 0.4]
            # metric_list = ['l2_p_z', 'rl2_p_z']
        elif args.s == 's9':
            model = deeponet_gru
            dataset_config.random_test_lower_bound = -2
            dataset_config.random_test_upper_bound = 2
            train_config.uq_gamma = 0.01
            alphas = [0.02, 0.1, 0.4]
            # metric_list = ['l2_p_z', 'rl2_p_z']
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    results = None

    test_points = dataset_config.get_test_points(n_point=args.n)
    for i, test_point in enumerate(test_points):
        plot_name = f'{args.s}-{args.m}-{i}'
        if args.m == 'alpha':
            print(alphas, dataset_config.random_test_lower_bound, dataset_config.random_test_upper_bound)
            result_dict = plot_alpha(test_point, plot_name, dataset_config, train_config, model=model, alphas=alphas,
                                     system=args.s, metric_list=metric_list)
        else:
            result_dict = plot_comparisons(
                test_point, plot_name, dataset_config, train_config, system=args.s, metric_list=metric_list,
                fno=fno,
                deeponet=deeponet,
                gru=gru,
                lstm=lstm,
                fno_gru=fno_gru,
                fno_lstm=fno_lstm,
                deeponet_gru=deeponet_gru,
                deeponet_lstm=deeponet_lstm,
                fno_cp=fno_cp,
                deeponet_cp=deeponet_cp,
                gru_cp=gru_cp,
                lstm_cp=lstm_cp,
                fno_gru_cp=fno_gru_cp,
                fno_lstm_cp=fno_lstm_cp,
                deeponet_gru_cp=deeponet_gru_cp,
                deeponet_lstm_cp=deeponet_lstm_cp,
                fno_gp=fno_gp,
                deeponet_gp=deeponet_gp,
                gru_gp=gru_gp,
                lstm_gp=lstm_gp,
                fno_gru_gp=fno_gru_gp,
                fno_lstm_gp=fno_lstm_gp,
                deeponet_gru_gp=deeponet_gru_gp,
                deeponet_lstm_gp=deeponet_lstm_gp,
            )
        if results is None:
            results = {k: [] for k in result_dict.keys()}

        for k, v in result_dict.items():
            results[k].append(v)

    result_list_num = results['Successive \n Approximation']
    avg_prediction_time_num = sum([r.avg_prediction_time for r in result_list_num]) / len(result_list_num)
    print(
        r'Method& Parameters& \makecell{Raw Prediction Time \\ (ms/prediction)}  &  Speedup & $\mathcal{E}$ & $\mathcal{E}^\prime$ & $\mathcal{E}_r$ & $\mathcal{E}_r^\prime$ \\ \midrule')
    for method, result_list in results.items():
        n_test = len(result_list)
        avg_prediction_time = sum([r.avg_prediction_time for r in result_list]) / n_test
        l2_p_z = sum([r.l2_p_z for r in result_list]) / n_test
        rl2_p_z = sum([r.rl2_p_z for r in result_list]) / n_test
        if 'l2_p_phat' not in metric_list:
            l2_p_phat = np.array(-1)
            rl2_p_phat = np.array(-1)
        else:
            l2_p_phat = sum([r.l2_p_phat for r in result_list]) / n_test
            rl2_p_phat = sum([r.rl2_p_phat for r in result_list]) / n_test
        n_success = sum([1 if r.success else 0 for r in result_list])
        line = f'{method} & {result_list[0].n_parameter} & {avg_prediction_time * 1000:.3f} ' \
               f'& {avg_prediction_time_num / avg_prediction_time:.3f} ' \
               f'& {l2_p_phat.item():.3f} & {l2_p_z.item():.3f} & {rl2_p_phat.item():.3f} & {rl2_p_z.item():.3f}\\\\'
        if method == 'Successive \n Approximation' or method == 'DeepONet' or method == 'LSTM':
            line += r' \midrule'
        if method == 'DeepONet-LSTM':
            line += r' \bottomrule'
        print(line)
