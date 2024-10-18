from typing import List, Dict

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


def plot_base(plot_name, dataset_config, system, Ps, Zs, Ds, Us, switching_indicators, labels, captions, results,
              plot_alpha: bool = False):
    if system == 's11':
        Ps = [P[:, 4:5] for P in Ps]
        Zs = [Z[:, 4:5] for Z in Zs]
        Ds = [D[:, 4:5] for D in Ds]
        Us = [U[:, 4:5] for U in Us]
        # switching_indicators = [switching_indicator[:, 4:5] for switching_indicator in switching_indicators]
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

    for i, (axes, P, Z, D, U, switching_indicator, result, label) in enumerate(
            zip(method_axes, Ps, Zs, Ds, Us, switching_indicators, results, labels)):
        comment = i == n_col - 1
        if 'CP' in label or 'GP' in label or 'alpha' in label:
            plot_switched_control(ts, U, switching_indicator, n_point_delay(0), ax=axes[2], comment=comment,
                                  ylim=[min_u, max_u])
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
            qs = []
            q_des_s = []
            q_mins = []
            q_maxs = []
            for i, (axes, P, Z, D, U) in enumerate(zip(method_axes, Ps, Zs, Ds, Us)):
                q = q_des - Z[:, :dataset_config.n_state // 2]
                if system == 's11':
                    q = q[:, 4:5]
                    q_des_ = q_des[:, 4:5]
                elif system == 's9':
                    q_des_ = q_des
                else:
                    raise NotImplementedError()

                q = q[n_point_start:]
                qs.append(q)
                q_des_s.append(q_des_)
                q_mins.append(min(q.min(), q_des_.min()))
                q_maxs.append(max(q.max(), q_des_.max()))

            min_q, max_q = interval(min(*q_mins), max(*q_maxs))
            for i, (axes, P, Z, D, U, q, q_des_) in enumerate(zip(method_axes, Ps, Zs, Ds, Us, qs, q_des_s)):
                comment = i == n_col - 1
                plot_q(ts[n_point_start:], [q], q_des_[n_point_start:], None, ax=axes[3], comment=comment,
                       ylim=[min_q, max_q])
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


def plot_comparisons(test_point, plot_name, dataset_config, train_config, system, model_dict=None, metric_list=None):
    def simulate_ml_methods(model, model_name):
        if model is None:
            print(f'Model {model_name} excluded')
            return
        if model_name.endswith(r'$_{CP}$'):
            prediction_method = 'switching'
            train_config.uq_type = 'conformal prediction'
            train_config.uq_non_delay = False
            train_config.uq_warmup = True
        elif model_name.endswith(r'$_{GP}$'):
            prediction_method = 'switching'
            train_config.uq_type = 'gaussian process'
            train_config.uq_non_delay = False
            train_config.uq_warmup = True
        elif model_name.endswith('$_{CP-non-delay}$'):
            prediction_method = 'switching'
            train_config.uq_type = 'conformal prediction'
            train_config.uq_non_delay = True
            train_config.uq_warmup = True
        elif model_name.endswith('$_{GP-non-delay}$'):
            prediction_method = 'switching'
            train_config.uq_type = 'gaussian process'
            train_config.uq_non_delay = True
            train_config.uq_warmup = True
        else:
            prediction_method = 'no'
        print(f'Simulating {model_name}')
        m_result = simulation(dataset_config=dataset_config, train_config=train_config, model_config=model_config,
                              Z0=test_point, model=model, method=prediction_method, silence=False,
                              metric_list=metric_list)
        Ps.append(m_result.P_no)
        Zs.append(m_result.Z)
        Ds.append(m_result.D_no)
        Us.append(m_result.U)
        switching_indicators.append(m_result.switching_indicator)
        labels.append(model_name)
        results.append(m_result)
        if not m_result.success:
            print(f'{model_name} failed')
        else:
            print(f'{model_name} succeeded')
            if prediction_method != 'no':
                print(f'Numerical approximation iteration for {model_name}', result.P_numerical_n_iters.mean())

    Ps = []
    Zs = []
    Ds = []
    Us = []
    switching_indicators = []
    labels = []
    results = []

    print(f'Begin simulation {plot_name}, with initial point {test_point}')
    points = np.round(test_point, decimals=2)
    points = [str(point) for point in points]
    points = ','.join(points)
    print(f'Solving system with initial point [{points}].')
    result = simulation(dataset_config=dataset_config, train_config=train_config, model_config=model_config,
                        Z0=test_point, method='numerical', silence=False, metric_list=metric_list)
    Ps.append(result.P_numerical)
    Zs.append(result.Z)
    Ds.append(result.D_numerical)
    Us.append(result.U)
    switching_indicators.append(result.switching_indicator)
    labels.append('Successive \n Approximation')
    results.append(result)
    print('Numerical approximation iteration', result.P_numerical_n_iters.mean())

    for model_name, model in model_dict.items():
        simulate_ml_methods(model, model_name=model_name)

    print(f'End simulation {plot_name}')
    print(labels)
    return plot_base(plot_name, dataset_config, system, Ps, Zs, Ds, Us, switching_indicators, labels, labels, results)


def plot_alpha(test_point, plot_name, dataset_config, train_config, model, alphas, system, metric_list):
    Ps = []
    Zs = []
    Ds = []
    Us = []
    labels = []
    results = []
    switching_indicators = []
    print('Begin simulation')

    result = simulation(dataset_config=dataset_config, train_config=train_config, model_config=model_config,
                        Z0=test_point, method='numerical',
                        silence=False, metric_list=metric_list)
    Ps.append(result.P_numerical)
    Zs.append(result.Z)
    Ds.append(result.D_numerical)
    Us.append(result.U)
    switching_indicators.append(result.switching_indicator)
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
        switching_indicators.append(m_result.switching_indicator)
        labels.append(rf'$\alpha = {alpha}$')
        results.append(m_result)

    print('End simulation')

    return plot_base(plot_name, dataset_config, system, Ps, Zs, Ds, Us, switching_indicators, labels, labels, results,
                     plot_alpha=True)


def load_model_for_experiments(model_dict: Dict[str, str], system: str):
    to_return = {}
    for model_name, model_version in model_dict.items():
        model_name_ = model_name.replace(
            r'$_{CP}$', '').replace(r'$_{GP}$', '').replace(r'$_{CP-non-delay}$', '').replace(r'$_{GP-non-delay}$', '')
        dataset_config, model_config, train_config = config.get_config(system_=system, model_name=model_name_)
        model, _ = model_config.get_model(run, train_config, dataset_config, model_version)
        to_return[model_name] = model
    return to_return


if __name__ == '__main__':
    set_everything(0)
    import wandb
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, default='s11')
    parser.add_argument('-n', type=int, default=5)
    # parser.add_argument('-m', type=str, default='figure')
    parser.add_argument('-m', type=str, default='cp-ood')
    # parser.add_argument('-t', type=float, default=16)
    parser.add_argument('-t', type=float, default=32)
    args = parser.parse_args()

    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    run = wandb.init(
        project="no",
        name=f'experiment {args.s} {args.n} {args.m} {get_time_str()}'
    )

    metric_list = ['l2_p_z', 'rl2_p_z']

    dataset_config, model_config, train_config = config.get_config(system_=args.s, duration=args.t)

    if args.m == 'table':
        if args.s == 's11':
            model_dict = {
                'DeepONet': 'v97',
                'GRU': 'v123',
                'GRU-FNO': 'v53'
            }
        elif args.s == 's9':
            model_dict = {
                'GRU-FNO': 'v0',
                'LSTM': 'v7',
                'FNO-GRU': 'v5'
            }
        else:
            raise NotImplementedError()
    elif args.m == 'figure':
        metric_list = ['l2_p_z', 'rl2_p_z']
        if args.s == 's11':
            model_dict = {
                'DeepONet': 'v97',
                'GRU': 'v123',
                'GRU-FNO': 'v53',
            }
        elif args.s == 's9':
            model_dict = {
                'GRU-FNO': 'v0',
                'LSTM': 'v7',
                'FNO-GRU': 'v5'
            }
        else:
            raise NotImplementedError()
    elif args.m == 'cp-ood':
        if args.s == 's11':
            dataset_config.random_test_lower_bound = -10
            dataset_config.random_test_upper_bound = 10
            train_config.uq_gamma = 0.01
            train_config.uq_alpha = 0.1
            model_dict = {
                # 'DeepONet': 'v97',
                # 'GRU': 'v123',
                'GRU-FNO': 'v53',
                'GRU-FNO$_{CP}$': 'v53',
                'GRU-FNO$_{CP-non-delay}$': 'v53'
            }
        elif args.s == 's9':
            dataset_config.random_test_lower_bound = 1
            dataset_config.random_test_upper_bound = 2
            train_config.uq_gamma = 0.01
            train_config.uq_alpha = 0.05
            model_dict = {
                'GRU-FNO': 'v0',
                'LSTM': 'v7',
                'FNO-GRU': 'v5'
            }
        else:
            raise NotImplementedError()
    elif args.m == 'alpha':
        if args.s == 's11':
            train_config.uq_type = 'conformal prediction'
            dataset_config.random_test_lower_bound = 1
            dataset_config.random_test_upper_bound = 2
            train_config.uq_gamma = 0.01
            alphas = [0.02, 1, 0.4]
        elif args.s == 's9':
            train_config.uq_type = 'conformal prediction'
            dataset_config.random_test_lower_bound = 1
            dataset_config.random_test_upper_bound = 2
            train_config.uq_gamma = 0.01
            alphas = [0.005, 0.05, 0.2]
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    to_return = load_model_for_experiments(model_dict=model_dict, system=args.s)

    results = None

    test_points = dataset_config.get_test_points(n_point=args.n)
    for i, test_point in enumerate(test_points):
        plot_name = f'{args.s}-{args.m}-{i}'
        if args.m == 'alpha':
            print(alphas, dataset_config.random_test_lower_bound, dataset_config.random_test_upper_bound)
            result_dict = plot_alpha(test_point, plot_name, dataset_config, train_config, model=None, alphas=alphas,
                                     system=args.s, metric_list=metric_list)
        else:
            result_dict = plot_comparisons(
                test_point, plot_name, dataset_config, train_config, system=args.s, metric_list=metric_list,
                model_dict=to_return
            )
        if results is None:
            results = {k: [] for k in result_dict.keys()}

        for k, v in result_dict.items():
            results[k].append(v)

    result_list_num = results['Successive \n Approximation']
    avg_prediction_time_num = sum([r.avg_prediction_time for r in result_list_num]) / len(result_list_num)
    # print(r'Method& Parameters& \makecell{Raw Prediction Time \\ (ms/prediction)}  &  Speedup & $\mathcal{E}$ & $\mathcal{E}^\prime$ & $\mathcal{E}_r$ & $\mathcal{E}_r^\prime$ \\ \midrule')
    print(
        r'Method& Parameters& \makecell{Raw Prediction Time \\ (ms/prediction)}  &  Speedup & $\mathcal{E}^\prime$ & $\mathcal{E}_r^\prime$ \\ \midrule')
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
               f'& {avg_prediction_time_num / avg_prediction_time:.3f} & {l2_p_z.item():.3f} & {rl2_p_z.item():.3f}\\\\'
        if method == 'Successive \n Approximation' or method == 'DeepONet' or method == 'LSTM':
            line += r' \midrule'
        if method == 'DeepONet-LSTM':
            line += r' \bottomrule'
        print(line)
