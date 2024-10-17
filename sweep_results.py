import numpy as np
import wandb


def configs(system):
    if system == 's11':
        return 'Baxter', [
            # '4tyhhp6d',
            # 'od3drbus',
            # 'bbwbjzdt',
            'k33ltrmv',
            'gb80iub8',
            'e2tasaff',
            # '23rynq7w',
            'rvqjd5ej',
            'ehjvkvno',
            # 'klabxquq',
            'r7co67up',
            '84f1va6o',

            '9shtn1hj',
            'rvici6ry',
            'alght5jf',
            # 'lp3qogo2',
            # 'aj60uvh6',

            'i79uwsj3',
            '6tc829a2'
        ]
    elif system == 's9':
        return 'Unicycle', [
            # 'eo39khhc',
            # '7rwo3uhm',
            # 'i6q10x08',
            'rretghwz',
            'lebgrwxz',
            'k0v1wzdd',
            # 'y2kwag49',
            'n0qyjcjv',
            '8ppa097n',
            # 'rrlx6c4y',
            # 'on678c70',
            'jbru47hv',
            'v676f0xr',

            'qqx1l4qg',
            'mv2el4ja',
            # '1it8mei5',
            # 'dak0vutk',
            # 'scpea7r1'
            'ti7nk6os',
            'gc405cwp',
            'ytkutmjf'
        ]


# s9 LSTM-FNO v676f0xr

orders = ['FNO', 'DeepONet', 'GRU', 'LSTM', 'FNO-GRU', 'FNO-LSTM', 'DeepONet-GRU', 'DeepONet-LSTM', 'GRU-FNO',
          'LSTM-FNO', 'GRU-DeepONet', 'LSTM-DeepONet']


def get_best_run(sweep, metric_name='l2', max_parameter=np.inf, **kwargs):
    runs = sweep.runs
    best_run = None
    best_metric = None
    for run in runs:
        if run.state != 'finished':
            continue
        current_metric = run.summary.get(metric_name)
        if current_metric == 'NaN':
            continue
        if run.summary.get('n params') > max_parameter or run.summary.get('speedup') < 1:
            continue
        if best_run is None or current_metric < best_metric:
            best_run = run
            best_metric = current_metric
    if best_run is None:
        print(f'No best result found for {sweep.name} under given conditions!')
        return run
    return best_run


if __name__ == '__main__':
    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    api = wandb.Api()

    systems = [
        {
            'system': 's9'
        },
        {
            'system': 's11',
            'max_parameter': 1_000_000
        }
    ]

    for system_info in systems:
        system_name, sweep_ids = configs(system_info['system'])
        output_csv = f'./misc/sweep_summary_{system_name}.csv'
        output_csv_with_float = f'./misc/sweep_summary_{system_name}_out.csv'

        data = []

        for sweep_id in sweep_ids:
            print(f"Extracting best run's data from sweep '{sweep_id}'")
            sweep = api.sweep(f'pqin/no/{sweep_id}')

            # best_run = sweep.best_run()
            best_run = get_best_run(sweep, **system_info)

            sweep_name_parts = sweep.name.split()
            if len(sweep_name_parts) < 3:
                print(f"Cannot extract method from sweep name {sweep.name}")
                method = "Unknown"
            else:
                method = sweep_name_parts[2]

            n_param = best_run.summary.get('n params', 'N/A')
            speed_up = best_run.summary.get('speedup', 'N/A')
            l2 = best_run.summary.get('l2', 'N/A')
            rl2 = best_run.summary.get('rl2', 'N/A')
            if l2 == 'NaN':
                l2 = 'Inf'
            if rl2 == 'NaN':
                rl2 = 'Inf'
            tr_loss = best_run.summary.get('training loss', 'N/A')
            val_loss = best_run.summary.get('validating loss', 'N/A')
            tr_time = best_run.summary.get('training time', 'N/A')
            model_version = best_run.summary.get('model_version', 'N/A')

            data.append(
                [method, n_param, speed_up, l2, rl2, tr_loss, val_loss, tr_time, 'v' + str(model_version), sweep_id])

            print(f"Extracted best run's data from sweep '{sweep_id}'")

        data_array = np.array(data, dtype=object)

        header = [
            'method', 'n_param', 'speedup', 'l2', 'rl2', 'training loss', 'validation loss', 'training time (minutes)',
            'version', 'sweep_id'
        ]

        order_dict = {method: index for index, method in enumerate(orders)}

        sort_keys = [order_dict.get(row[0], len(orders)) for row in data_array]

        sorted_indices = np.argsort(sort_keys)
        sorted_data_array = data_array[sorted_indices]

        data_with_header = np.vstack([header, sorted_data_array])
        np.savetxt(output_csv, data_with_header, delimiter=',', fmt='%s')


        def format_data(data):
            try:
                if isinstance(data, float):
                    return f"{float(data):.3f}"
                return data
            except ValueError:
                return data


        data_without_columns = np.delete(data_with_header,
                                         [
                                             # 'method',
                                             # 'n_param',
                                             # 'speedup',
                                             # 'l2',
                                             # 'rl2',
                                             5,6,7,8,9
                                             # 'training loss',
                                             # 'validation loss',
                                             # 'training time (minutes)',
                                             # 'version',
                                             # 'sweep_id'
                                         ], axis=1)

        formatted_data = np.vectorize(format_data)(data_without_columns)

        np.savetxt(output_csv_with_float, formatted_data, delimiter=',', fmt='%s')
