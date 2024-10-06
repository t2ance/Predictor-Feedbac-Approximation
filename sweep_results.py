import numpy as np
import wandb


def configs(system):
    if system == 's11':
        return 'Baxter', [
            '4tyhhp6d',
            'od3drbus',
            'bbwbjzdt',
            'k33ltrmv',
            'gb80iub8',
            'e2tasaff',
            '23rynq7w',
            'rvqjd5ej',
            'ehjvkvno',
            'klabxquq',
            'r7co67up',
            '84f1va6o'
        ]
    elif system == 's9':
        return 'Unicycle', [
            'eo39khhc',
            '7rwo3uhm',
            'i6q10x08',
            'rretghwz',
            'lebgrwxz',
            'k0v1wzdd',
            'y2kwag49',
            'n0qyjcjv',
            '8ppa097n',
            'rrlx6c4y',
            'on678c70',
            'jbru47hv'
        ]


orders = ['FNO', 'DeepONet', 'GRU', 'LSTM', 'FNO-GRU', 'FNO-LSTM', 'DeepONet-GRU', 'DeepONet-LSTM', 'GRU-FNO',
          'LSTM-FNO', 'GRU-DeepONet', 'LSTM-DeepONet']
if __name__ == '__main__':
    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    api = wandb.Api()

    system_name, sweep_ids = configs('s9')
    output_csv = f'./misc/sweep_summary_{system_name}.csv'

    data = []

    for sweep_id in sweep_ids:
        sweep = api.sweep(f'pqin/no/{sweep_id}')

        best_run = sweep.best_run()

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

        data.append([method, n_param, speed_up, l2, rl2, tr_loss, val_loss, tr_time, 'v' + str(model_version)])

        print(f"Extracted best run's data from sweep '{sweep_id}'")

    data_array = np.array(data, dtype=object)

    header = ['method', 'n_param', 'speedup', 'l2', 'rl2', 'training loss', 'validation loss',
              'training time (minutes)', 'version']

    order_dict = {method: index for index, method in enumerate(orders)}

    sort_keys = [order_dict.get(row[0], len(orders)) for row in data_array]

    sorted_indices = np.argsort(sort_keys)
    sorted_data_array = data_array[sorted_indices]

    data_with_header = np.vstack([header, sorted_data_array])
    np.savetxt(output_csv, data_with_header, delimiter=',', fmt='%s')
