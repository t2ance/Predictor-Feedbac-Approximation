import numpy as np
import wandb


def configs(system):
    if system == 's11':
        return 'Baxter', [
            "pqin/no/quvawpkf",
            "pqin/no/9r0bavra",
            "pqin/no/udhjtnfo",
            "pqin/no/1o5hwfs8",
            "pqin/no/36l10gkv",
            "pqin/no/gtrshkls",
            "pqin/no/jahqq1hw",
            "pqin/no/c79wh9go",
            "pqin/no/f3494yci",
            "pqin/no/7neth19c",
            "pqin/no/pstv1cq0",
            "pqin/no/6f0xx2zm",
        ]
    elif system == 's8':
        return 'Unicycle', [
            "pqin/no/bju3cjjz",
            "pqin/no/b1iszd5b",
            "pqin/no/tdzjci0n",
            "pqin/no/2h6kcsp6",
            "pqin/no/1h8hiluk",
            "pqin/no/5osogwht",
            "pqin/no/mzxcsi37",
            "pqin/no/psjwg65d",
            "pqin/no/xsz6wa5e",
            "pqin/no/e7pr5ams",
            "pqin/no/wxcokpbo",
            "pqin/no/ket8c0k7",
        ]


orders = ['FNO', 'DeepONet', 'GRU', 'LSTM', 'FNO-GRU', 'FNO-LSTM', 'DeepONet-GRU', 'DeepONet-LSTM', 'GRU-FNO',
          'LSTM-FNO', 'GRU-DeepONet', 'LSTM-DeepONet']
if __name__ == '__main__':
    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    api = wandb.Api()

    system_name, sweep_ids = configs('s11')
    output_csv = f'./misc/sweep_summary_{system_name}.csv'

    data = []

    for sweep_id in sweep_ids:
        try:
            sweep = api.sweep(sweep_id)

            best_run = sweep.best_run()

            sweep_name_parts = sweep.name.split()
            if len(sweep_name_parts) < 3:
                print(f"Cannot extract method from sweep name {sweep.name}")
                method = "Unknown"
            else:
                method = sweep_name_parts[2]
                if 'Inverted' in method:
                    splits = method.split('-')
                    method = splits[2] + '-' + splits[1]

            n_param = best_run.summary.get('n params', 'N/A')
            speed_up = best_run.summary.get('speedup', 'N/A')
            l2 = best_run.summary.get('l2', 'N/A')
            rl2 = best_run.summary.get('rl2', 'N/A')
            tr_loss = best_run.summary.get('training loss', 'N/A')
            val_loss = best_run.summary.get('validating loss', 'N/A')
            tr_time = best_run.summary.get('training time', 'N/A')

            data.append([method, n_param, speed_up, l2, rl2, tr_loss, val_loss, tr_time])

            print(f"Extracted best run's data from sweep '{sweep_id}'")

        except Exception as e:
            print(f"Error occurred in processing sweep {sweep_id}")

    data_array = np.array(data, dtype=object)

    header = ['method', 'n_param', 'speedup', 'l2', 'rl2', 'training loss', 'validation loss',
              'training time (minutes)']

    order_dict = {method: index for index, method in enumerate(orders)}

    sort_keys = [order_dict.get(row[0], len(orders)) for row in data_array]

    sorted_indices = np.argsort(sort_keys)
    sorted_data_array = data_array[sorted_indices]

    data_with_header = np.vstack([header, sorted_data_array])
    np.savetxt(output_csv, data_with_header, delimiter=',', fmt='%s')
