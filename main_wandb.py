from dataclasses import asdict

import numpy as np
import wandb

import config
from main import main
from utils import set_seed, get_time_str


def run_wandb(sweep: bool = True):
    set_seed(0)
    # setup_plt()
    dataset_config, model_config, train_config = config.get_default_config()
    print(f'Running with system {config.system}')
    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    if sweep:
        sweep_config = {
            'name': get_time_str(),
            'method': 'random',
            'metric': {
                'name': 'metric',
                'goal': 'minimize'
            },
            'parameters': {
                'fno_n_layers': {
                    'values': list(range(1, 5))
                },
                'fno_n_modes_height': {
                    'values': [4, 8, 16, 32, 64]
                },
                'fno_hidden_channels': {
                    'values': [4, 8, 16, 32, 64]
                },
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-2
                },
                'weight_decay': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-1
                }
            }
        }
        sweep_id = wandb.sweep(sweep_config, project='no-sweep')

        def sweep_main(config=None):
            with wandb.init(config=config):
                config = wandb.config
                train_config.learning_rate = config.learning_rate
                model_config.fno_hidden_channels = config.fno_hidden_channels
                model_config.fno_n_layers = config.fno_n_layers
                model_config.fno_n_modes_height = config.fno_n_modes_height
                train_config.weight_decay = config.weight_decay
                metric_rd, metric_mse, n_success = main(dataset_config, model_config, train_config)
                wandb.log({
                    "metric": np.nan if n_success != len(dataset_config.test_points) or metric_mse > 5 else metric_mse,
                    "n_success": n_success
                })

        wandb.agent(sweep_id, sweep_main)
    else:
        wandb.init(
            project="no",
            name=get_time_str(),
            config={
                **asdict(dataset_config),
                **asdict(model_config),
                **asdict(train_config)
            }
        )
        metric_rd, metric_mse, n_success = main(dataset_config, model_config, train_config)
        print(f'Relative L2 error: {metric_rd}')
        print(f'L2 error: {metric_mse}')
        print(f'Successful cases: [{n_success}/{len(dataset_config.test_points)}]')


if __name__ == '__main__':
    run_wandb(False)
