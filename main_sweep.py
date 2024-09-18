import wandb

from config import get_config
from main import main, load_dataset
from utils import print_args, get_time_str
import numpy as np


def set_config(config, dataset_config, model_config, train_config):
    print('Setting configuration')
    print(config)
    train_config.lr_scheduler_type = 'cosine_annealing_with_warmup'
    train_config.scheduler_min_lr = 0
    train_config.batch_size = 2048
    if dataset_config.system_ == 's8':
        dataset_config.n_training_dataset = 1400
        train_config.n_epoch = 200
        model_config.init_type = 'kaiming'
    elif dataset_config.system_ == 's9':
        dataset_config.n_training_dataset = 250
        train_config.n_epoch = 100
    else:
        raise NotImplementedError()

    train_config.learning_rate = config.learning_rate
    train_config.weight_decay = config.weight_decay
    model_name = model_config.model_name
    if '-' in model_name:
        ffn, rnn = model_name.split('-')
        if ffn == 'FNO':
            model_config.fno_n_layer = config.fno_n_layer
            model_config.fno_n_modes_height = config.fno_n_modes_height
            model_config.fno_hidden_channels = config.fno_hidden_channels
        elif ffn == 'DeepONet':
            model_config.deeponet_hidden_size = config.deeponet_hidden_size
            model_config.deeponet_n_layer = config.deeponet_n_layer
        else:
            raise NotImplementedError()

        if rnn == 'GRU':
            model_config.gru_n_layer = config.gru_n_layer
            model_config.gru_hidden_size = config.gru_hidden_size
        elif rnn == 'LSTM':
            model_config.lstm_n_layer = config.lstm_n_layer
            model_config.lstm_hidden_size = config.lstm_hidden_size
        else:
            raise NotImplementedError()
    else:
        if model_name == 'FNO':
            model_config.fno_n_layer = config.fno_n_layer
            model_config.fno_n_modes_height = config.fno_n_modes_height
            model_config.fno_hidden_channels = config.fno_hidden_channels
        elif model_name == 'DeepONet':
            model_config.deeponet_hidden_size = config.deeponet_hidden_size
            model_config.deeponet_n_layer = config.deeponet_n_layer
        elif model_name == 'GRU':
            model_config.gru_n_layer = config.gru_n_layer
            model_config.gru_hidden_size = config.gru_hidden_size
        elif model_name == 'LSTM':
            model_config.lstm_n_layer = config.lstm_n_layer
            model_config.lstm_hidden_size = config.lstm_hidden_size
        else:
            raise NotImplementedError()

    print_args(dataset_config)
    print_args(model_config)
    print_args(train_config)


def get_parameters(system: str, model_name: str):
    parameters = {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-2
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        }
    }

    fno_params = {
        'fno_n_layer': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 4
        },
        'fno_n_modes_height': {
            'distribution': 'q_log_uniform_values',
            'q': 4,
            'min': 4,
            'max': 256
        },
        'fno_hidden_channels': {
            'distribution': 'q_log_uniform_values',
            'q': 4,
            'min': 4,
            'max': 256
        }
    }
    deeponet_params = {
        'deeponet_n_layer': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 4
        },
        'deeponet_hidden_size': {
            'distribution': 'q_log_uniform_values',
            'q': 4,
            'min': 4,
            'max': 64
        },
    }

    if system == 's8':
        gru_params = {
            'gru_n_layer': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 6
            },
            'gru_hidden_size': {
                'distribution': 'q_log_uniform_values',
                'q': 16,
                'min': 1,
                'max': 256
            }
        }
        lstm_params = {
            'lstm_n_layer': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 6
            },
            'lstm_hidden_size': {
                'distribution': 'q_log_uniform_values',
                'q': 1,
                'min': 64,
                'max': 256
            }
        }
        parameters = {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-7,
                'max': 1e-2
            }
        }
    elif system == 's9':
        gru_params = {
            'gru_n_layer': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 4
            },
            'gru_hidden_size': {
                'distribution': 'q_log_uniform_values',
                'q': 4,
                'min': 16,
                'max': 64
            }
        }
        lstm_params = {
            'lstm_n_layer': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 4
            },
            'lstm_hidden_size': {
                'distribution': 'q_log_uniform_values',
                'q': 4,
                'min': 16,
                'max': 64
            }
        }
    else:
        raise NotImplementedError()
    if '-' in model_name:
        ffn, rnn = model_name.split('-')
        if ffn == 'FNO':
            parameters.update(fno_params)
        elif ffn == 'DeepONet':
            parameters.update(deeponet_params)
        else:
            raise NotImplementedError()
        if rnn == 'GRU':
            parameters.update(gru_params)
        elif rnn == 'LSTM':
            parameters.update(lstm_params)
        else:
            raise NotImplementedError()
        print(f'Getting parameters from {ffn}-{rnn}')
    else:
        if model_name == 'FNO':
            parameters.update(fno_params)
        elif model_name == 'DeepONet':
            parameters.update(deeponet_params)
        elif model_name == 'GRU':
            parameters.update(gru_params)
        elif model_name == 'LSTM':
            parameters.update(lstm_params)
        else:
            raise NotImplementedError()

    if 'GRU' in model_name or 'LSTM' in model_name:
        parameters['learning_rate'] = {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        }
    print(parameters)
    return parameters


def do_sweep(system, model_name):
    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    sweep_config = {
        "name": 'Sweep ' + system + ' ' + model_name + ' ' + get_time_str(),
        'method': 'bayes',
        'metric': {
            'name': 'validating loss',
            'goal': 'minimize'
        },
        'parameters': get_parameters(system, model_name)
    }

    sweep_id = wandb.sweep(sweep_config, project="no")

    class Data:
        training_dataset = None
        validation_dataset = None
        test_points = None

    data = Data()

    def train(config=None):
        with wandb.init(config=config) as run:
            config = wandb.config
            dataset_config, model_config, train_config = get_config(system_=system, model_name=model_name)
            set_config(config, dataset_config, model_config, train_config)
            if data.training_dataset is None:
                test_points = dataset_config.test_points
                training_dataset, validation_dataset = load_dataset(dataset_config, train_config, test_points, run)
                data.training_dataset = training_dataset
                data.validation_dataset = validation_dataset
                data.test_points = test_points
                print('Create data for current sweep')
            else:
                print('Data already created for current sweep, skip')

            results, model = main(dataset_config, model_config, train_config, run, only_no_out=True, save_model=True,
                                  training_dataset=data.training_dataset,
                                  validation_dataset=data.validation_dataset,
                                  test_points=data.test_points)
            l2 = results['no'].l2
            if l2 > 10:
                l2 = np.nan
            wandb.log(
                {
                    'l2': l2
                }
            )

    wandb.agent(sweep_id, train)


if __name__ == '__main__':
    import argparse

    # successful:
    # s8:
    #
    # s9: all can work, except some combined method
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str)
    parser.add_argument('-model_name', type=str)
    args = parser.parse_args()

    do_sweep(system=args.s, model_name=args.model_name)
