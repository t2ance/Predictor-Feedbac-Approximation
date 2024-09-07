import wandb

from config import get_config
from main import main
from utils import print_args, set_everything


def set_config(config, dataset_config, model_config, train_config):
    print('Setting configuration')
    print(config)
    ffn, rnn = model_config.model_name.split('-')
    train_config.two_stage = False
    train_config.train_first_stage = False
    train_config.residual = True
    train_config.zero_init = True
    train_config.auxiliary_loss = False
    train_config.lr_scheduler_type = 'cosine_annealing_with_warmup'
    train_config.scheduler_min_lr = 0
    train_config.batch_size = 2048
    train_config.n_epoch = 2

    train_config.learning_rate = config.learning_rate
    train_config.weight_decay = config.weight_decay

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
        model_config.gru_layer_width = config.gru_layer_width
    elif rnn == 'LSTM':
        model_config.lstm_n_layer = config.lstm_n_layer
        model_config.lstm_layer_width = config.lstm_layer_width
    else:
        raise NotImplementedError()

    print_args(dataset_config)
    print_args(model_config)
    print_args(train_config)


def get_parameters(system: str, model_name: str):
    ffn, rnn = model_name.split('-')
    parameters = {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        }
    }

    if ffn == 'FNO':
        parameters.update({
            'fno_n_layer': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 6
            },
            'fno_n_modes_height': {
                'distribution': 'q_log_uniform_values',
                'q': 4,
                'min': 4,
                'max': 64
            },
            'fno_hidden_channels': {
                'distribution': 'q_log_uniform_values',
                'q': 4,
                'min': 4,
                'max': 128
            }
        })
    elif ffn == 'DeepONet':
        parameters.update({
            'deeponet_n_layer': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 6
            },
            'deeponet_hidden_size': {
                'distribution': 'q_log_uniform_values',
                'q': 4,
                'min': 4,
                'max': 64
            },
        })
    else:
        raise NotImplementedError()

    if rnn == 'GRU':
        parameters.update({
            'gru_n_layer': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 6
            },
            'gru_layer_width': {
                'distribution': 'q_log_uniform_values',
                'q': 4,
                'min': 4,
                'max': 128
            }
        })
    elif rnn == 'LSTM':
        parameters.update({
            'lstm_n_layer': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 6
            },
            'lstm_layer_width': {
                'distribution': 'q_log_uniform_values',
                'q': 4,
                'min': 4,
                'max': 128
            }
        })
    else:
        raise NotImplementedError()
    print(f'Getting parameters from {ffn}-{rnn}')
    print(parameters)
    return parameters


def do_sweep(system, model_name):
    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'l2',
            'goal': 'minimize'
        },
        'parameters': get_parameters(system, model_name)
    }

    sweep_id = wandb.sweep(sweep_config, project="no")

    def train(config=None):
        with wandb.init(config=config) as run:
            config = wandb.config
            dataset_config, model_config, train_config = get_config(system_=system, model_name=model_name)
            set_config(config, dataset_config, model_config, train_config)
            results, model = main(dataset_config, model_config, train_config, run, only_no_out=True)
            wandb.log(
                {
                    'l2': results['no'].l2
                }
            )

    wandb.agent(sweep_id, train)


if __name__ == '__main__':
    import argparse

    set_everything(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str)
    parser.add_argument('-model_name', type=str)
    args = parser.parse_args()

    do_sweep(system=args.s, model_name=args.model_name)
