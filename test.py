import numpy as np

from config import get_config
from dynamic_systems import TimeVaryingDelay
from main import simulation, result_to_samples
from utils import load_model


def baxter_test_n_dof():
    # import wandb
    # wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    # run = wandb.init(
    #     project="no",
    #     name=f'test'
    # )
    # dataset_config, model_config, train_config = get_config(system_='s11', model_name='Inverted-FNO-GRU')
    dataset_config, model_config, train_config = get_config(system_='s11', model_name='FNO')
    # dataset_config.n_step = 4
    # method = 'no'
    # model = load_model(train_config, model_config, dataset_config)
    method = 'numerical'
    model = None
    # model_config.get_model(run, train_config, dataset_config, version='v168')
    Z0 = np.array(dataset_config.test_points[0])
    # Z0[dataset_config.baxter_dof:] = 0
    Z0[:] = 0
    print('initial point', Z0)
    # dataset_config.dataset_version = 'v0'
    # training_dataset, validation_dataset = load_dataset(dataset_config, train_config, [], run)
    # training_dataset
    result = simulation(method=method, Z0=Z0, train_config=train_config, dataset_config=dataset_config,
                        img_save_path='./misc', silence=False, model=model)
    print(result.runtime)
    # result_to_samples(result, dataset_config)
    return result


def baxter_test_unicycle():
    # dataset_config, model_config, train_config = get_config(system_='s9')
    from config import DatasetConfig, TrainConfig
    dataset_config = DatasetConfig(recreate_dataset=False, data_generation_strategy='trajectory',
                                   delay=TimeVaryingDelay(), duration=8, dt=0.004, n_training_dataset=100,
                                   n_validation_dataset=1, n_sample_per_dataset=-1, ic_lower_bound=0,
                                   # integral_method='rectangle',
                                   # integral_method='simpson',
                                   integral_method='successive adaptive',
                                   successive_approximation_threshold=1,
                                   ic_upper_bound=1, random_test_lower_bound=0, random_test_upper_bound=5)
    train_config = TrainConfig(learning_rate=1e-4, training_ratio=0.8, n_epoch=750, batch_size=64,
                               weight_decay=1e-3, log_step=-1, lr_scheduler_type='exponential',
                               scheduler_min_lr=1e-5)
    Z0 = dataset_config.test_points[0]
    print('initial point', Z0)
    dataset_config.duration = 6
    dataset_config.delay = TimeVaryingDelay()
    dataset_config.dt = 0.02
    result = simulation(method='numerical', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
                        img_save_path='./misc', silence=False)
    samples = result_to_samples(result, dataset_config)
    print(result.runtime)
    return result


def mini_train():
    from config import get_config

    # wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    # run = wandb.init(
    #     project="no",
    #     name=f'test'
    # )
    # dataset_config, model_config, train_config = get_config(system_='s9', model_name='FNO-GRU')
    # dataset_config, model_config, train_config = get_config(system_='s9', model_name='GRU')
    # dataset_config, model_config, train_config = get_config(system_='s9', model_name='FNO')
    # dataset_config, model_config, train_config = get_config(system_='s9', model_name='DeepONet-GRU')
    # dataset_config, model_config, train_config = get_config(system_='s9', model_name='DeepONet')
    # dataset_config, model_config, train_config = get_config(system_='s9', model_name='Inverted-FNO-GRU')
    dataset_config, model_config, train_config = get_config(system_='s9', model_name='Inverted-DeepONet-GRU')
    # dataset_config.dataset_version = 'v0'
    # training_dataset, validation_dataset = load_dataset(dataset_config, train_config, [], run)
    #
    model = load_model(train_config, model_config, dataset_config)
    # run_training(model_config=model_config, train_config=train_config, training_dataset=training_dataset,
    #              validation_dataset=validation_dataset, model=model)
    Z0 = dataset_config.test_points[0]
    result = simulation(method='no', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
                        model=model, img_save_path='./misc', silence=False)


if __name__ == '__main__':
    # mini_train()
    result = baxter_test_n_dof()
    # result = baxter_test_unicycle()
    # import wandb
    # from config import get_config
    #
    # wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    # run = wandb.init(
    #     project="no",
    #     name=f'test'
    # )
    # art = run.use_artifact(f"FNOProjectionGRU-s9:latest")
    # dataset_config_, model_config_, train_config_ = get_config(system_='s9', model_name='FNO')
    # training_dataset, validation_dataset = dataset_config_.load_dataset(run, resize=False)
