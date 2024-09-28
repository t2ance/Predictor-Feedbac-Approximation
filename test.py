import numpy as np

import dynamic_systems
from config import get_config
from dynamic_systems import ConstantDelay, TimeVaryingDelay
from main import simulation, result_to_samples, load_dataset, run_training
from utils import load_model


def baxter_test_n_dof():
    # import wandb
    # wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    # run = wandb.init(
    #     project="no",
    #     name=f'test'
    # )
    dataset_config, model_config, train_config = get_config(system_='s10', model_name='FNO')
    # model = load_model(train_config, model_config, dataset_config)
    # model_config.get_model(run, train_config, dataset_config, version='v168')
    Z0 = dataset_config.test_points[0]
    print('initial point', Z0)
    # dataset_config.dataset_version = 'v0'
    # training_dataset, validation_dataset = load_dataset(dataset_config, train_config, [], run)
    # training_dataset
    Z0 = [0.07407145, 0.11798713, 0.06306392, 0.24340997, 0.27855349, 0.11211986, 0.1920229, 0.19603325, 0.00758645,
          0.23880707]
    result = simulation(method='numerical', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
                        img_save_path='./misc', silence=False)
    print(result.runtime)
    # result_to_samples(result, dataset_config)
    return result


def baxter_test_unicycle():
    dataset_config, model_config, train_config = get_config(system_='s9')
    Z0 = tuple([1, 1, 1])
    print('initial point', Z0)
    dataset_config.duration = 10
    dataset_config.delay = TimeVaryingDelay()
    dataset_config.dt = 0.1
    result = simulation(method='numerical', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
                        img_save_path='./misc', silence=False)
    samples = result_to_samples(result, dataset_config)
    print(result.runtime)


def mini_train():
    import wandb
    from config import get_config

    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    run = wandb.init(
        project="no",
        name=f'test'
    )
    dataset_config, model_config, train_config = get_config(system_='s9', model_name='FNO-GRU')
    # dataset_config, model_config, train_config = get_config(system_='s9', model_name='DeepONet-GRU')
    # dataset_config, model_config, train_config = get_config(system_='s9', model_name='GRU')
    # dataset_config, model_config, train_config = get_config(system_='s9', model_name='FNO')
    # dataset_config, model_config, train_config = get_config(system_='s9', model_name='DeepONet')
    dataset_config.dataset_version = 'v0'
    training_dataset, validation_dataset = load_dataset(dataset_config, train_config, [], run)

    model = load_model(train_config, model_config, dataset_config)
    run_training(model_config=model_config, train_config=train_config, training_dataset=training_dataset,
                 validation_dataset=validation_dataset, model=model)
    # result = simulation(method='no', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
    #                     img_save_path='./misc', silence=False)


if __name__ == '__main__':
    # mini_train()
    # result = baxter_test_unicycle()
    result = baxter_test_n_dof()
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
