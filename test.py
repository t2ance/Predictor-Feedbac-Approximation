import numpy as np

import dynamic_systems
from config import get_config
from dynamic_systems import ConstantDelay, TimeVaryingDelay
from main import simulation
from utils import l2_p_phat


def baxter_test1dof():
    dataset_config, model_config, train_config = get_config(system_='s5')
    Z0 = tuple([1, 1])
    print('initial point', Z0)
    dataset_config.duration = 20
    dataset_config.delay = 1
    dataset_config.dt = 0.05
    dataset_config.baxter_dof = 1
    result = simulation(method='numerical', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
                        img_save_path='./misc', silence=False)
    print(result.runtime)


def baxter_test2dof():
    dataset_config, model_config, train_config = get_config(system_='s5')
    dataset_config.baxter_dof = 5
    Z0 = tuple(
        np.concatenate([np.random.uniform(0, 0.3, dataset_config.baxter_dof), np.zeros(dataset_config.baxter_dof)]))
    print('initial point', Z0)
    dataset_config.duration = 10
    dataset_config.delay = ConstantDelay(0.5)
    dataset_config.dt = 0.02
    dataset_config.successive_approximation_threshold = 1e-14
    dataset_config.integral_method = 'rectangle'
    result = simulation(method='numerical', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
                        img_save_path='./misc', silence=False)
    print(result.runtime)
    return result


def baxter_test_n_dof(n):
    dataset_config, model_config, train_config = get_config(system_='s5')
    Z0 = tuple([0.2] * (n * 2))
    print('initial point', Z0)
    dataset_config.duration = 10
    dataset_config.delay = dynamic_systems.ConstantDelay(0.2)
    dataset_config.successive_approximation_threshold = 1e-10
    dataset_config.dt = 0.005
    dataset_config.baxter_dof = n
    result = simulation(method='numerical', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
                        img_save_path='./misc', silence=False)
    print(result.P_numerical_n_iters.mean())
    print(result.runtime)


def baxter_test7dof():
    dataset_config, model_config, train_config = get_config(system_='s5')
    Z0 = tuple([0.1, 0.1] + [0] * 12)
    # Z0 = tuple((np.random.random(14) * 0.3).tolist())
    # Z0 = tuple(np.zeros(14).tolist())
    print('initial point', Z0)
    dataset_config.duration = 10
    dataset_config.delay = ConstantDelay(0)
    dataset_config.dt = 0.01
    dataset_config.baxter_dof = 7
    result = simulation(method='numerical', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
                        img_save_path='./misc', silence=False)
    print(result.runtime)


def baxter_test_s6():
    dataset_config, model_config, train_config = get_config(system_='s6')
    Z0 = tuple([0.5, 0.5, 0.5])
    print('initial point', Z0)
    dataset_config.duration = 50
    dataset_config.delay = 0.1
    dataset_config.dt = 0.05
    result = simulation(method='numerical', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
                        img_save_path='./misc', silence=False)
    print(result.runtime)


def baxter_test_unicycle():
    dataset_config, model_config, train_config = get_config(system_='s7')
    Z0 = tuple([1, 1, 1])
    print('initial point', Z0)
    dataset_config.duration = 10
    dataset_config.delay = TimeVaryingDelay()
    dataset_config.dt = 0.05
    result = simulation(method='numerical', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
                        img_save_path='./misc', silence=False)
    # l2 = l2_p_phat(result.P_numerical, result.P_numerical, dataset_config.n_point_start())
    # print(l2)
    # print(result.runtime)


if __name__ == '__main__':
    # result = baxter_test2dof()
    # result = baxter_test_unicycle()
    import wandb
    from config import get_config

    wandb.login(key='ed146cfe3ec2583a2207a02edcc613f41c4e2fb1')
    run = wandb.init(
        project="no",
        name=f'test'
    )
    art = run.use_artifact(f"FNOProjectionGRU-s9:latest")
    # dataset_config_, model_config_, train_config_ = get_config(system_='s9', model_name='FNO')
    # training_dataset, validation_dataset = dataset_config_.load_dataset(run, resize=False)
