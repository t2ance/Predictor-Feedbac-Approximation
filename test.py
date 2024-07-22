from config import get_config
from main import simulation


def baxter_test2dof():
    dataset_config, model_config, train_config = get_config(system_='s5')
    Z0 = tuple([1, 1, 0, 0])
    print('initial point', Z0)
    dataset_config.duration = 10
    dataset_config.delay = 0.1
    dataset_config.dt = 0.01
    dataset_config.baxter_dof = 2
    result = simulation(method='numerical', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
                        img_save_path='./misc', silence=False)
    print(result.runtime)


def baxter_test7dof():
    dataset_config, model_config, train_config = get_config(system_='s5')
    Z0 = tuple([1, 1] + [0] * 12)
    # Z0 = tuple((np.random.random(14) * 0.3).tolist())
    # Z0 = tuple(np.zeros(14).tolist())
    print('initial point', Z0)
    dataset_config.duration = 6
    dataset_config.delay = 0.05
    dataset_config.dt = 0.01
    dataset_config.baxter_dof = 7
    result = simulation(method='numerical', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
                        img_save_path='./misc', silence=False)
    print(result.runtime)


if __name__ == '__main__':
    baxter_test2dof()
    # baxter_test7dof()
