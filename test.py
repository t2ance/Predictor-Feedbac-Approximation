import numpy as np
from matplotlib import pyplot as plt

from config import DatasetConfig
from dynamic_systems import predict_integral, DynamicSystem
from main import create_trajectory_dataset, create_stateless_dataset


def draw_distribution():
    testing_random_samples = create_stateless_dataset(dataset_config, filter=True)

    print(len(testing_random_samples))
    testing_trajectory_samples = create_trajectory_dataset(dataset_config)
    u_trajectory_list = []
    u_random_list = []
    z0_trajectory_list = []
    z0_random_list = []
    z1_trajectory_list = []
    z1_random_list = []
    p0_trajectory_list = []
    p0_random_list = []
    p1_trajectory_list = []
    p1_random_list = []
    for feature, p in testing_trajectory_samples:
        feature = feature.cpu().numpy()
        # t = feature[:1]
        z0 = feature[1:2]
        z1 = feature[2:3]
        u = feature[3:]
        p = p.cpu().numpy()
        p0 = p[0:1]
        p1 = p[1:2]
        u_trajectory_list += u.tolist()
        z0_trajectory_list += z0.tolist()
        z1_trajectory_list += z1.tolist()
        p0_trajectory_list += p0.tolist()
        p1_trajectory_list += p1.tolist()
    for feature, p in testing_random_samples:
        feature = feature.cpu().numpy()
        # t = feature[:1]
        z0 = feature[1:2]
        z1 = feature[2:3]
        u = feature[3:]
        p = p.cpu().numpy()
        p0 = p[0:1]
        p1 = p[1:2]
        u_random_list += u.tolist()
        z0_random_list += z0.tolist()
        z1_random_list += z1.tolist()
        p0_random_list += p0.tolist()
        p1_random_list += p1.tolist()

    bins = 10
    alpha = 0.3
    plt.hist(u_trajectory_list, bins=bins, density=True, label='trajectory', alpha=alpha)
    plt.hist(u_random_list, bins=bins, density=True, label='random', alpha=alpha)
    plt.legend()
    plt.title('U')
    plt.xlabel('data')
    plt.ylabel('frequency')
    plt.show()

    plt.hist(z0_trajectory_list, bins=bins, density=True, label='trajectory', alpha=alpha)
    plt.hist(z0_random_list, bins=bins, density=True, label='random', alpha=alpha)
    plt.legend()
    plt.title('Z0')
    plt.xlabel('data')
    plt.ylabel('frequency')
    plt.show()

    plt.hist(z1_trajectory_list, bins=bins, density=True, label='trajectory', alpha=alpha)
    plt.hist(z1_random_list, bins=bins, density=True, label='random', alpha=alpha)
    plt.legend()
    plt.title('Z1')
    plt.xlabel('data')
    plt.ylabel('frequency')
    plt.show()

    plt.hist(p0_trajectory_list, bins=bins, density=True, label='trajectory', alpha=alpha)
    plt.hist(p0_random_list, bins=bins, density=True, label='random', alpha=alpha)
    plt.legend()
    plt.title('P0')
    plt.xlabel('data')
    plt.ylabel('frequency')
    plt.show()

    plt.hist(p1_trajectory_list, bins=bins, density=True, label='trajectory', alpha=alpha)
    plt.hist(p1_random_list, bins=bins, density=True, label='random', alpha=alpha)
    plt.legend()
    plt.title('P1')
    plt.xlabel('data')
    plt.ylabel('frequency')
    plt.show()


def draw_difference():
    samples = create_trajectory_dataset(dataset_config)
    loss = 0
    for feature, p in samples:
        feature = feature.cpu().numpy()
        # t = feature[:1]
        z0 = feature[1:2]
        z1 = feature[2:3]
        Z_t = feature[1:3]
        U_D = feature[3:]
        p = p.cpu().numpy()
        p0 = p[0:1]
        p1 = p[1:2]
        P_t = predict_integral(Z_t=Z_t, U_D=U_D, dt=dataset_config.dt, n_state=dataset_config.n_state,
                               n_point_delay=dataset_config.n_point_delay,
                               dynamic=DynamicSystem.dynamic_static)
        l = (P_t - p) ** 2
        loss += l
    print(loss / len(samples))
    ...


if __name__ == '__main__':
    dataset_config = DatasetConfig(
        recreate_training_dataset=True,
        recreate_testing_dataset=True,
        trajectory=False,
        random_u_type='spline',
        # random_u_type='poly',
        # random_u_type='sinexp',
        dt=0.1,
        n_dataset=200,
        duration=8,
        delay=3.,
        n_sample_per_dataset=1,
        ic_lower_bound=0,
        ic_upper_bound=1,
        postprocess=False,
        n_plot_sample=False
    )
    draw_difference()
    ...
