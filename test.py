import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt

from config import DatasetConfig
from dynamic_systems import solve_integral_equation
from main import create_trajectory_dataset, create_random_dataset, run


def draw_distribution(dataset_config):
    testing_random_samples = create_random_dataset(dataset_config)

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
    p_z_ratio_trajectory_list = []
    p_z_ratio_random_list = []
    for feature, p in testing_trajectory_samples:
        feature = feature.cpu().numpy()
        # t = feature[:1]
        z = feature[1:3]
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
        p_z_ratio_trajectory_list.append(np.linalg.norm(p) / np.linalg.norm(z))
    for feature, p in testing_random_samples:
        feature = feature.cpu().numpy()
        # t = feature[:1]
        z = feature[1:3]
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
        p_z_ratio_random_list.append(np.linalg.norm(p) / np.linalg.norm(z))

    bins = 100
    alpha = 0.5

    def draw_1d(t, r, title, xlabel='data', ylabel='density', xlim=None):
        if xlim is None:
            xlim = [-5, 5]
        plt.hist(t, bins=bins, density=True, label='trajectory', alpha=alpha)
        plt.hist(r, bins=bins, density=True, label='random', alpha=alpha)
        plt.title(title)
        plt.xlim(xlim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def draw_2d(t0, t1, r0, r1, title, xlabel='data', ylabel='density', xlim=None):
        if xlim is None:
            xlim = [-5, 5]
        plt.hist2d(t0, t1, bins=[10, 10], alpha=alpha, cmap='Greens')
        plt.hist2d(r0, r1, bins=[10, 10], alpha=alpha, cmap='Reds')
        plt.colorbar(label='Density')
        plt.title(title)
        plt.xlim(xlim)
        plt.ylim(xlim)
        blue_patch = mpatches.Patch(color='green', label='trajectory')
        red_patch = mpatches.Patch(color='red', label='random')
        plt.legend(handles=[blue_patch, red_patch])
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        plt.show()

    draw_1d(p_z_ratio_trajectory_list, p_z_ratio_random_list, r'$\frac{||P||_2}{||Z||_2}$')
    draw_1d(u_trajectory_list, u_random_list, 'U')
    draw_1d(z0_trajectory_list, z0_random_list, '$Z_0$')
    draw_1d(z1_trajectory_list, z1_random_list, '$Z_1$')
    draw_1d(p0_trajectory_list, p0_random_list, '$P_0$')
    draw_1d(p1_trajectory_list, p1_random_list, '$P_1$')
    # draw_2d(z0_trajectory_list, z1_trajectory_list, z0_random_list, z1_random_list, title='Z')
    # draw_2d(p0_trajectory_list, p1_trajectory_list, p0_random_list, p1_random_list, title='P')


def draw_difference(dataset_config):
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
        P_t = solve_integral_equation(Z_t=Z_t, U_D=U_D, dt=dataset_config.dt, n_state=dataset_config.n_state,
                                      n_point_delay=dataset_config.n_point_delay,
                                      dynamic=None)
        l = (P_t - p) ** 2
        loss += l
    print(loss / len(samples))
    ...


if __name__ == '__main__':
    dataset_config = DatasetConfig(
        recreate_training_dataset=True,
        recreate_testing_dataset=True,
        data_generation_strategy='random',
        random_u_type='spline',
        dt=0.125,
        # dt=0.05,
        n_dataset=500,
        duration=8,
        delay=3.,
        system_n=1,
        system_c=1,
        n_sample_per_dataset=1,
        filter_ood_sample=True
    )
    # Z_t = np.array([0, 1])
    # n_point_delay = dataset_config.n_point_delay
    # U_D = np.random.randn(n_point_delay)
    # predict_integral(Z_t=Z_t, n_point_delay=n_point_delay, n_state=2, dt=0.125, U_D=U_D,
    #                  dynamic=dataset_config.system.dynamic, )
    # run(dataset_config, (0, 1), method='numerical', img_save_path='./misc/result')
    # for _ in tqdm(range(36)):
    #     U, Z, P = run(method='numerical', Z0=np.random.random(2), dataset_config=dataset_config)
    # draw_distribution(dataset_config, False)
    draw_distribution(dataset_config)
    # draw_difference()
    # with open('./datasets/train.pkl', 'wb') as file:
    #     pickle.dump([], file)
    # with open('./datasets/train.pkl', 'rb') as file:
    #     training_samples_loaded = pickle.load(file)
    # print(len(training_samples_loaded))
    ...
