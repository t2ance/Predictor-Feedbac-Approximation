import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt

from config import get_config
from dynamic_systems import solve_integral_eular
from main import create_trajectory_dataset, create_random_dataset, simulation
from utils import load_model


def draw_distribution2(dataset_config):
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
        P_t = solve_integral_eular(Z_t=Z_t, U_D=U_D, dt=dataset_config.dt, n_state=dataset_config.n_state,
                                   n_points=dataset_config.n_point_delay,
                                   f=None)
        l = (P_t - p) ** 2
        loss += l
    print(loss / len(samples))
    ...


if __name__ == '__main__':
    dataset_config, model_config, train_config = get_config(system_='s5')
    # Z0 = tuple([1, 1] + [0] * 12)
    Z0 = tuple((np.random.random(14)).tolist())
    # Z0 = tuple(np.zeros(14).tolist())
    print('initial point', Z0)
    dataset_config.duration = 10
    dataset_config.delay = 0.0
    dataset_config.dt = 0.05
    # model, model_loaded = load_model(train_config, model_config, dataset_config)
    # result = simulation(method='no', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
    #                     img_save_path='./misc', model=model, silence=False)
    result = simulation(method='numerical', Z0=Z0, train_config=train_config, dataset_config=dataset_config,
                        img_save_path='./misc', silence=False)
    print(result.runtime)
