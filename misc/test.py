import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import curve_fit

from config import DatasetConfig
from main import run, create_trajectory_dataset


def fit(x_data, y_data):
    def func(x, c, k):
        return c * np.exp(-k * x)

    popt, pcov = curve_fit(func, x_data, y_data)
    # print(popt)
    plt.plot(x_data, y_data, 'ro', label='original data')
    plt.plot(x_data, func(x_data, *popt), 'b-', label='fitted curve')
    plt.legend()
    plt.show()
    return popt


if __name__ == '__main__':
    dataset_config = DatasetConfig(n_sample_per_dataset=5, dt=0.01, duration=4, recreate_training_dataset=True,
                                   recreate_testing_dataset=True)
    samples = create_trajectory_dataset(dataset_config)
    for sample in samples:
        t_z_u = sample[0].cpu().numpy()
        t = t_z_u[0]
        z = t_z_u[1:3]
        u = t_z_u[3:]
        p = sample[1].cpu().numpy()
        # print(sample)
        ts = np.linspace(t - dataset_config.delay, t, dataset_config.n_point_delay)
        plt.plot(ts, u, label='u')
        plt.scatter(ts[-1], z[0], label='Zt_0')
        plt.scatter(ts[-1], z[1], label='Zt_1')
        plt.scatter(ts[-1], p[0], label='Pt_0')
        plt.scatter(ts[-1], p[1], label='Pt_1')
        plt.legend()
        # plt.ylim([-2, 2])
        plt.show()
        plt.clf()
        print('_')
    # U, Z, _ = run(method='explict', silence=True, Z0=(0, 1.), dataset_config=dataset_config)
    # U, Z, _ = run(method='numerical', silence=True, Z0=(0, 1.), dataset_config=dataset_config)
    # popt = fit(dataset_config.ts[dataset_config.n_point_delay:], U[dataset_config.n_point_delay:])
    # operator = FNO1d(n_modes_height=16, hidden_channels=64, in_channels=1, out_channels=1)
    # operator = PredictionFNO(n_modes_height=16, hidden_channels=64, in_features=100,
    #                          out_features=3, n_layers=4)
    # batch = torch.ones(size=(16, 100))
    # result = operator(batch)
    # train_loader, test_loaders, data_processor = load_darcy_flow_small(
    #     n_train=100, batch_size=4,
    #     test_resolutions=[16, 32], n_tests=[50, 50], test_batch_sizes=[4, 2],
    # )
    #
    # train_dataset = train_loader.dataset
    # for res, test_loader in test_loaders.items():
    #     print(res)
    #     # Get first batch
    #     batch = next(iter(test_loader))
    #     x = batch['x']
    #     y = batch['y']
    #
    #     print(f'Testing samples for res {res} have shape {x.shape[1:]}')
    #
    # data = train_dataset[0]
    # x = data['x']
    # y = data['y']
    #
    # print(f'Training sample have shape {x.shape[1:]}')
    #
    # # Which sample to view
    # index = 0
    #
    # data = train_dataset[index]
    # data = data_processor.preprocess(data, batched=False)
    # x = data['x']
    # y = data['y']
    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(2, 2, 1)
    # ax.imshow(x[0], cmap='gray')
    # ax.set_title('input x')
    # ax = fig.add_subplot(2, 2, 2)
    # ax.imshow(y.squeeze())
    # ax.set_title('input y')
    # ax = fig.add_subplot(2, 2, 3)
    # ax.imshow(x[1])
    # ax.set_title('x: 1st pos embedding')
    # ax = fig.add_subplot(2, 2, 4)
    # ax.imshow(x[2])
    # ax.set_title('x: 2nd pos embedding')
    # fig.suptitle('Visualizing one input sample', y=0.98)
    # plt.tight_layout()
    # fig.show()
