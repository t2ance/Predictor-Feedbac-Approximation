import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt

from config import DatasetConfig
from dynamic_systems import solve_integral_eular
from main import create_trajectory_dataset, create_random_dataset, run


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


def classify_sample():
    import torch
    from torch.utils.data import TensorDataset
    from torch.utils.data import random_split
    from torch.utils.data import DataLoader
    from torch import nn
    from torch import optim
    def preprocess(samples):
        samples = [torch.hstack([sample[0], sample[1].cuda()])[1:] for sample in samples]
        return torch.vstack(samples)

    trajectory = torch.load('./s1/datasets/trajectory/train.pt')
    random = torch.load('./s1/datasets/random/train.pt')
    trajectory = preprocess(trajectory)[:2500].float()
    random = preprocess(random)[:2500].float()
    # 创建标签
    trajectory_labels = torch.ones(len(trajectory)).float()
    random_labels = torch.zeros(len(random)).float()

    # 合并数据和标签
    data = torch.cat((trajectory, random), dim=0)
    labels = torch.cat((trajectory_labels, random_labels), dim=0)

    # 创建数据集
    dataset = TensorDataset(data, labels)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], torch.Generator(device='cuda'))

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    class Classifier(nn.Module):
        def __init__(self, input_dim):
            super(Classifier, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            x = self.sigmoid(x)
            return x

    # 假设数据是向量，获取向量维度
    input_dim = data.shape[1]
    model = Classifier(input_dim)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs.squeeze(), batch_labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # 验证模型
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_data, batch_labels in val_loader:
            outputs = model(batch_data)
            loss = criterion(outputs.squeeze(), batch_labels)
            val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss}')

    # 获取模型预测的置信度
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        val_loss = 0
        for batch_data, batch_labels in val_loader:
            batch_data, batch_labels = batch_data, batch_labels
            outputs = model(batch_data)
            predicted = (outputs.squeeze() >= 0.5).float()
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
            loss = criterion(outputs.squeeze(), batch_labels)
            val_loss += loss.item()
        val_loss /= len(val_loader)
        accuracy = correct / total
        print(f'Validation Loss: {val_loss}')
        print(f'Validation Accuracy: {accuracy}')

    # 获取模型预测的置信度
    with torch.no_grad():
        confidences = model(data).squeeze()

    # 设置置信度阈值 (例如 0.9)
    threshold = 0.99
    high_confidence_random_indices = (confidences < (1 - threshold)) & (labels == 0)
    high_confidence_trajectory_indices = (confidences > threshold) & (labels == 1)

    high_confidence_random_data = data[high_confidence_random_indices]
    high_confidence_trajectory_data = data[high_confidence_trajectory_indices]

    print(f'Number of high confidence random data points: {len(high_confidence_random_data)}')
    print(f'Number of high confidence trajectory data points: {len(high_confidence_trajectory_data)}')

    # print(f'High confidence random data: {high_confidence_random_data}')
    # print(f'High confidence trajectory data: {high_confidence_trajectory_data}')

    def plot_sample(sample, name):
        dataset_config = DatasetConfig()
        feature, label = sample[:-2].squeeze(), sample[-2:].squeeze()
        if isinstance(feature, torch.Tensor):
            feature = feature.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        n_state = dataset_config.system.n_state
        z = feature[: n_state]
        u = feature[n_state:]
        p = label
        ts = np.linspace(0, dataset_config.delay, dataset_config.n_point_delay)
        plt.plot(ts, u, label='U')
        p_z_colors = ['red', 'blue']
        for i in range(n_state):
            plt.scatter(ts[-1], z[i], label=f'$Z_t({i})$', c=p_z_colors[i])
            plt.scatter(ts[-1], p[i], label=f'$P_t({i})$', c=p_z_colors[i], marker='^')
        plt.legend(loc='upper left')
        out_dir = f'./misc/sample'
        random_string = ""
        import random as r
        for i in range(8):
            random_string += str(r.randint(0, 9))
        plt.savefig(f'{out_dir}/{name}_{random_string}')
        plt.clf()

    for random_sample in high_confidence_random_data[:10]:
        plot_sample(random_sample, 'random')

    for trajectory_sample in high_confidence_trajectory_data[:10]:
        plot_sample(trajectory_sample, 'trajectory')


def successive_approximation_test():
    # 定义逐次逼近方法
    def successive_approximation(f_values, y0, x_values, n_iterations):
        num_points = len(x_values)
        y = np.zeros((n_iterations + 1, num_points))

        # 初始函数
        y[0, :] = y0

        # 逐次逼近
        for n in range(n_iterations):
            y[n + 1, 0] = y0  # 初值
            for i in range(1, num_points):
                dx = x_values[i] - x_values[i - 1]
                y[n + 1, i] = y[n + 1, i - 1] + f_values[n, i - 1] * dx

        return y

    # 初值
    y0 = 1
    x_values = np.linspace(0, 1, 1000)
    f_values = np.array([np.exp(x) for x in x_values])  # 用于测试的 f 值，这里用 e^x 代替
    f_values = np.tile(f_values, (5, 1))  # 模拟逐次逼近的 f 值序列

    n_iterations = 5

    # 计算逐次逼近
    y = successive_approximation(f_values, y0, x_values, n_iterations)

    # 解析解
    y_exact = np.exp(x_values)

    # 绘制结果
    plt.figure(figsize=(10, 6))
    for n in range(n_iterations + 1):
        plt.plot(x_values, y[n], label=f'Iteration {n}')
    plt.plot(x_values, y_exact, 'k--', label='Exact solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Successive Approximation Method')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # successive_approximation_test()
    # dataset_config = DatasetConfig(delay=0.5, duration=8, dt=0.05, integral_method='successive')
    dataset_config = DatasetConfig(delay=0.5, duration=8, dt=0.05, integral_method='successive')
    z = 0.5
    U, Z, P = run(method='numerical', Z0=(z, z), dataset_config=dataset_config, img_save_path='./misc')
    # classify_sample()
    # dataset_config = DatasetConfig(delay=1, duration=16)
    # U, Z, P = run(method='numerical', Z0=(2, 2, 2), dataset_config=dataset_config, img_save_path='./misc')

    # dataset_config = DatasetConfig(
    #     recreate_training_dataset=True,
    #     recreate_testing_dataset=False,
    #     data_generation_strategy='trajectory',
    #     random_u_type='spline',
    #     dt=0.02,
    #     n_dataset=500,
    #     system_n=2,
    #     system_c=5,
    #     n_sample_per_dataset=1,
    #     filter_ood_sample=True
    # )
    # samples = create_trajectory_dataset(dataset_config)
    # draw_distribution(samples)
    # Z_t = np.array([0, 1])
    # n_point_delay = dataset_config.n_point_delay
    # U_D = np.random.randn(n_point_delay)
    # predict_integral(Z_t=Z_t, n_point_delay=n_point_delay, n_state=2, dt=0.125, U_D=U_D,
    #                  dynamic=dataset_config.system.dynamic, )
    # run(dataset_config, (0, 1), method='numerical', img_save_path='./misc/result')
    # for _ in tqdm(range(36)):
    #     U, Z, P = run(method='numerical', Z0=np.random.random(2), dataset_config=dataset_config)
    # draw_distribution(dataset_config, False)
    # draw_distribution2(dataset_config)
    # draw_difference()
    # with open('./datasets/train.pkl', 'wb') as file:
    #     pickle.dump([], file)
    # with open('./datasets/train.pkl', 'rb') as file:
    #     training_samples_loaded = pickle.load(file)
    # print(len(training_samples_loaded))
    ...
