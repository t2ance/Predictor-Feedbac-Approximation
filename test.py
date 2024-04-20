from config import DatasetConfig
from main import create_trajectory_dataset, f, calculate_P_t

if __name__ == '__main__':
    dataset_config = DatasetConfig()
    samples_ = create_trajectory_dataset(
        1, dataset_config.duration, dataset_config.delay, dataset_config.n_point,
        dataset_config.n_delay_step, dataset_config.n_sample_per_dataset, dataset_config.dataset_file)
    loss = 0.
    for sample in samples_:
        features, label = sample
        t = features[:1].cpu().numpy()
        z = features[1:3].cpu().numpy()
        u = features[3:].cpu().numpy()
        label = label.cpu().numpy()
        P = calculate_P_t(f, z, u, dataset_config.n_delay_step, dataset_config.dt, 2)
        loss += (P - label) ** 2
    P = calculate_P_t(f, z, u, dataset_config.n_delay_step, dataset_config.dt, 2)
    print(loss)
