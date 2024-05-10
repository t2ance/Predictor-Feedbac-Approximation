from config import DatasetConfig
from main import create_trajectory_dataset

if __name__ == '__main__':
    dataset_config = DatasetConfig(
        recreate_training_dataset=True,
        recreate_testing_dataset=True,
        trajectory=True,
        random_u_type='spline',
        dt=0.1,
        n_dataset=100,
        duration=8,
        delay=3.,
        n_sample_per_dataset=100,
        ic_lower_bound=-1,
        ic_upper_bound=1,
        system_c=1.,
        postprocess=False,
        plot_sample=False
    )
    testing_samples = create_trajectory_dataset(dataset_config, test_points=dataset_config.test_points)
    for feature, p in testing_samples:
        feature = feature.cpu().numpy()
        t = feature[:1]
        z = feature[1:3]
        u = feature[3:]
        p = p.cpu().numpy()


    ...
