import itertools
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal

import numpy as np

import dynamic_systems


@dataclass
class ModelConfig:
    deeponet_n_hidden_size: Optional[int] = field(default=512)
    deeponet_merge_size: Optional[int] = field(default=256)
    deeponet_n_hidden: Optional[int] = field(default=5)
    fno_n_modes_height: Optional[int] = field(default=16)
    fno_hidden_channels: Optional[int] = field(default=32)
    fno_n_layers: Optional[int] = field(default=4)
    fno_end_to_end: Optional[bool] = field(default=True)
    model_name: Optional[Literal['FNO', 'DeepONet', 'FNOTwoStage', 'PIFNO', 'FNOTwoStage2']] = field(default='FNO')

    @property
    def base_path(self):
        return f'./result/{self.model_name}'


@dataclass
class TrainConfig:
    debug: Optional[bool] = field(default=False)
    do_test: Optional[bool] = field(default=False)

    batch_size: Optional[int] = field(default=64)
    learning_rate: Optional[float] = field(default=1e-4)
    weight_decay: Optional[float] = field(default=.0)

    training_ratio: Optional[float] = field(default=0.8)
    log_step: Optional[int] = field(default=10)
    n_epoch: Optional[int] = field(default=100)
    device: Optional[str] = field(default='cuda:0')
    model_save_path: Optional[str] = field(default='./checkpoint')
    load_model: Optional[bool] = field(default=False)

    scheduler_step_size: Optional[int] = field(default=1)
    scheduler_gamma: Optional[float] = field(default=1.)
    scheduler_min_lr: Optional[float] = field(default=0.)
    scheduler_ratio_warmup: Optional[float] = field(default=0.1)
    lr_scheduler_type: Optional[Literal['linear_with_warmup', 'exponential']] = field(default='linear_with_warmup')


@dataclass
class DatasetConfig:
    append_training_dataset: Optional[bool] = field(default=False)
    append_testing_dataset: Optional[bool] = field(default=False)
    delay: Optional[float] = field(default=3.)
    duration: Optional[int] = field(default=8)
    dt: Optional[float] = field(default=0.005)
    test_points: Optional[List[Tuple[float, float]]] = field(
        default_factory=lambda: [(x, y) for x, y in itertools.product(
            np.linspace(-1, 1, 6),
            np.linspace(-1, 1, 6)
        )])
    ic_lower_bound: Optional[float] = field(default=-2)
    ic_upper_bound: Optional[float] = field(default=2)
    n_state: Optional[int] = field(default=2)
    n_sample_per_dataset: Optional[int] = field(default=100)
    n_dataset: Optional[int] = field(default=200)
    recreate_training_dataset: Optional[bool] = field(default=True)
    recreate_testing_dataset: Optional[bool] = field(default=True)
    base_path: Optional[str] = field(default='./datasets')

    data_generation_strategy: Optional[Literal['trajectory', 'random', 'nn']] = field(default='trajectory')
    explicit: Optional[bool] = field(default=False)
    noise_sigma_numerical: Optional[float] = field(default=0.)
    ood_sample_bound: Optional[float] = field(default=0.1)
    system_c: Optional[float] = field(default=1.)
    system_n: Optional[float] = field(default=2.)
    postprocess: Optional[bool] = field(default=False)
    n_plot_sample: Optional[int] = field(default=0)
    filter_ood_sample: Optional[bool] = field(default=True)
    random_u_type: Optional[Literal['line', 'sin', 'exp', 'spline', 'poly', 'sinexp', 'chebyshev']] = field(
        default='spline')

    net_dataset_size: Optional[int] = field(default=1000)
    net_batch_size: Optional[int] = field(default=64)
    net_lr: Optional[float] = field(default=1e-3)
    net_weight_decay: Optional[int] = field(default=0)
    net_n_epoch: Optional[int] = field(default=5000)
    net_type: Optional[Literal['fc', 'fourier', 'chebyshev']] = field(default='fc')

    lamda: Optional[float] = field(default=1.)
    regularization_type: Optional[str] = field(default='total variation')

    fourier_n_mode: Optional[int] = field(default=4)

    chebyshev_n_term: Optional[int] = field(default=4)

    scheduler_step_size: Optional[int] = field(default=1)
    scheduler_gamma: Optional[float] = field(default=1.)
    scheduler_min_lr: Optional[float] = field(default=0.)
    scheduler_ratio_warmup: Optional[float] = field(default=0.1)
    lr_scheduler_type: Optional[Literal['linear_with_warmup', 'exponential']] = field(default='linear_with_warmup')

    @property
    def dataset_base_path(self):
        return f'{self.base_path}/{self.data_generation_strategy}'

    @property
    def training_dataset_file(self):
        return f'{self.dataset_base_path}/train.pkl'

    @property
    def validating_dataset_file(self):
        return f'{self.dataset_base_path}/validate.pkl'

    @property
    def testing_dataset_file(self):
        return f'{self.dataset_base_path}/test.pkl'

    @property
    def n_epoch(self):
        return self.net_n_epoch

    @property
    def system(self) -> dynamic_systems.DynamicSystem:
        return dynamic_systems.DynamicSystem(c=self.system_c, n=self.system_n, delay=self.delay)

    @property
    def ts(self) -> np.ndarray:
        return np.linspace(-self.delay, self.duration, self.n_point)

    @property
    def n_point(self) -> int:
        return self.n_point_delay + self.n_point_duration

    @property
    def n_point_delay(self) -> int:
        return int(round(self.delay / self.dt))

    @property
    def n_point_duration(self) -> int:
        return int(round(self.duration / self.dt))

    def noise(self):
        if self.noise_sigma_numerical == 0:
            return 0
        return np.random.randn() * self.noise_sigma_numerical


if __name__ == '__main__':
    ...
