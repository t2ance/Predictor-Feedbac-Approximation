import itertools
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal

import numpy as np


@dataclass
class ModelConfig:
    deeponet_n_hidden_size: Optional[int] = field(default=512)
    deeponet_merge_size: Optional[int] = field(default=256)
    deeponet_n_hidden: Optional[int] = field(default=5)
    fno_n_modes_height: Optional[int] = field(default=16)
    fno_hidden_channels: Optional[int] = field(default=32)
    fno_n_layers: Optional[int] = field(default=4)
    model_name: Optional[Literal['FNO', 'DeepONet']] = field(default='FNO')

    @property
    def base_path(self):
        return f'./result/{self.model_name}'


@dataclass
class TrainConfig:
    debug: Optional[bool] = field(default=False)
    batch_size: Optional[int] = field(default=64)
    learning_rate: Optional[float] = field(default=1e-4)
    weight_decay: Optional[float] = field(default=.0)
    scheduler_step_size: Optional[int] = field(default=1)
    scheduler_gamma: Optional[float] = field(default=1.)
    scheduler_min_lr: Optional[float] = field(default=0.)
    training_ratio: Optional[float] = field(default=0.8)
    log_step: Optional[float] = field(default=10)
    n_epoch: Optional[int] = field(default=100)
    device: Optional[str] = field(default='cuda')
    model_save_path: Optional[str] = field(default='./checkpoint')
    load_model: Optional[bool] = field(default=False)


@dataclass
class DatasetConfig:
    delay: Optional[float] = field(default=3.)
    duration: Optional[int] = field(default=8)
    dt: Optional[float] = field(default=0.005)
    test_points: Optional[List[Tuple[float, float]]] = field(
        default_factory=lambda: [(x, y) for x, y in itertools.product(
            np.linspace(-1, 1, 11),
            np.linspace(-1, 1, 11)
        )])
    ic_lower_bound: Optional[float] = field(default=0.)
    ic_upper_bound: Optional[float] = field(default=1.)
    n_state: Optional[int] = field(default=2)
    n_sample_per_dataset: Optional[int] = field(default=100)
    n_dataset: Optional[int] = field(default=200)
    recreate_training_dataset: Optional[bool] = field(default=True)
    recreate_testing_dataset: Optional[bool] = field(default=True)
    training_dataset_file: Optional[str] = field(default='./datasets/train.pkl')
    validating_dataset_file: Optional[str] = field(default='./datasets/validate.pkl')
    testing_dataset_file: Optional[str] = field(default='./datasets/test.pkl')
    trajectory: Optional[bool] = field(default=True)
    implicit: Optional[bool] = field(default=False)
    noise_sigma_numerical: Optional[float] = field(default=0.)
    system_c: Optional[float] = field(default=1.)
    system_n: Optional[float] = field(default=2.)
    postprocess: Optional[bool] = field(default=False)
    n_plot_sample: Optional[int] = field(default=0)
    random_u_type: Optional[Literal['line', 'sin', 'exp', 'spline', 'poly', 'sinexp']] = field(default='poly')

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
