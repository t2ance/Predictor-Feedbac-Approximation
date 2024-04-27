from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal

import numpy as np


@dataclass
class ModelConfig:
    deeponet_n_hidden_size: Optional[int] = field(default=512)
    deeponet_n_merge_size: Optional[int] = field(default=256)
    deeponet_n_hidden: Optional[int] = field(default=3)
    fno_n_modes_height: Optional[int] = field(default=8)
    fno_hidden_channels: Optional[int] = field(default=16)
    model_name: Optional[Literal['FNO', 'DeepONet']] = field(default='FNO')

    # model_name = 'DeepONet'

    @property
    def base_path(self):
        return f'./result/{self.model_name}'


@dataclass
class TrainConfig:
    batch_size: Optional[int] = field(default=64)
    learning_rate: Optional[float] = field(default=3e-5)
    weight_decay: Optional[float] = field(default=1e-5)
    training_ratio: Optional[float] = field(default=0.8)
    n_epoch: Optional[int] = field(default=100)
    device: Optional[str] = field(default='cuda')


@dataclass
class DatasetConfig:
    delay: Optional[float] = field(default=3.)
    duration: Optional[int] = field(default=12)
    dt: Optional[float] = field(default=0.005)
    test_points: Optional[List[Tuple[float, float]]] = \
        field(default_factory=lambda: [(0, 1), (0, 0.5), (0.5, 0), (0.5, 0.5), (1, 0), (1, 1)])
    n_state: Optional[int] = field(default=2)
    n_sample_per_dataset: Optional[int] = field(default=200)
    n_dataset: Optional[int] = field(default=100)
    recreate_dataset: Optional[bool] = field(default=True)
    dataset_file: Optional[str] = field(default='./datasets/dataset.pkl')
    trajectory: Optional[bool] = field(default=True)
    implicit: Optional[bool] = field(default=True)

    @property
    def ts(self) -> np.ndarray:
        return np.linspace(-self.delay, self.duration, self.n_point)

    @property
    def n_point(self) -> int:
        return int(round((self.duration + self.delay) / self.dt))

    @property
    def n_delay_point(self) -> int:
        return int(self.delay / self.dt)


if __name__ == '__main__':
    ...
