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
    n_layer: Optional[int] = field(default=4)
    ffn_layer_width: Optional[int] = field(default=8)
    fno_end_to_end: Optional[bool] = field(default=True)
    model_name: Optional[Literal['FFN', 'FNO', 'DeepONet', 'FNOTwoStage', 'PIFNO']] = field(default='FNO')
    system: Optional[str] = field(default='s1')

    @property
    def base_path(self):
        return f'./{self.system}/result/{self.model_name}'


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
    device: Optional[str] = field(default='cuda')
    load_model: Optional[bool] = field(default=False)

    scheduler_step_size: Optional[int] = field(default=1)
    scheduler_gamma: Optional[float] = field(default=1.)
    scheduler_min_lr: Optional[float] = field(default=0.)
    scheduler_ratio_warmup: Optional[float] = field(default=0.02)
    lr_scheduler_type: Optional[Literal['linear_with_warmup', 'exponential']] = field(default='linear_with_warmup')

    # conformal prediction
    alpha: Optional[float] = field(default=0.1)
    # adversarial training
    adversarial_epsilon: Optional[float] = field(default=0.0)
    # scheduled sampling probability
    scheduled_sampling_type: Optional[Literal['exponential', 'inverse sigmoid', 'linear']] = field(
        default='linear')
    scheduled_sampling_p: Optional[float] = field(default=None)
    scheduled_sampling_k: Optional[float] = field(default=0.01)
    scheduled_sampling_warm_start: Optional[int] = field(default=0)
    system: Optional[str] = field(default='s1')

    training_type: Optional[Literal['offline', 'switching', 'scheduled sampling']] = field(default='scheduled sampling')

    @property
    def model_save_path(self):
        return f'./{self.system}/checkpoint'

    def set_scheduled_sampling_p(self, epoch):
        if epoch < self.scheduled_sampling_warm_start:
            self.scheduled_sampling_p = 1
        else:
            epoch -= self.scheduled_sampling_warm_start
            n_epoch = self.n_epoch - self.scheduled_sampling_warm_start
            if self.scheduled_sampling_type == 'exponential':
                # k is a constant that controls the decay rate. Adjust it based on your requirements.
                self.scheduled_sampling_p = np.exp(-self.scheduled_sampling_k * epoch)
            elif self.scheduled_sampling_type == 'inverse sigmoid':
                # k controls how steep the sigmoid is. x0 shifts the sigmoid along the x-axis.
                self.scheduled_sampling_p = 1 / (1 + np.exp(self.scheduled_sampling_k * (epoch - n_epoch / 2)))
            elif self.scheduled_sampling_type == 'linear':
                # Here k is the total number of epochs after which the value reaches 0.
                self.scheduled_sampling_p = max(0, 1 - epoch / n_epoch)
            else:
                raise ValueError("Unsupported scheduled_sampling_type")


@dataclass
class DatasetConfig:
    append_training_dataset: Optional[bool] = field(default=False)
    append_testing_dataset: Optional[bool] = field(default=False)
    delay: Optional[float] = field(default=3.)
    duration: Optional[int] = field(default=8)
    dt: Optional[float] = field(default=0.125)
    integral_method: Optional[Literal['rectangle', 'trapezoidal', 'simpson', 'eular', 'successive']] = field(
        default='successive')
    successive_approximation_n_iteration: Optional[int] = field(default=1)
    ic_lower_bound: Optional[float] = field(default=-2)
    ic_upper_bound: Optional[float] = field(default=2)
    n_sample_per_dataset: Optional[int] = field(default=100)
    n_dataset: Optional[int] = field(default=200)
    recreate_training_dataset: Optional[bool] = field(default=True)
    recreate_testing_dataset: Optional[bool] = field(default=True)

    data_generation_strategy: Optional[Literal['trajectory', 'random', 'nn']] = field(default='trajectory')
    z_u_p_pair: Optional[bool] = field(default=True)
    noise_epsilon: Optional[float] = field(default=0.)
    ood_sample_bound: Optional[float] = field(default=0.1)
    system_c: Optional[float] = field(default=1.)
    system_n: Optional[float] = field(default=2.)
    postprocess: Optional[bool] = field(default=False)
    n_plot_sample: Optional[int] = field(default=0)
    filter_ood_sample: Optional[bool] = field(default=False)
    random_u_type: Optional[Literal['line', 'sin', 'exp', 'spline', 'poly', 'sinexp', 'chebyshev', 'sparse']] = field(
        default='spline')
    n_sample_sparse: Optional[int] = field(default=0)
    epsilon: Optional[float] = field(default=0)
    n_augment: Optional[int] = field(default=0)

    net_dataset_size: Optional[int] = field(default=1000)
    net_batch_size: Optional[int] = field(default=64)
    net_lr: Optional[float] = field(default=1e-3)
    net_weight_decay: Optional[int] = field(default=0)
    net_n_epoch: Optional[int] = field(default=5000)
    net_type: Optional[Literal['fc', 'fourier', 'chebyshev', 'bspline']] = field(default='fc')
    load_net: Optional[bool] = field(default=False)

    lamda: Optional[float] = field(default=1.)
    regularization_type: Optional[str] = field(default='total variation')

    fourier_n_mode: Optional[int] = field(default=4)

    chebyshev_n_term: Optional[int] = field(default=4)
    bspline_n_knot: Optional[int] = field(default=4)
    bspline_degree: Optional[int] = field(default=1)

    scheduler_step_size: Optional[int] = field(default=1)
    scheduler_gamma: Optional[float] = field(default=1.)
    scheduler_min_lr: Optional[float] = field(default=0.)
    scheduler_ratio_warmup: Optional[float] = field(default=0.1)
    lr_scheduler_type: Optional[Literal['linear_with_warmup', 'exponential']] = field(default='linear_with_warmup')

    system_: Optional[str] = field(default='s1')

    random_test: Optional[bool] = field(default=True)
    random_test_points = None

    @property
    def n_sample(self):
        if self.n_sample_per_dataset < 0:
            return self.n_dataset * self.n_point_duration
        else:
            return self.n_dataset * self.n_sample_per_dataset

    @property
    def base_path(self):
        return f'./{self.system_}/datasets'

    @property
    def test_points(self) -> List[Tuple]:
        if self.system_ == 's1':
            return [(x, y) for x, y in itertools.product(
                np.linspace(-1, 1, 6),
                np.linspace(-1, 1, 6)
            )]
        elif self.system_ == 's2':
            return [(x, y, z) for x, y, z in itertools.product(
                np.linspace(-0.3, 0.3, 3),
                np.linspace(-0.3, 0.3, 3),
                np.linspace(-0.3, 0.3, 3)
            )]
        elif self.system_ == 's3':
            return [(x, y, z, w) for x, y, z, w in itertools.product(
                np.linspace(-0.2, 0.2, 2),
                np.linspace(-0.2, 0.2, 2),
                np.linspace(-0.2, 0.2, 2),
                np.linspace(-0.2, 0.2, 2)
            )]
        elif self.system_ == 's4':
            return [(x, y) for x, y in itertools.product(
                np.linspace(-1, 1, 6),
                np.linspace(-1, 1, 6),
            )]
        elif self.system_ == 's5':
            bound = 0.2
            if self.random_test:
                if self.random_test_points is None:
                    self.random_test_points = [tuple((np.random.random(14) * bound).tolist()) for _ in range(10)]
                return self.random_test_points
            return list(itertools.product(
                np.linspace(-bound, bound, 6),
                np.linspace(-bound, bound, 6),
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0]
            ))
        else:
            raise NotImplementedError()

    @property
    def n_state(self):
        return self.system.n_state

    @property
    def dataset_base_path(self):
        base_path = f'{self.base_path}/{self.data_generation_strategy}'
        if self.data_generation_strategy == 'nn':
            return f'{base_path}/{self.net_type}'
        else:
            return base_path

    @property
    def training_dataset_file(self):
        return f'{self.dataset_base_path}/train.pt'

    @property
    def validating_dataset_file(self):
        return f'{self.dataset_base_path}/validate.pt'

    @property
    def testing_dataset_file(self):
        return f'{self.dataset_base_path}/test.pt'

    @property
    def n_epoch(self):
        return self.net_n_epoch

    @property
    def system(self):
        if self.system_ == 's1':
            return dynamic_systems.DynamicSystem1(c=self.system_c, n=self.system_n, delay=self.delay)
        elif self.system_ == 's2':
            return dynamic_systems.DynamicSystem2(delay=self.delay)
        elif self.system_ == 's3':
            return dynamic_systems.InvertedPendulum(delay=self.delay)
        elif self.system_ == 's4':
            return dynamic_systems.VanDerPolOscillator(delay=self.delay)
        elif self.system_ == 's5':
            return dynamic_systems.Baxter(delay=self.delay)
        else:
            raise NotImplementedError()

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
        if self.noise_epsilon == 0:
            return 0
        return np.random.randn() * self.noise_epsilon


def get_config(system_, n_iteration=None, duration=None, delay=None):
    if system_ == 's1':
        dataset_config = DatasetConfig(recreate_training_dataset=True, recreate_testing_dataset=False,
                                       data_generation_strategy='trajectory', delay=1,
                                       duration=8, dt=0.02, n_dataset=250, n_sample_per_dataset=-1, n_plot_sample=20,
                                       system_n=2, system_c=1, ic_lower_bound=-2, ic_upper_bound=2,
                                       successive_approximation_n_iteration=3)
        train_config = TrainConfig(learning_rate=1e-3, training_ratio=0.8, n_epoch=250, batch_size=64,
                                   weight_decay=1e-2, log_step=-1, lr_scheduler_type='exponential',
                                   scheduler_gamma=0.97, scheduler_step_size=1, scheduler_min_lr=1e-5, debug=False,
                                   do_test=False, load_model=False)
        model_config = ModelConfig(model_name='FFN', n_layer=5, fno_n_modes_height=32, fno_hidden_channels=32)
    elif system_ == 's2':
        dataset_config = DatasetConfig(recreate_training_dataset=True, data_generation_strategy='trajectory', delay=1,
                                       duration=8, dt=0.05, n_dataset=100, n_sample_per_dataset=-1, n_plot_sample=20,
                                       ic_lower_bound=-1, ic_upper_bound=1, successive_approximation_n_iteration=5)
        train_config = TrainConfig(learning_rate=1e-3, training_ratio=0.8, n_epoch=2000, batch_size=64,
                                   weight_decay=1e-3, log_step=-1, do_test=False, scheduled_sampling_warm_start=500,
                                   scheduled_sampling_type='linear', scheduled_sampling_k=1e-2)
        model_config = ModelConfig(model_name='FFN', n_layer=5, fno_n_modes_height=32, fno_hidden_channels=64,
                                   ffn_layer_width=8)
    elif system_ == 's3':
        dataset_config = DatasetConfig(recreate_training_dataset=False, data_generation_strategy='trajectory',
                                       delay=0.3, duration=8, dt=0.05, n_dataset=250, n_sample_per_dataset=-1,
                                       n_plot_sample=20, ic_lower_bound=-1, ic_upper_bound=1,
                                       successive_approximation_n_iteration=5)
        train_config = TrainConfig(learning_rate=1e-3, training_ratio=0.8, n_epoch=2000, batch_size=64,
                                   weight_decay=1e-3, log_step=-1, do_test=False, scheduled_sampling_warm_start=500,
                                   scheduled_sampling_type='linear', scheduled_sampling_k=1e-2)
        model_config = ModelConfig(n_layer=5, fno_n_modes_height=32, fno_hidden_channels=64,
                                   ffn_layer_width=8)
    elif system_ == 's4':
        dataset_config = DatasetConfig(recreate_training_dataset=False, data_generation_strategy='trajectory', delay=1,
                                       duration=8, dt=0.05, n_dataset=200, n_sample_per_dataset=-1, n_plot_sample=20,
                                       ic_lower_bound=-2, ic_upper_bound=2, successive_approximation_n_iteration=5)
        model_config = ModelConfig(model_name='FFN', n_layer=4, fno_n_modes_height=8, fno_hidden_channels=16)
        train_config = TrainConfig(learning_rate=1e-3, training_ratio=0.8, n_epoch=1000, batch_size=64,
                                   weight_decay=1e-2, log_step=-1, lr_scheduler_type='exponential', alpha=0.01,
                                   load_model=False, do_test=False, scheduled_sampling_type='inverse sigmoid',
                                   scheduled_sampling_k=1e-2)
    elif system_ == 's5':
        dataset_config = DatasetConfig(recreate_training_dataset=True, data_generation_strategy='trajectory',
                                       delay=0.5, duration=12, dt=0.02, n_dataset=200, n_sample_per_dataset=-1,
                                       n_plot_sample=20, ic_lower_bound=-0.2, ic_upper_bound=0.2,
                                       successive_approximation_n_iteration=5)
        model_config = ModelConfig(model_name='FNO', n_layer=5, fno_n_modes_height=32, fno_hidden_channels=64)
        train_config = TrainConfig(learning_rate=1e-3, training_ratio=0.8, n_epoch=3000, batch_size=64,
                                   weight_decay=1e-4, log_step=-1, lr_scheduler_type='exponential', alpha=0.01,
                                   scheduled_sampling_warm_start=0, load_model=False, do_test=False,
                                   scheduled_sampling_type='linear', scheduled_sampling_k=1e-2)
    else:
        raise NotImplementedError()
    if n_iteration is not None:
        dataset_config.successive_approximation_n_iteration = n_iteration
    if delay is not None:
        dataset_config.delay = delay
    if duration is not None:
        dataset_config.duration = duration
    dataset_config.system_ = system_
    model_config.system = system_
    train_config.system = system_
    return dataset_config, model_config, train_config


if __name__ == '__main__':
    ...
