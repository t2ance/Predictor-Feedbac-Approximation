import os
import pickle
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Literal

import numpy as np
import torch
import wandb

import dynamic_systems
from dynamic_systems import ConstantDelay, TimeVaryingDelay
from utils import load_model

root_dir = '/data/hdc/peijia/OperatorLearning'


@dataclass
class ModelConfig:
    z2u: bool = field(default=False)
    deeponet_hidden_size: Optional[int] = field(default=64)
    deeponet_n_layer: Optional[int] = field(default=5)
    fno_n_modes_height: Optional[int] = field(default=16)
    fno_hidden_channels: Optional[int] = field(default=32)
    fno_n_layer: Optional[int] = field(default=4)
    ffn_n_layer: Optional[int] = field(default=4)
    ffn_layer_width: Optional[int] = field(default=8)

    gru_n_layer: Optional[int] = field(default=4)
    gru_hidden_size: Optional[int] = field(default=64)
    gru_layer_width: Optional[int] = field(default=8)
    lstm_n_layer: Optional[int] = field(default=4)
    lstm_hidden_size: Optional[int] = field(default=64)
    lstm_layer_width: Optional[int] = field(default=8)
    model_name: Optional[Literal['FFN', 'FNO', 'DeepONet', 'GRU', 'LSTM']] = field(default='FNO')
    model_version: Optional[str] = field(default='latest')
    init_type: Optional[str] = field(default='xavier')

    system: Optional[str] = field(default='s1')

    @property
    def base_path(self):
        return f'{root_dir}/{self.system}/result/{self.model_name}'

    def save_model(self, run, model, model_name: str = None):
        if model_name is None:
            model_name = model.name()
        print('save model as', model_name)
        art_name = f'{model_name}-{self.system}'
        model_artifact = wandb.Artifact(
            art_name, type="model",
            description=f"{model_name} model for system {self.system}", metadata=self.__dict__
        )

        model_save_name = f"{root_dir}/{art_name}.pth"
        torch.save(model.state_dict(), model_save_name)
        model_artifact.add_file(model_save_name)
        # wandb.save(model_save_name)
        logged_artifact = run.log_artifact(model_artifact)
        logged_artifact.wait()
        art_version = logged_artifact.version
        print(f'Logged {art_name} as version {art_version}')
        os.remove(model_save_name)
        return art_version

    def load_model(self, run, model, model_name: str = None, version: str = None):
        if version is None:
            version = self.model_version
        if model_name is None:
            model_name = model.name()
        model_artifact = run.use_artifact(f"{model_name}-{self.system}:{version}")

        model_dir = model_artifact.download(root=root_dir)
        model_path = os.path.join(model_dir, f"{model_name}-{self.system}.pth")

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        return state_dict

    def get_model(self, run, train_config, dataset_config, version: str = None):
        if version is None:
            version = self.model_version
        model_artifact = run.use_artifact(f"{self.model_name}-{self.system}:{version}")
        model_name = self.model_name
        metadata = model_artifact.metadata
        model_dir = model_artifact.download(root=root_dir)
        print(f'Loaded model from wandb to {model_dir}')
        try:
            model_path = os.path.join(model_dir, f"{self.model_name}-{self.system}.pth")
            state_dict = torch.load(model_path)
        except:
            model_path = os.path.join(model_dir, f"model.pth")
            state_dict = torch.load(model_path)
        if 'FNO' in model_name:
            self.fno_n_layer = metadata['fno_n_layer']
            self.fno_n_modes_height = metadata['fno_n_modes_height']
            self.fno_hidden_channels = metadata['fno_hidden_channels']
        if 'DeepONet' in model_name:
            self.deeponet_hidden_size = metadata['deeponet_hidden_size']
            self.deeponet_n_layer = metadata['deeponet_n_layer']
        if 'GRU' in model_name:
            self.gru_n_layer = metadata['gru_n_layer']
            self.gru_layer_width = metadata['gru_layer_width']
            self.gru_hidden_size = metadata['gru_hidden_size']
        if 'LSTM' in model_name:
            self.lstm_n_layer = metadata['lstm_n_layer']
            self.lstm_layer_width = metadata['lstm_layer_width']
            self.lstm_hidden_size = metadata['lstm_hidden_size']

        model, n_params = load_model(train_config, self, dataset_config, model_name=self.model_name, n_param_out=True)
        model.load_state_dict(state_dict)
        return model, n_params


@dataclass
class TrainConfig:
    debug: Optional[bool] = field(default=False)
    do_testing: Optional[bool] = field(default=False)
    do_training: Optional[bool] = field(default=True)

    batch_size: Optional[int] = field(default=64)
    batch_size2_: Optional[int] = field(default=-1)
    learning_rate: Optional[float] = field(default=1e-4)
    weight_decay: Optional[float] = field(default=.0)

    training_ratio: Optional[float] = field(default=0.8)
    log_step: Optional[int] = field(default=10)
    n_epoch: Optional[int] = field(default=100)
    n_epoch2_: Optional[int] = field(default=-1)
    device: Optional[str] = field(default='cuda')
    load_model: Optional[bool] = field(default=False)

    scheduler_step_size: Optional[int] = field(default=1)
    scheduler_gamma: Optional[float] = field(default=.99)
    scheduler_min_lr: Optional[float] = field(default=0.)
    scheduler_min_lr2_: Optional[float] = field(default=-1)
    scheduler_ratio_warmup: Optional[float] = field(default=0.02)
    scheduled_sampling_frequency: Optional[int] = field(default=10)
    lr_scheduler_type: Optional[
        Literal['linear_with_warmup', 'cosine_annealing_with_warmup', 'exponential', 'none']] = field(
        default='linear_with_warmup')

    # conformal prediction
    uq_alpha: Optional[float] = field(default=0.1)
    uq_gamma: Optional[float] = field(default=0.01)
    uq_adaptive: Optional[bool] = field(default=True)
    uq_type: Optional[Literal['conformal prediction', 'gaussian process']] = field(default='conformal prediction')
    uq_switching_type: Optional[Literal['switching', 'alternating']] = field(default='switching')
    uq_non_delay: Optional[bool] = field(default=False)
    uq_warmup: Optional[bool] = field(default=True)

    # adversarial training
    adversarial_epsilon: Optional[float] = field(default=0.0)
    # scheduled sampling probability
    scheduled_sampling_type: Optional[Literal['exponential', 'inverse sigmoid', 'linear']] = field(
        default='linear')
    scheduled_sampling_p: Optional[float] = field(default=None)
    scheduled_sampling_k: Optional[float] = field(default=0.01)
    scheduled_sampling_warm_start: Optional[int] = field(default=0)
    system: Optional[str] = field(default='s1')

    # train FNO-GRU jointly or not
    two_stage: Optional[bool] = field(default=True)
    train_first_stage: Optional[bool] = field(default=False)
    # freeze FNO in FNO-GRU or not
    freeze_ffn: Optional[bool] = field(default=False)
    # let GRU in FNO-GRU to only model the residual (i.e. x = gru(fno(x))+x) or not
    residual: Optional[bool] = field(default=False)
    # initilize the RNN to zero (usually combined with residual x)
    zero_init: Optional[bool] = field(default=False)
    use_t: Optional[bool] = field(default=False)
    # use auxiliary loss or not
    auxiliary_loss: Optional[bool] = field(default=False)
    training_type: Optional[Literal['offline', 'switching', 'scheduled sampling']] = field(default='sequence')

    @property
    def batch_size2(self):
        return self.batch_size2_ if self.batch_size2_ != -1 else self.batch_size

    @property
    def scheduler_min_lr2(self):
        return self.scheduler_min_lr2_ if self.scheduler_min_lr2_ != -1 else self.scheduler_min_lr

    @property
    def n_epoch2(self):
        return self.n_epoch2_ if self.n_epoch2_ != -1 else self.n_epoch

    @property
    def model_save_path(self):
        return f'{root_dir}/{self.system}/checkpoint'

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
    delay: dynamic_systems.Delay = field(default=dynamic_systems.ConstantDelay(3))
    duration: Optional[int] = field(default=8)
    dt: Optional[float] = field(default=0.125)
    integral_method: Optional[
        Literal['rectangle', 'trapezoidal', 'simpson', 'eular', 'successive', 'successive adaptive']] = field(
        default='successive adaptive')

    successive_approximation_n_iteration: Optional[int] = field(default=1)
    successive_approximation_threshold: Optional[float] = field(default=1e-7)

    ic_lower_bound: Optional[float] = field(default=-2)
    ic_upper_bound: Optional[float] = field(default=2)
    n_sample_per_dataset: Optional[int] = field(default=100)
    n_training_dataset: Optional[int] = field(default=200)
    n_validation_dataset: Optional[int] = field(default=200)
    recreate_dataset: Optional[bool] = field(default=True)

    data_generation_strategy: Optional[Literal['trajectory', 'random', 'nn']] = field(default='trajectory')
    noise_epsilon: Optional[float] = field(default=0.)
    ood_sample_bound: Optional[float] = field(default=0.1)
    system_c: Optional[float] = field(default=1.)
    system_n: Optional[float] = field(default=2.)
    baxter_dof: Optional[int] = field(default=2)
    baxter_f: Optional[float] = field(default=0.1)
    baxter_alpha: Optional[float] = field(default=1)
    baxter_beta: Optional[float] = field(default=1)
    baxter_magnitude: Optional[float] = field(default=0.2)
    baxter_q_des_type: Optional[str] = field(default='sine')
    scheduler_step_size: Optional[int] = field(default=1)
    scheduler_gamma: Optional[float] = field(default=1.)
    scheduler_min_lr: Optional[float] = field(default=0.)
    scheduler_ratio_warmup: Optional[float] = field(default=0.1)
    lr_scheduler_type: Optional[Literal['linear_with_warmup', 'exponential']] = field(default='linear_with_warmup')

    n_step: int = field(default=1)
    n_test_point: int = field(default=25)
    system_: Optional[str] = field(default='s1')

    random_test: Optional[bool] = field(default=True)
    random_test_points = None
    random_test_upper_bound: Optional[float] = field(default=1.)
    random_test_lower_bound: Optional[float] = field(default=0.)

    dataset_version: Optional[str] = field(default='latest')

    @property
    def base_path(self):
        return f'{root_dir}/{self.system_}/datasets'

    def get_test_points(self, n_point=1, lower_bound=None, upper_bound=None):
        if lower_bound is None:
            lower_bound = self.random_test_lower_bound
        if upper_bound is None:
            upper_bound = self.random_test_upper_bound
        print(
            f'Getting test points from dataset config, with lower_bound = {lower_bound} and upper_bound = {upper_bound}')
        state = np.random.RandomState(seed=0)
        return [
            tuple((state.uniform(lower_bound, upper_bound, self.system.n_state)).tolist()) for _ in range(n_point)
        ]

    @property
    def test_points(self) -> List[Tuple]:
        if self.random_test_points is None:
            self.random_test_points = self.get_test_points(self.n_test_point)
        return self.random_test_points

    @property
    def n_state(self):
        return self.system.n_state

    @property
    def system(self):
        if self.system_ == 's1':
            return dynamic_systems.DynamicSystem1(c=self.system_c, n=self.system_n)
        elif self.system_ == 's2':
            return dynamic_systems.DynamicSystem2()
        elif self.system_ == 's3':
            return dynamic_systems.InvertedPendulum()
        elif self.system_ == 's4':
            return dynamic_systems.VanDerPolOscillator()
        elif self.system_ == 's5' or self.system_ == 's8' or self.system_ == 's10' or self.system_ == 's11' or self.system_ == 's12':
            return dynamic_systems.Baxter(alpha=self.baxter_alpha, beta=self.baxter_beta, dof=self.baxter_dof,
                                          f=self.baxter_f, magnitude=self.baxter_magnitude,
                                          q_des_type=self.baxter_q_des_type)
        elif self.system_ == 's6':
            return dynamic_systems.DynamicSystem3()
        elif self.system_ == 's7' or self.system_ == 's9':
            return dynamic_systems.Unicycle()
        else:
            raise NotImplementedError('Unknown system', self.system_)

    @property
    def ts(self) -> np.ndarray:
        return np.linspace(-self.delay(0), self.duration - self.dt, self.n_point)

    @property
    def n_point(self) -> int:
        return self.n_point_start() + self.n_point_duration

    def n_point_start(self) -> int:
        return self.n_point_delay(0)

    def n_point_delay(self, t) -> int:
        return int(round(self.delay(t) / self.dt))

    def max_n_point_delay(self) -> int:
        return int(round(self.delay.max_delay() / self.dt))

    @property
    def n_point_duration(self) -> int:
        return int(round(self.duration / self.dt))

    def noise(self):
        if self.noise_epsilon == 0:
            return 0
        return np.random.randn() * self.noise_epsilon

    def load_dataset(self, run, resize: bool = True, version=None):
        if version is None:
            version = self.dataset_version

        def read(dataset_dir, split):
            with open(os.path.join(dataset_dir, split + ".pkl"), mode="rb") as file:
                return pickle.load(file)

        data = run.use_artifact(f'{self.system_}:{version}')
        dataset = data.download(root=root_dir)

        training_dataset = read(dataset, "training")
        validation_dataset = read(dataset, "validation")
        if resize:
            return training_dataset[:self.n_training_dataset], validation_dataset[:self.n_validation_dataset]
        else:
            return training_dataset, validation_dataset

    def save_dataset(self, run, training_results, validating_results):
        datasets = [training_results, validating_results]
        names = ["training", "validation"]

        art = wandb.Artifact(
            self.system_, type="dataset",
            description=f"Dataset for system {self.system_}, used for training in sequence model",
            metadata={'sizes': [len(dataset) for dataset in datasets]}
        )

        for name, data in zip(names, datasets):
            with art.new_file(name + ".pkl", mode="wb") as file:
                pickle.dump(data, file)

        run.log_artifact(art)


def get_config(system_, n_iteration=None, duration=None, delay=None, model_name=None, z2u=None):
    if system_ == 's1':
        dataset_config = DatasetConfig(recreate_dataset=False, data_generation_strategy='trajectory',
                                       delay=ConstantDelay(1), duration=6, dt=0.1, n_training_dataset=900,
                                       n_validation_dataset=100, n_sample_per_dataset=-1, ic_lower_bound=0,
                                       ic_upper_bound=1, random_test_lower_bound=0, random_test_upper_bound=1)
        model_config = ModelConfig(model_name='GRU')
        train_config = TrainConfig(learning_rate=1e-3, training_ratio=0.8, n_epoch=750, batch_size=128,
                                   do_training=True, do_testing=False, load_model=False,
                                   weight_decay=1e-3, log_step=-1, lr_scheduler_type='exponential',
                                   scheduler_gamma=0.97, scheduler_step_size=1, scheduler_min_lr=1e-5)
    elif system_ == 's2':
        dataset_config = DatasetConfig(recreate_dataset=False, data_generation_strategy='trajectory',
                                       delay=ConstantDelay(1), duration=8, dt=0.05, n_training_dataset=900,
                                       n_validation_dataset=100,
                                       n_sample_per_dataset=-1, ic_lower_bound=-1, ic_upper_bound=1,
                                       successive_approximation_n_iteration=5)
        train_config = TrainConfig(learning_rate=1e-3, training_ratio=0.8, n_epoch=2000, batch_size=64,
                                   weight_decay=1e-3, log_step=-1, do_testing=False, scheduled_sampling_warm_start=500,
                                   scheduled_sampling_type='linear', scheduled_sampling_k=1e-2)
        model_config = ModelConfig(model_name='FFN', fno_n_layer=5, fno_n_modes_height=32, fno_hidden_channels=64,
                                   ffn_layer_width=8)
    elif system_ == 's3':
        dataset_config = DatasetConfig(recreate_dataset=False, data_generation_strategy='trajectory',
                                       delay=ConstantDelay(0.3), duration=8, dt=0.05, n_training_dataset=900,
                                       n_validation_dataset=100,
                                       n_sample_per_dataset=-1, ic_lower_bound=-1, ic_upper_bound=1)
        train_config = TrainConfig(learning_rate=1e-3, training_ratio=0.8, n_epoch=100, batch_size=64,
                                   weight_decay=1e-3, log_step=-1, do_testing=False, scheduled_sampling_warm_start=500,
                                   scheduled_sampling_type='linear', scheduled_sampling_k=1e-2)
        model_config = ModelConfig(model_name='FNO', fno_n_layer=5, fno_n_modes_height=32, fno_hidden_channels=64,
                                   ffn_layer_width=8)
    elif system_ == 's4':
        dataset_config = DatasetConfig(recreate_dataset=False, data_generation_strategy='trajectory',
                                       delay=ConstantDelay(1), duration=8, dt=0.05, n_training_dataset=900,
                                       n_validation_dataset=100,
                                       n_sample_per_dataset=-1, ic_lower_bound=-2, ic_upper_bound=2,
                                       successive_approximation_n_iteration=5)
        model_config = ModelConfig(model_name='FFN', fno_n_layer=4, fno_n_modes_height=8, fno_hidden_channels=16)
        train_config = TrainConfig(learning_rate=1e-3, training_ratio=0.8, n_epoch=1000, batch_size=64,
                                   weight_decay=1e-2, log_step=-1, lr_scheduler_type='exponential', uq_alpha=0.01,
                                   load_model=False, do_testing=False, scheduled_sampling_type='inverse sigmoid',
                                   scheduled_sampling_k=1e-2)
    elif system_ == 's5':
        dataset_config = DatasetConfig(recreate_dataset=True, data_generation_strategy='trajectory',
                                       delay=ConstantDelay(.5), duration=8, dt=0.02, n_training_dataset=900,
                                       n_validation_dataset=100, n_sample_per_dataset=-1, baxter_dof=2,
                                       ic_lower_bound=0, ic_upper_bound=1, random_test_lower_bound=0,
                                       random_test_upper_bound=1)
        model_config = ModelConfig(model_name='FNO')
        train_config = TrainConfig(learning_rate=3e-4, training_ratio=0.8, n_epoch=750, batch_size=64,
                                   weight_decay=1e-3, log_step=-1, lr_scheduler_type='exponential', uq_alpha=0.01,
                                   scheduled_sampling_warm_start=0, scheduled_sampling_type='linear',
                                   scheduled_sampling_k=1e-2, scheduler_min_lr=1e-5)
    elif system_ == 's6':
        dataset_config = DatasetConfig(recreate_dataset=False, data_generation_strategy='trajectory',
                                       delay=ConstantDelay(.5), duration=32, dt=0.01, n_training_dataset=900,
                                       n_validation_dataset=100, n_sample_per_dataset=-1, ic_lower_bound=-0.5,
                                       ic_upper_bound=0.5)
        model_config = ModelConfig(model_name='FNO', fno_n_layer=3, fno_n_modes_height=16, fno_hidden_channels=16)
        train_config = TrainConfig(learning_rate=1e-4, training_ratio=0.8, n_epoch=2000, batch_size=128,
                                   weight_decay=1e-3, log_step=-1, lr_scheduler_type='none', uq_alpha=0.01,
                                   scheduled_sampling_warm_start=0, scheduled_sampling_type='linear',
                                   scheduled_sampling_k=1e-2)
    elif system_ == 's7':
        dataset_config = DatasetConfig(recreate_dataset=False, data_generation_strategy='trajectory',
                                       delay=TimeVaryingDelay(), duration=8, dt=0.01, n_training_dataset=900,
                                       n_validation_dataset=100, n_sample_per_dataset=-1, ic_lower_bound=-0.5,
                                       ic_upper_bound=0.5)
        model_config = ModelConfig(model_name='FFN')
        train_config = TrainConfig(learning_rate=1e-4, training_ratio=0.8, n_epoch=750, batch_size=64,
                                   weight_decay=1e-3, log_step=-1, lr_scheduler_type='exponential',
                                   scheduler_min_lr=1e-5)
    elif system_ == 's8':
        dataset_config = DatasetConfig(recreate_dataset=False, data_generation_strategy='trajectory',
                                       delay=ConstantDelay(.5), duration=8, dt=0.02, n_training_dataset=100,
                                       n_validation_dataset=1, n_sample_per_dataset=-1, baxter_dof=5, ic_lower_bound=0,
                                       ic_upper_bound=1, random_test_lower_bound=0, random_test_upper_bound=1)
        model_config = ModelConfig(model_name='FNO')
        train_config = TrainConfig(learning_rate=3e-4, training_ratio=0.8, n_epoch=750, batch_size=64,
                                   weight_decay=1e-3, log_step=-1, lr_scheduler_type='exponential', uq_alpha=0.01,
                                   scheduled_sampling_warm_start=0, scheduled_sampling_type='linear',
                                   scheduled_sampling_k=1e-2, scheduler_min_lr=1e-5)
    elif system_ == 's9':
        dataset_config = DatasetConfig(recreate_dataset=False, data_generation_strategy='trajectory',
                                       delay=TimeVaryingDelay(), duration=8, dt=0.004, n_training_dataset=100,
                                       n_validation_dataset=1, n_sample_per_dataset=-1, ic_lower_bound=0,
                                       ic_upper_bound=1, random_test_lower_bound=0, random_test_upper_bound=1)
        model_config = ModelConfig(model_name='FFN')
        train_config = TrainConfig(learning_rate=1e-4, training_ratio=0.8, n_epoch=750, batch_size=64,
                                   weight_decay=1e-3, log_step=-1, lr_scheduler_type='exponential',
                                   scheduler_min_lr=1e-5)
    elif system_ == 's10':
        dataset_config = DatasetConfig(recreate_dataset=False, data_generation_strategy='trajectory',
                                       delay=ConstantDelay(.5), duration=8, dt=0.02, n_training_dataset=100,
                                       n_validation_dataset=1, n_sample_per_dataset=-1, baxter_dof=5, baxter_f=0.5,
                                       baxter_magnitude=0.1, ic_lower_bound=0, ic_upper_bound=1,
                                       random_test_lower_bound=0, random_test_upper_bound=1)
        model_config = ModelConfig(model_name='FNO')
        train_config = TrainConfig(learning_rate=3e-4, training_ratio=0.8, n_epoch=750, batch_size=64,
                                   weight_decay=1e-3, log_step=-1, lr_scheduler_type='exponential', uq_alpha=0.01,
                                   scheduled_sampling_warm_start=0, scheduled_sampling_type='linear',
                                   scheduled_sampling_k=1e-2, scheduler_min_lr=1e-5)
    elif system_ == 's11':
        dataset_config = DatasetConfig(recreate_dataset=True, data_generation_strategy='trajectory', system_='s11',
                                       delay=ConstantDelay(0.5), duration=8, dt=0.02, n_training_dataset=25,
                                       n_validation_dataset=1, n_sample_per_dataset=-1, baxter_dof=5, baxter_f=1,
                                       baxter_magnitude=0.1, baxter_alpha=1, baxter_beta=2, ic_lower_bound=0,
                                       ic_upper_bound=1, random_test_lower_bound=0, random_test_upper_bound=1)
        model_config = ModelConfig(model_name='FNO')
        train_config = TrainConfig(learning_rate=3e-4, training_ratio=0.8, n_epoch=200, batch_size=256,
                                   weight_decay=1e-3, log_step=-1, lr_scheduler_type='exponential', uq_alpha=0.01,
                                   scheduled_sampling_warm_start=0, scheduled_sampling_type='linear',
                                   scheduled_sampling_k=1e-2, scheduler_min_lr=1e-5)
    elif system_ == 's12':
        dataset_config = DatasetConfig(recreate_dataset=True, data_generation_strategy='trajectory', system_='s12',
                                       delay=ConstantDelay(1.), duration=12, dt=0.05, n_training_dataset=500,
                                       n_validation_dataset=1, n_sample_per_dataset=-1, baxter_dof=5, baxter_f=1,
                                       baxter_magnitude=0.1, baxter_alpha=1, baxter_beta=2, ic_lower_bound=0,
                                       ic_upper_bound=1, random_test_lower_bound=0, random_test_upper_bound=1)
        model_config = ModelConfig(model_name='FNO')
        train_config = TrainConfig(learning_rate=3e-4, training_ratio=0.8, n_epoch=200, batch_size=256,
                                   weight_decay=1e-3, log_step=-1, lr_scheduler_type='exponential', uq_alpha=0.01,
                                   scheduled_sampling_warm_start=0, scheduled_sampling_type='linear',
                                   scheduled_sampling_k=1e-2, scheduler_min_lr=1e-5)
    else:
        raise NotImplementedError()
    if n_iteration is not None:
        dataset_config.successive_approximation_n_iteration = n_iteration
    if delay is not None:
        dataset_config.delay = delay
    if duration is not None:
        dataset_config.duration = duration
    if model_name is not None:
        model_config.model_name = model_name
    if z2u is not None:
        model_config.z2u = z2u
        if z2u:
            print('Using z2u mode')
        else:
            print('Do not use z2u mode')
    dataset_config.system_ = system_
    model_config.system = system_
    train_config.system = system_
    return dataset_config, model_config, train_config


if __name__ == '__main__':
    ...
