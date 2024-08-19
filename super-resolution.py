import uuid

import wandb

from config import get_config
from main import run_test
from utils import load_model, get_time_str

if __name__ == '__main__':
    dataset_config, model_config, train_config = get_config('s1', model_name='FNO-GRU')
    # dataset_config.duration = 15
    # dataset_config.dt = 0.2
    model, loaded = load_model(train_config, model_config, dataset_config)
    run = wandb.init(
        project="no",
        name=f'super-resolution {dataset_config.system_} {model_config.model_name} {get_time_str()}'
    )
    model_config.load_model(run, model)
    test_points = [(tp, uuid.uuid4()) for tp in dataset_config.test_points[:1]]
    result = run_test(m=model, dataset_config=dataset_config, train_config=train_config,
                      base_path=model_config.base_path, test_points=test_points, method='no')
    run.finish()