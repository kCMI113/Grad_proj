import argparse
import itertools

import yaml
from main import main
from src.utils import get_config


def update_config(settings, param_dict):
    new_settings = settings.copy()
    for key, value in param_dict.items():
        new_settings[key] = value
    return new_settings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", action="store", required=False)
    args = parser.parse_args()

    setting_yaml_path = f"./settings/{args.config}.yaml"
    base_settings = get_config(setting_yaml_path)

    # 그리드 서치할 하이퍼파라미터 정의
    param_grid = {
        "lr": [0.0001, 0.001, 0.005],
        "weight_decay": [0.0001, 0.001, 0.005],
        "batch_size": [64, 128],
        "hidden_size": [32, 256],
        "dropout_prob": [0.2, 0.4],
        "max_len": [30, 50, 70, 90],
    }

    # 모든 조합 생성
    param_combinations = list(itertools.product(*param_grid.values()))
    param_keys = list(param_grid.keys())

    for param_values in param_combinations:
        param_dict = dict(zip(param_keys, param_values))
        updated_settings = update_config(base_settings, param_dict)
        main(args, updated_settings)
