from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    return config