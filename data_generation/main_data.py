from dataset_gen import create_solutions
from trayectory_parser import parse_traject
from record import record_env

from config import config
import torch

config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    cases_train = config.get("cases_train", 1000)
    cases_val = config.get("cases_val", 200)

    config_train = config.copy()
    config_train.update({
        "path": config.get("train_root_dir", "dataset/5_8_28/train"),
        "nb_agents": config["num_agents"],
        "nb_obstacles": config.get("obstacles", 0),
        "sensor_range": config.get("sensing_range"),
    })
    config_val = config.copy()
    config_val.update({
        "path": config.get("valid_root_dir", "dataset/5_8_28/val"),
        "nb_agents": config["num_agents"],
        "nb_obstacles": config.get("obstacles", 0),
        "sensor_range": config.get("sensing_range"),
    })

    # Generate training data
    create_solutions(config_train["path"], cases_train, config_train)
    parse_traject(config_train["path"])
    record_env(config_train["path"], config_train)

    # Generate validation data
    create_solutions(config_val["path"], cases_val, config_val)
    parse_traject(config_val["path"])
    record_env(config_val["path"], config_val)
