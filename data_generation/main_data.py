from dataset_gen import create_solutions
from trayectory_parser import parse_traject
from record import record_env

import argparse, yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="configs/config_snn.yaml")
args = parser.parse_args()

with open(args.config, "r") as config_path:
    config = yaml.safe_load(config_path)


if __name__ == "__main__":
    cases_train = 1000
    cases_val = 200

    config_train = {
        "num_agents": config["num_agents"],
        "map_shape": config["map_shape"],
        "nb_agents": config["num_agents"],
        "nb_obstacles": config["obstacles"],
        "sensor_range": config["sensing_range"],
        "board_size": config["board_size"],
        "max_time": config["max_time"],
        "min_time": config["min_time"],
        "path": "dataset/5_8_28/train",
    }
    config_val = config_train.copy()
    config_val["path"] = "dataset/5_8_28/val"

    # Generate training data
    create_solutions(config_train["path"], cases_train, config_train)
    parse_traject(config_train["path"])
    record_env(config_train["path"], config_train)

    # Generate validation data
    create_solutions(config_val["path"], cases_val, config_val)
    parse_traject(config_val["path"])
    record_env(config_val["path"], config_val)
