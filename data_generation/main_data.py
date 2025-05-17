from dataset_gen import create_solutions
from trayectory_parser import parse_traject
from record import record_env


if __name__ == "__main__":
    cases_train = 1000
    cases_val = 200

    config_train = {
        "num_agents": 9,
        "map_shape": [28, 28],
        "nb_agents": 5,
        "nb_obstacles": 8,
        "sensor_range": 4,
        "board_size": [28, 28],
        "max_time": 32,
        "min_time": 9,
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
