from dataset_gen import create_solutions
from trayectory_parser import parse_traject
from record import record_env
import torch
import os

# Absolute base so this script works regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))

# Default config
config = {
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'cases_train': 500,
    'cases_val': 100,
    'map_shape': [9, 9],
    'board_size': [9, 9],
    'num_agents': 5,
    'obstacles': 5,
    'sensing_range': 3,
    'sensor_range': 3,
    'max_time': 100,
    'min_time': 0,
    'train_root_dir': os.path.join(_ROOT, 'dataset', '5_5_9_recovery', 'train'),
    'valid_root_dir': os.path.join(_ROOT, 'dataset', '5_5_9_recovery', 'valid'),
    'mistake_epsilon': 0.10
}

if __name__ == "__main__":
    cases_train = config.get("cases_train", 10000)
    cases_val = config.get("cases_val", 200)
    
    # ===== EPSILON-GREEDY MISTAKE INJECTION CONFIG ===== #
    # Set epsilon=0.10 for 10% mistake rate, or 0.0 to disable
    epsilon = config.get("mistake_epsilon", 0.10)
    print(f"\n🎲 Mistake Injection: {epsilon*100:.0f}% error rate (epsilon={epsilon})")
    print("   This injects recovery states to prevent OOD errors during deployment.\n")
    # ===== END CONFIG ===== #

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
    create_solutions(config_train["path"], cases_train, config_train, timeout=5)
    parse_traject(config_train["path"])
    record_env(config_train["path"], config_train, epsilon=epsilon)

    # Generate validation data
    create_solutions(config_val["path"], cases_val, config_val, timeout=5)
    parse_traject(config_val["path"])
    record_env(config_val["path"], config_val, epsilon=epsilon)
