from dataset_gen import create_solutions
from trayectory_parser import parse_traject
from record import record_env
import torch
import os

# Absolute base so this script works regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))

# LaCAM0 binary (built from solvers/lacam0 in the project root)
_LACAM_BIN = os.path.join(_ROOT, 'solvers', 'lacam0', 'build', 'main')

# Default config
config = {
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'cases_train': 500,
    'cases_val':   100,
    'map_shape':   [9, 9],
    'board_size':  [9, 9],
    'num_agents':  5,
    'obstacles':   5,
    'sensing_range': 3,
    'sensor_range':  3,
    'max_time': 100,
    'min_time':   0,
    'train_root_dir': os.path.join(_ROOT, 'dataset', '5_5_9_recovery', 'train'),
    'valid_root_dir': os.path.join(_ROOT, 'dataset', '5_5_9_recovery', 'valid'),
    'mistake_epsilon': 0.10,
    # Solver: 'lacam' (fast, near-optimal) or 'cbs' (exact but slow/times-out)
    'solver': 'lacam',
}

if __name__ == "__main__":
    cases_train = config.get("cases_train", 500)
    cases_val   = config.get("cases_val",   100)
    solver      = config.get("solver", "lacam")
    epsilon     = config.get("mistake_epsilon", 0.10)

    print(f"\n  Solver : {solver.upper()}")
    print(f"  Train  : {cases_train} cases → {config['train_root_dir']}")
    print(f"  Valid  : {cases_val}  cases → {config['valid_root_dir']}")
    print(f"  Agents : {config['num_agents']}  Obstacles: {config['obstacles']}"
          f"  Grid: {config['map_shape']}")
    print(f"  Epsilon-greedy mistake injection: {epsilon*100:.0f}%\n")

    if solver == 'lacam' and not os.path.exists(_LACAM_BIN):
        print(f"  [error] LaCAM binary not found: {_LACAM_BIN}")
        print("          Build it with:")
        print("            cmake -B solvers/lacam0/build solvers/lacam0 -DCMAKE_BUILD_TYPE=Release")
        print("            make -C solvers/lacam0/build -j$(nproc)")
        raise SystemExit(1)

    config_train = config.copy()
    config_train.update({
        "path":        config["train_root_dir"],
        "nb_agents":   config["num_agents"],
        "nb_obstacles": config.get("obstacles", 0),
        "sensor_range": config.get("sensing_range"),
    })
    config_val = config.copy()
    config_val.update({
        "path":        config["valid_root_dir"],
        "nb_agents":   config["num_agents"],
        "nb_obstacles": config.get("obstacles", 0),
        "sensor_range": config.get("sensing_range"),
    })

    # Generate training data
    create_solutions(config_train["path"], cases_train, config_train,
                     timeout=10, solver=solver, lacam_bin=_LACAM_BIN)
    parse_traject(config_train["path"])
    record_env(config_train["path"], config_train, epsilon=epsilon)

    # Generate validation data
    create_solutions(config_val["path"], cases_val, config_val,
                     timeout=10, solver=solver, lacam_bin=_LACAM_BIN)
    parse_traject(config_val["path"])
    record_env(config_val["path"], config_val, epsilon=epsilon)
