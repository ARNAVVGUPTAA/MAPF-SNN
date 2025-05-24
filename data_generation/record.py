import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.append(r"C:\Users\victo\Desktop\VU master\MLGP\Extra")

import os
import yaml
import numpy as np
from grid.env_graph_gridv1 import GraphEnv
import matplotlib.pyplot as plt


def make_env(pwd_path, config):
    with open(os.path.join(pwd_path, "input.yaml")) as input_params:
        params = yaml.load(input_params, Loader=yaml.FullLoader)
    nb_agents = len(params["agents"])
    dimensions = params["map"]["dimensions"]
    obstacles = params["map"]["obstacles"]
    starting_pos = np.zeros((nb_agents, 2), dtype=np.int32)
    goals = np.zeros((nb_agents, 2), dtype=np.int32)
    obstacles_list = np.zeros((len(obstacles), 2), dtype=np.int32)
    for i in range(len(obstacles)):
        obstacles_list[i, :] = np.array([int(obstacles[i][0]), int(obstacles[i][1])])

    for d, i in zip(params["agents"], range(0, nb_agents)):
        #   name = d["name"]
        starting_pos[i, :] = np.array([int(d["start"][0]), int(d["start"][1])])
        goals[i, :] = np.array([int(d["goal"][0]), int(d["goal"][1])])

    env = GraphEnv(
        config=config,
        goal=goals,
        board_size=int(dimensions[0]),
        starting_positions=starting_pos,
        obstacles=obstacles_list,
        sensing_range=config["sensor_range"],
    )
    return env


def record_env(path, config):
    cases = [d for d in os.listdir(path) if d.startswith("case_")]
    t = np.zeros(len(cases))

    print("Recording environment for cases:", cases)
    for idx, case in enumerate(cases):
        case_path = os.path.join(path, case)
        traj_file = os.path.join(case_path, "trajectory.npy")
        if not os.path.exists(traj_file):
            print(f"Warning: {traj_file} does not exist, skipping.")
            continue
        trayectory = np.load(traj_file, allow_pickle=True)
        t[idx] = trayectory.shape[1]

    print(f"max steps {np.max(t)}")
    print(f"min steps {np.min(t)}")
    print(f"mean steps {np.mean(t)}")
    with open(os.path.join(path, "stats.txt"), "w") as f:
        f.write(f"max steps {np.max(t)}\n")
        f.write(f"min steps {np.min(t)}\n")
        f.write(f"mean steps {np.mean(t)}\n")

    print("Recording states...")
    for idx, case in enumerate(cases):
        case_path = os.path.join(path, case)
        traj_file = os.path.join(case_path, "trajectory.npy")
        if not os.path.exists(traj_file):
            print(f"Warning: {traj_file} does not exist, skipping.")
            continue
        trayectory = np.load(traj_file, allow_pickle=True)
        trayectory = trayectory[:, 1:]
        agent_nb = trayectory.shape[0]
        env = make_env(case_path, config)
        recordings = np.zeros(
            (trayectory.shape[1], agent_nb, 2, 5, 5)
        )  # timestep, agents, channels of FOV, dimFOVx, dimFOVy
        adj_record = np.zeros((trayectory.shape[1], agent_nb, agent_nb, agent_nb))
        assert (
            agent_nb == env.nb_agents
        ), rf"Trayectory has {agent_nb} agents, env expects {env.nb_agents}"
        obs = env.reset()
        emb = np.ones(env.nb_agents)
        for i in range(trayectory.shape[1]):
            recordings[i, :, :, :, :] = obs["fov"]
            adj_record[i, :, :, :] = obs["adj_matrix"]
            actions = trayectory[:, i]
            obs, _, _, _ = env.step(actions, emb)
        recordings[i, :, :, :, :] = obs["fov"]
        adj_record[i, :, :, :] = obs["adj_matrix"]

        np.save(os.path.join(case_path, "states.npy"), recordings)
        np.save(os.path.join(case_path, "gso.npy"), adj_record)
        np.save(os.path.join(case_path, "trajectory_record.npy"), trayectory)
        if idx % 25 == 0:
            print(f"Recorded -- [{idx}/{len(cases)}]")
    print(f"Recorded -- [{idx}/{len(cases)}] --- completed")


if __name__ == "__main__":
    from config import config
    import torch
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Determine path for recording from config
    val_conf = config.get('valid', {})
    path = val_conf.get('root_dir', 'dataset/5_8_28') + '/' + val_conf.get('mode', '')
    record_env(path, config)
