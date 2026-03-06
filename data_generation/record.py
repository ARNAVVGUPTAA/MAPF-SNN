import sys
import os
import gc
from collections import deque
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np
from grid.env_graph_gridv1 import GraphEnv
import matplotlib.pyplot as plt


# ===== EPSILON-GREEDY MISTAKE INJECTION WITH RECOVERY ===== #

def replan_with_astar(start_pos, goal_pos, board_size, obstacles):
    """
    Lightweight BFS replan — no CBS overhead, no constraint tables.
    Returns the first action of the shortest path from start to goal.
    """
    if start_pos == goal_pos:
        return [4]  # already there, STAY

    obs_set = set(obstacles)
    W, H = board_size
    MOVES = [(1, 0, 0), (0, 1, 1), (-1, 0, 2), (0, -1, 3)]  # dx,dy,action

    queue = deque([(start_pos, [])])
    visited = {start_pos}

    while queue:
        pos, path = queue.popleft()
        for dx, dy, act in MOVES:
            npos = (pos[0] + dx, pos[1] + dy)
            if npos in visited:
                continue
            if not (0 <= npos[0] < W and 0 <= npos[1] < H):
                continue
            if npos in obs_set:
                continue
            new_path = path + [act]
            if npos == goal_pos:
                return new_path
            visited.add(npos)
            queue.append((npos, new_path))

    return None  # no path found


def inject_mistakes_and_replan(trajectory, env, epsilon=0.10):
    """
    Inject epsilon-greedy mistakes into trajectory with A* replanning.

    Mistake selection: 80% STAY, 20% random valid move.
    The *label* saved is the first step of the replanned recovery path so the
    network learns "given this delayed/off-path FOV, do X to recover".
    The *executed* action is the actual mistake so the FOV sequence reflects
    the true post-mistake positions.

    Args:
        trajectory: [num_agents, timesteps] original expert actions
        env:        GraphEnv environment
        epsilon:    Probability of mistake (default 0.10 = 10%)

    Returns:
        labels:   [num_agents, timesteps] recovery labels (for training)
        executed: [num_agents, timesteps] actions actually applied (for env stepping)
    """
    num_agents, timesteps = trajectory.shape
    board_size = (env.board_size, env.board_size)
    obstacles  = [(int(obs[0]), int(obs[1])) for obs in env.obstacles] if env.obstacles is not None else []

    labels_list   = [[] for _ in range(num_agents)]
    executed_list = [[] for _ in range(num_agents)]

    current_positions = [tuple(env.starting_positions[i]) for i in range(num_agents)]
    goals             = [tuple(env.goal[i])               for i in range(num_agents)]

    def apply_action(pos, action):
        x, y = pos
        if   action == 0: return (min(board_size[0] - 1, x + 1), y)
        elif action == 1: return (x, min(board_size[1] - 1, y + 1))
        elif action == 2: return (max(0, x - 1), y)
        elif action == 3: return (x, max(0, y - 1))
        else:             return (x, y)

    def get_valid_actions(pos):
        x, y  = pos
        valid = []
        if x + 1 < board_size[0] and (x + 1, y) not in obstacles: valid.append(0)
        if y + 1 < board_size[1] and (x, y + 1) not in obstacles: valid.append(1)
        if x - 1 >= 0           and (x - 1, y) not in obstacles: valid.append(2)
        if y - 1 >= 0           and (x, y - 1) not in obstacles: valid.append(3)
        valid.append(4)  # STAY always valid
        return valid

    for agent_id in range(num_agents):
        pos  = (int(current_positions[agent_id][0]), int(current_positions[agent_id][1]))
        goal = (int(goals[agent_id][0]),             int(goals[agent_id][1]))

        for t in range(timesteps):
            expert_action = int(trajectory[agent_id, t])

            if np.random.rand() < epsilon:
                # ── mistake selection: 80% STAY, 20% random valid ──
                if np.random.rand() < 0.8:
                    mistake_action = 4  # STAY
                else:
                    valids = get_valid_actions(pos)
                    mistake_action = int(np.random.choice(valids))

                # execute mistake → agent ends up at new_pos
                new_pos = apply_action(pos, mistake_action)

                # replan from new_pos
                recovery = replan_with_astar(new_pos, goal, board_size, obstacles)
                if recovery:
                    recovery_label = recovery[0]   # first step of recovery
                else:
                    recovery_label = expert_action  # fallback: trust expert

                # label = recovery action; executed = mistake
                labels_list[agent_id].append(recovery_label)
                executed_list[agent_id].append(mistake_action)
                pos = new_pos
            else:
                labels_list[agent_id].append(expert_action)
                executed_list[agent_id].append(expert_action)
                pos = apply_action(pos, expert_action)

    labels   = np.array(labels_list,   dtype=np.int32)
    executed = np.array(executed_list, dtype=np.int32)
    return labels, executed


# ===== END MISTAKE INJECTION ===== #


def make_env(pwd_path, config):
    try:
        with open(os.path.join(pwd_path, "input.yaml")) as input_params:
            params = yaml.load(input_params, Loader=yaml.FullLoader)
        if params is None:
            raise ValueError("YAML file is empty or returned None")
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
    except (yaml.YAMLError, ValueError, KeyError) as e:
        raise ValueError(f"Failed to load YAML from {pwd_path}: {e}")

    env = GraphEnv(
        config=config,
        goal=goals,
        board_size=int(dimensions[0]),
        starting_positions=starting_pos,
        obstacles=obstacles_list,
        sensing_range=config["sensor_range"],
    )
    return env


def record_env(path, config, epsilon=0.10):
    """
    Record environment states with epsilon-greedy mistake injection.
    
    Args:
        path: Path to dataset cases
        config: Configuration dict
        epsilon: Probability of mistake injection (default 0.10 = 10%)
    """
    cases = [d for d in os.listdir(path) if d.startswith("case_")]
    t = np.zeros(len(cases))

    print("Recording environment for cases:", cases)
    print(f"🎲 Epsilon-greedy mistake injection enabled: {epsilon*100:.0f}% error rate")
    
    for idx, case in enumerate(cases):
        case_path = os.path.join(path, case)
        traj_file = os.path.join(case_path, "trajectory.npy")
        
        # Load trajectory - no skipping
        try:
            trayectory = np.load(traj_file, allow_pickle=True)
            t[idx] = trayectory.shape[1]
        except Exception as e:
            print(f"Warning: Error loading {traj_file}: {e}, setting timestep to 0")
            t[idx] = 0

    print(f"max steps {np.max(t)}")
    print(f"min steps {np.min(t)}")
    print(f"mean steps {np.mean(t)}")
    with open(os.path.join(path, "stats.txt"), "w") as f:
        f.write(f"max steps {np.max(t)}\n")
        f.write(f"min steps {np.min(t)}\n")
        f.write(f"mean steps {np.mean(t)}\n")

    print("Recording states with mistake injection...")
    for idx, case in enumerate(cases):
        case_path = os.path.join(path, case)
        traj_file = os.path.join(case_path, "trajectory.npy")
        
        # Process all cases - no skipping
        try:
            trayectory = np.load(traj_file, allow_pickle=True)
            trayectory = trayectory[:, 1:]  # Remove first timestep (starting position)
            agent_nb = trayectory.shape[0]
            
            env = make_env(case_path, config)
            
            # ===== INJECT MISTAKES AND REPLAN ===== #
            # labels  = recovery actions (saved for training)
            # executed = actual mistake actions (used to step the env → real FOVs)
            if epsilon > 0:
                labels, executed = inject_mistakes_and_replan(trayectory, env, epsilon=epsilon)
            else:
                labels   = trayectory
                executed = trayectory
            # ===== END INJECTION ===== #

        except Exception as e:
            print(f"Warning: Error processing {case}: {e}, creating empty recording")
            # Create minimal empty recordings to maintain case count
            try:
                agent_nb = config.get('num_agents', 5)
                fov_size = 7
                empty_recording = np.zeros((1, agent_nb, 2, fov_size, fov_size))
                empty_adj = np.zeros((1, agent_nb, agent_nb, agent_nb))
                empty_traj = np.zeros((agent_nb, 1))
                np.save(os.path.join(case_path, "states.npy"), empty_recording)
                np.save(os.path.join(case_path, "gso.npy"), empty_adj)
                np.save(os.path.join(case_path, "trajectory_record.npy"), empty_traj)
            except Exception as save_err:
                print(f"Error saving empty recording for {case}: {save_err}")
            if idx % 25 == 0:
                print(f"Recorded -- [{idx}/{len(cases)}]")
            continue
        
        # FOV size is (2*pad - 1) where pad=4, so 7x7
        T_steps = executed.shape[1]
        fov_size = 7
        recordings = np.zeros(
            (T_steps, agent_nb, 2, fov_size, fov_size)
        )  # timestep, agents, channels of FOV, dimFOVx, dimFOVy
        adj_record = np.zeros((T_steps, agent_nb, agent_nb, agent_nb))
        assert (
            agent_nb == env.nb_agents
        ), rf"Trayectory has {agent_nb} agents, env expects {env.nb_agents}"
        obs = env.reset()
        emb = np.ones(env.nb_agents)
        for i in range(T_steps):
            recordings[i, :, :, :, :] = obs["fov"]
            adj_record[i, :, :, :] = obs["adj_matrix"]
            # Step env with EXECUTED actions (mistake positions → real FOV sequence)
            obs, _, _, _ = env.step(executed[:, i], emb)
        recordings[i, :, :, :, :] = obs["fov"]
        adj_record[i, :, :, :] = obs["adj_matrix"]

        np.save(os.path.join(case_path, "states.npy"), recordings)
        np.save(os.path.join(case_path, "gso.npy"), adj_record)
        # Save RECOVERY LABELS (not executed actions) as training targets
        np.save(os.path.join(case_path, "trajectory_record.npy"), labels)
        if idx % 25 == 0:
            print(f"Recorded -- [{idx}/{len(cases)}]")
        # free large arrays and A* objects before next case
        del recordings, adj_record, labels, executed
        gc.collect()
    print(f"Recorded -- [{idx}/{len(cases)}] --- completed")


if __name__ == "__main__":
    from config import config
    import torch
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ===== EPSILON-GREEDY CONFIG ===== #
    epsilon = config.get("mistake_epsilon", 0.10)  # Default 10% mistake rate
    # ===== END CONFIG ===== #
    
    # Determine path for recording from config
    val_conf = config.get('valid', {})
    path = val_conf.get('root_dir', 'dataset/5_8_28') + '/' + val_conf.get('mode', '')
    record_env(path, config, epsilon=epsilon)
