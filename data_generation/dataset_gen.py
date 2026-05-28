import sys
import os
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import argparse
import numpy as np
from cbs.cbs import Environment, CBS
import multiprocessing
import shutil

"""
MAPF Data Generation with CBS and Timeout/Retry
"""


def gen_input(dimensions: tuple[int, int], nb_obs: int, nb_agents: int) -> dict:
    input_dict = {"agents": [], "map": {"dimensions": dimensions, "obstacles": []}}
    starts = []
    goals = []
    obstacles = []

    def assign_obstacle(obstacles):
        good = False
        while not good:
            ag_obstacle = [
                np.random.randint(0, dimensions[0]),
                np.random.randint(0, dimensions[1]),
            ]
            if ag_obstacle not in obstacles:
                good = True
        return ag_obstacle

    def assign_start(starts, obstacles):
        good = False
        while not good:
            ag_start = [
                np.random.randint(0, dimensions[0]),
                np.random.randint(0, dimensions[1]),
            ]
            if ag_start not in starts and ag_start not in obstacles:
                good = True
        return ag_start

    def assign_goal(goals, obstacles):
        good = False
        while not good:
            ag_goal = [
                np.random.randint(0, dimensions[0]),
                np.random.randint(0, dimensions[1]),
            ]
            if ag_goal not in goals and ag_goal not in obstacles:
                good = True
        return ag_goal

    # Generate initial obstacles
    for obstacle in range(nb_obs):
        obstacle = assign_obstacle(obstacles)
        obstacles.append(obstacle)

    # Generate agent starts and goals
    for agent in range(nb_agents):
        start = assign_start(starts, obstacles)
        starts.append(start)
        goal = assign_goal(goals, obstacles)
        goals.append(goal)

    # Apply obstacle conflict resolution
    obstacles = _resolve_obstacle_conflicts_for_generation(
        obstacles, starts, goals, dimensions
    )

    # Update input dict with resolved obstacles
    input_dict["map"]["obstacles"] = [tuple(obs) for obs in obstacles]
    
    for agent in range(nb_agents):
        input_dict["agents"].append(
            {"start": starts[agent], "goal": goals[agent], "name": f"agent{agent}"}
        )

    return input_dict


def _resolve_obstacle_conflicts_for_generation(obstacles, agent_positions, goal_positions, dimensions):
    """
    Resolve conflicts between obstacles and agent positions/goals during data generation.
    This is similar to the enhanced_train.py version but adapted for the generation pipeline.
    """
    if not obstacles:
        return obstacles
        
    # Convert positions to sets for faster lookup
    occupied_positions = set()
    for pos in agent_positions + goal_positions:
        occupied_positions.add(tuple(pos))
    
    resolved_obstacles = []
    
    for obs in obstacles:
        obs_tuple = tuple(obs)
        
        # Check if obstacle conflicts with any agent position or goal
        if obs_tuple in occupied_positions:
            # Find a free space for this obstacle
            new_obs = _find_free_space_for_generation(occupied_positions, dimensions)
            if new_obs is not None:
                resolved_obstacles.append(new_obs)
                occupied_positions.add(tuple(new_obs))
            # If no free space found, skip this obstacle
        else:
            # No conflict, keep original obstacle
            resolved_obstacles.append(obs)
            occupied_positions.add(obs_tuple)
    
    return resolved_obstacles


def _find_free_space_for_generation(occupied_positions, dimensions, max_attempts=100):
    """
    Find a free space on the grid for obstacle relocation during data generation.
    """
    width, height = dimensions
    
    # Random search first (faster for sparse grids)
    for _ in range(max_attempts):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        if (x, y) not in occupied_positions:
            return [x, y]
    
    # Systematic search if random fails
    for x in range(width):
        for y in range(height):
            if (x, y) not in occupied_positions:
                return [x, y]
    
    # No free space found
    return None


def cbs_search_worker(env, result_queue):
    cbs = CBS(env, verbose=False)
    solution = cbs.search()
    result_queue.put(solution)


def lacam_data_gen(input_dict, output_path, lacam_bin, timeout=30):
    """
    Generate a MAPF solution using the LaCAM0 binary.
    Writes solution.yaml and input.yaml in the same format as data_gen (CBS).

    Args:
        input_dict: Standard map/agent dict from gen_input().
        output_path: Directory to write solution.yaml + input.yaml into.
        lacam_bin:   Absolute path to the lacam0 executable.
        timeout:     Time limit in seconds.
    """
    import subprocess, tempfile

    os.makedirs(output_path, exist_ok=True)

    param      = input_dict
    dimensions = param["map"]["dimensions"]   # [W, H] — cols, rows
    W, H       = dimensions[0], dimensions[1]
    obstacles  = param["map"]["obstacles"]
    agents     = param["agents"]
    n          = len(agents)

    with tempfile.TemporaryDirectory() as tmp:
        map_path  = os.path.join(tmp, 'env.map')
        scen_path = os.path.join(tmp, 'agents.scen')
        out_path  = os.path.join(tmp, 'result.txt')

        # ── map file ──────────────────────────────────────────────────────────
        grid = [['.' for _ in range(W)] for _ in range(H)]
        for obs in obstacles:
            ox, oy = int(obs[0]), int(obs[1])
            if 0 <= ox < W and 0 <= oy < H:
                grid[oy][ox] = '@'
        with open(map_path, 'w') as f:
            f.write(f"type octile\nheight {H}\nwidth {W}\nmap\n")
            for row in grid:
                f.write(''.join(row) + '\n')

        # ── scen file (col, row convention — same as CBS x, y) ───────────────
        with open(scen_path, 'w') as f:
            f.write("version 1\n")
            for ag in agents:
                sx, sy = ag["start"][0], ag["start"][1]
                gx, gy = ag["goal"][0],  ag["goal"][1]
                f.write(f"0\tenv.map\t{W}\t{H}\t{sx}\t{sy}\t{gx}\t{gy}\t0\n")

        # ── call LaCAM binary ─────────────────────────────────────────────────
        try:
            subprocess.run(
                [lacam_bin, '-m', map_path, '-i', scen_path,
                 '-N', str(n), '-o', out_path, '-v', '0', '-t', str(timeout)],
                timeout=timeout + 5,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            print(f" LaCAM timed out for {output_path}")
            return
        except FileNotFoundError:
            print(f" LaCAM binary not found: {lacam_bin}")
            raise

        if not os.path.exists(out_path):
            return

        with open(out_path) as f:
            content = f.read()

        if 'solved=1' not in content:
            return

        # ── parse paths ────────────────────────────────────────────────────────
        # LaCAM output format: each numbered line is ONE TIMESTEP, not an agent.
        # "t:(x0,y0),(x1,y1),...,(xN-1,yN-1),"  — N positions per line.
        paths = {i: [] for i in range(n)}
        for line in content.splitlines():
            if not line:
                continue
            colon_idx = line.find(':')
            if colon_idx < 0:
                continue
            if not line[:colon_idx].strip().isdigit():
                continue   # skip header lines (solution=, map_file=, etc.)
            try:
                pos_str   = line[colon_idx + 1:].strip().rstrip(',')
                positions = pos_str.split('),(')
                for agent_idx, pos in enumerate(positions):
                    if agent_idx >= n:
                        break
                    x, y = pos.strip('()').split(',')
                    paths[agent_idx].append((int(x), int(y)))
            except (ValueError, IndexError):
                continue

        if not all(paths[i] for i in range(n)):
            return   # incomplete parse

        # ── convert to CBS solution.yaml format ───────────────────────────────
        solution = {}
        for i, ag in enumerate(agents):
            solution[ag["name"]] = [
                {"t": t, "x": col, "y": row}
                for t, (col, row) in enumerate(paths[i])
            ]
        cost = sum(len(paths[i]) - 1 for i in range(n))

        with open(os.path.join(output_path, "solution.yaml"), 'w') as f:
            yaml.safe_dump({"schedule": solution, "cost": cost}, f)
        with open(os.path.join(output_path, "input.yaml"), 'w') as f:
            yaml.safe_dump(param, f)


def data_gen(input_dict, output_path, timeout=30):
    os.makedirs(output_path, exist_ok=True)
    param = input_dict
    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param["agents"]

    env = Environment(dimension, agents, obstacles)
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=cbs_search_worker, args=(env, result_queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        print(f" CBS timed out for {output_path}, skipping.")
        p.terminate()
        p.join()
        return

    if result_queue.empty():
        print(" Solution not found")
        return

    solution = result_queue.get()
    if not solution:
        print(" Solution not found")
        return

    # Write to output file
    output = dict()
    output["schedule"] = solution
    output["cost"] = env.compute_solution_cost(solution)
    solution_path = os.path.join(output_path, "solution.yaml")
    with open(solution_path, "w") as solution_path_f:
        yaml.safe_dump(output, solution_path_f)

    parameters_path = os.path.join(output_path, "input.yaml")
    with open(parameters_path, "w") as parameters_path_f:
        yaml.safe_dump(param, parameters_path_f)


def create_solutions(path, num_cases, config, max_attempts=5, timeout=30,
                     solver='lacam', lacam_bin=None):
    """
    Generate MAPF solution files for num_cases instances.

    Args:
        solver:    'lacam' (default, fast) or 'cbs' (slower, may time out).
        lacam_bin: Path to the lacam0 binary (required when solver='lacam').
    """
    if solver == 'lacam' and lacam_bin is None:
        raise ValueError("lacam_bin must be provided when solver='lacam'")

    os.makedirs(path, exist_ok=True)
    cases_ready = len(os.listdir(path))
    print(f"Generating solutions  [solver={solver}]")
    for i in range(cases_ready, num_cases):
        if i % 25 == 0:
            print(f"Solution -- [{i}/{num_cases}]")
        for attempt in range(max_attempts):
            inpt = gen_input(
                config["map_shape"], config["nb_obstacles"], config["nb_agents"]
            )
            case_path = os.path.join(path, f"case_{i}")
            if os.path.exists(case_path):
                shutil.rmtree(case_path)

            if solver == 'lacam':
                lacam_data_gen(inpt, case_path, lacam_bin=lacam_bin, timeout=timeout)
            else:
                data_gen(inpt, case_path, timeout=timeout)

            if os.path.exists(os.path.join(case_path, "solution.yaml")):
                break
            else:
                print(f"Retrying case {i} (attempt {attempt+1}/{max_attempts})")
        else:
            print(
                f"Case {i}: Failed after {max_attempts} attempts, skipping."
            )
        gc.collect()
    print(f"Cases stored in {path}")


if __name__ == "__main__":
    from config import config
    import torch

    config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Determine dataset path and number of cases from config
    train_conf = config.get("train", {})
    path = f"{train_conf.get('root_dir')}/{train_conf.get('mode')}"
    num_cases = config.get("cases_train", 2)
    create_solutions(
        path,
        num_cases,
        {
            "map_shape": config["map_shape"],
            "nb_agents": config["num_agents"],
            "nb_obstacles": config.get("obstacles", 0),
            "device": config["device"],
        },
    )
    # create_solutions(path, 2000, config)
    # total = 200
    # for i in range(0,total):
    #     if i%25 == 0:
    #         print(f"Solution[{i}/{total}]")
    #     inpt = gen_input([5,5],0,2)
    #     data_gen(inpt, path)
    # print(f"Solution[{i}/{total}]")
