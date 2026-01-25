import os
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


def create_solutions(path, num_cases, config, max_attempts=5, timeout=30):
    os.makedirs(path, exist_ok=True)
    cases_ready = len(os.listdir(path))
    print("Generating solutions")
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
            data_gen(inpt, case_path, timeout=timeout)
            if os.path.exists(os.path.join(case_path, "solution.yaml")):
                break
            else:
                print(f"Retrying case {i} (attempt {attempt+1}/{max_attempts})")
        else:
            print(
                f"Case {i}: Failed to generate a solvable instance after {max_attempts} attempts, skipping."
            )
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
