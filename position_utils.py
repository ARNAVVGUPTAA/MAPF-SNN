#!/usr/bin/env python3
"""
Utilities for generating deterministic, non-overlapping agent positions and goals.
Implements the hardened API requirements for proper spatial relationships.
"""

import numpy as np
import torch
from typing import Tuple, Optional, List
import warnings


def calculate_manhattan_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculate Manhattan distance between two positions."""
    return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])


def check_minimum_distance(pos: np.ndarray, existing_positions: List[np.ndarray], 
                          min_distance: int = 2) -> bool:
    """Check if position maintains minimum distance from existing positions."""
    for existing_pos in existing_positions:
        if calculate_manhattan_distance(pos, existing_pos) < min_distance:
            return False
    return True


def create_deterministic_obstacles(board_size: Tuple[int, int],
                                   num_obstacles: int,
                                   min_separation_padding: int = 1, # Added parameter
                                   seed: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Generate deterministic obstacle positions with minimum separation.

    Args:
        board_size: (width, height) of the grid.
        num_obstacles: Number of obstacles to place.
        min_separation_padding: Minimum number of empty cells between obstacles.
                                 0 means obstacles can be adjacent.
        seed: Random seed for reproducible results.

    Returns:
        Array of obstacle positions [num_obstacles, 2], or None if num_obstacles is 0.
    """
    if num_obstacles == 0:
        return None

    if seed is not None:
        np.random.seed(seed)

    width, height = board_size
    obstacle_positions = []
    attempts = 0
    # Allow more attempts for obstacles, especially with separation constraints
    max_attempts = num_obstacles * width * height # Increased max_attempts

    # The effective minimum Manhattan distance required by check_minimum_distance
    # If min_separation_padding = 0, obstacles can be adjacent (dist=1).
    # If min_separation_padding = 1, one cell gap (dist=2).
    required_manhattan_dist = min_separation_padding + 1

    while len(obstacle_positions) < num_obstacles and attempts < max_attempts:
        attempts += 1
        pos = np.array([np.random.randint(0, width), np.random.randint(0, height)])
        
        # Check if position maintains minimum distance from other obstacles
        if check_minimum_distance(pos, obstacle_positions, required_manhattan_dist):
            obstacle_positions.append(pos)
        # No need for a separate duplicate check, as check_minimum_distance handles it
        # if required_manhattan_dist >= 1 (which it will be for non-negative padding)

    if len(obstacle_positions) < num_obstacles:
        warnings.warn(
            f"Could not place all {num_obstacles} obstacles with min_separation_padding {min_separation_padding}. "
            f"Only placed {len(obstacle_positions)} in {attempts} attempts. "
            f"Consider increasing board size, reducing num_obstacles, or decreasing min_separation_padding."
        )
        if not obstacle_positions: # if no obstacles were placed, return None
            return None

    return np.array(obstacle_positions)


def create_deterministic_positions_and_goals(board_size: Tuple[int, int],
                                           num_agents: int,
                                           obstacles: Optional[np.ndarray] = None,
                                           padding_radius: int = 1, # Min Manhattan dist between agent/goal entities is padding_radius + 1
                                           min_goal_dist_manhattan: int = 5, # Direct Manhattan distance for start-goal
                                           start_goal_separation_padding: int = 1, # Min Manhattan dist between any start and any goal entity
                                           seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create deterministic, non-overlapping agent and goal positions.
    
    This is the main function that implements all the requirements:
    - Agent/Goal positions have a minimum Manhattan distance of `padding_radius + 1` from each other.
    - Start-Goal pairs have a minimum Manhattan distance of `min_goal_dist_manhattan`.
    - Any start position has a minimum Manhattan distance of `start_goal_separation_padding + 1` from any goal position.
    - Ensure no overlapping positions with obstacles.
    
    Args:
        board_size: (width, height) of the grid
        num_agents: Number of agents
        obstacles: Array of obstacle positions [N, 2]
        padding_radius: Defines min Manhattan distance (padding_radius + 1) between entities of the same type (e.g., agent-agent).
        min_goal_dist_manhattan: Minimum Manhattan distance between an agent's start and its own goal.
        start_goal_separation_padding: Defines min Manhattan distance (start_goal_separation_padding + 1) between any start and any goal.
        seed: Random seed for reproducible results
        
    Returns:
        Tuple of (agent_positions, goal_positions), each [num_agents, 2]
        
    Raises:
        ValueError: If constraints cannot be satisfied
    """
    if seed is not None:
        np.random.seed(seed)

    width, height = board_size
    all_potential_positions = [np.array([x, y]) for x in range(width) for y in range(height)]

    # Filter out obstacle positions
    valid_positions_for_starts = []
    if obstacles is not None:
        obstacle_set = {tuple(obs) for obs in obstacles}
        for pos in all_potential_positions:
            if tuple(pos) not in obstacle_set:
                valid_positions_for_starts.append(pos)
    else:
        valid_positions_for_starts = all_potential_positions

    if len(valid_positions_for_starts) < num_agents:
        raise ValueError(f"Not enough non-obstacle cells ({len(valid_positions_for_starts)}) to place {num_agents} agents.")

    agent_positions = []
    attempts = 0
    max_attempts_starts = num_agents * len(valid_positions_for_starts) # Heuristic for max attempts

    # 1. Generate Agent Start Positions
    temp_available_starts = list(valid_positions_for_starts)
    np.random.shuffle(temp_available_starts) # Shuffle for randomness

    for _ in range(num_agents):
        placed_agent = False
        current_agent_attempts = 0
        while temp_available_starts and not placed_agent and current_agent_attempts < len(valid_positions_for_starts):
            candidate_start = temp_available_starts.pop(0) # Try next available shuffled position
            current_agent_attempts += 1
            attempts += 1
            if check_minimum_distance(candidate_start, agent_positions, padding_radius + 1):
                agent_positions.append(candidate_start)
                placed_agent = True
                break
        if not placed_agent:
            raise ValueError(f"Could not place all {num_agents} agent starting positions with padding {padding_radius}. Placed {len(agent_positions)}.")
    
    # 2. Generate Goal Positions
    goal_positions = []
    # Valid positions for goals are also non-obstacle cells
    valid_positions_for_goals = list(valid_positions_for_starts) 

    for i in range(num_agents):
        start_pos = agent_positions[i]
        placed_goal_for_agent = False
        attempts_goal = 0
        max_attempts_goal_per_agent = len(valid_positions_for_goals) * 2 # Heuristic
        
        # Shuffle available goal positions for this agent to try different spots
        np.random.shuffle(valid_positions_for_goals)
        temp_available_goals = list(valid_positions_for_goals)

        while temp_available_goals and not placed_goal_for_agent and attempts_goal < max_attempts_goal_per_agent:
            candidate_goal = temp_available_goals.pop(0)
            attempts_goal += 1
            attempts += 1

            # Constraint 1: Min Manhattan distance from its own start
            if calculate_manhattan_distance(start_pos, candidate_goal) < min_goal_dist_manhattan:
                continue

            # Constraint 2: Min Manhattan distance from other already placed goals (padding_radius + 1)
            if not check_minimum_distance(candidate_goal, goal_positions, padding_radius + 1):
                continue
            
            # Constraint 3: Min Manhattan distance from ANY agent start position (start_goal_separation_padding + 1)
            # This is a stricter global separation between the set of starts and set of goals.
            valid_separation_from_all_starts = True
            for any_start_pos in agent_positions:
                if calculate_manhattan_distance(any_start_pos, candidate_goal) < start_goal_separation_padding + 1:
                    valid_separation_from_all_starts = False
                    break
            if not valid_separation_from_all_starts:
                continue
            
            # Constraint 4: Goal cannot be same as any start position (even if different agent)
            # This is implicitly handled by start_goal_separation_padding >= 0, but good to be explicit if padding is 0.
            is_a_start_position = False
            for s_pos in agent_positions:
                if np.array_equal(candidate_goal, s_pos):
                    is_a_start_position = True
                    break
            if is_a_start_position:
                continue

            goal_positions.append(candidate_goal)
            placed_goal_for_agent = True
            break # Move to next agent's goal

        if not placed_goal_for_agent:
            raise ValueError(
                f"Could not place goal for agent {i} satisfying all constraints: "
                f"start_pos={start_pos}, min_goal_dist_manhattan={min_goal_dist_manhattan}, "
                f"padding_radius={padding_radius} (for goal-goal), "
                f"start_goal_separation_padding={start_goal_separation_padding} (for start-any_goal). "
                f"Placed {len(goal_positions)} goals so far. Total attempts: {attempts}."
            )

    # Final validation (optional, as checks are done during generation)
    # if not validate_position_constraints(...): # Adapt validate_position_constraints if used
    #     raise ValueError("Final validation failed for generated positions and goals.")

    return np.array(agent_positions), np.array(goal_positions)


def validate_position_constraints(agent_positions: np.ndarray,
                                goal_positions: np.ndarray,
                                board_size: Tuple[int, int],
                                obstacles: Optional[np.ndarray] = None,
                                agent_padding_radius: int = 1, # For agent-agent
                                goal_padding_radius: int = 1,  # For goal-goal
                                min_goal_dist_manhattan: int = 5, # For start_i - goal_i
                                start_goal_separation_padding: int = 1 # For any start - any goal
                                ) -> bool:
    """
    Validate that positions meet all constraints.
    
    Args:
        agent_positions: Agent starting positions [num_agents, 2]
        goal_positions: Goal positions [num_agents, 2]
        board_size: (width, height) of the grid
        obstacles: Array of obstacle positions [N, 2]
        agent_padding_radius: Min Manhattan distance between agents is agent_padding_radius + 1
        goal_padding_radius: Min Manhattan distance between goals is goal_padding_radius + 1
        min_goal_dist_manhattan: Minimum Manhattan distance between an agent's start and its own goal.
        start_goal_separation_padding: Min Manhattan distance between any start and any goal is start_goal_separation_padding + 1.
        
    Returns:
        bool: True if all constraints are satisfied
    """
    width, height = board_size
    num_agents = len(agent_positions)

    if len(goal_positions) != num_agents:
        print(f"Validation Error: Mismatch in number of agents and goals. Agents: {num_agents}, Goals: {len(goal_positions)}")
        return False

    # Check bounds and obstacle collision for agent positions
    for i in range(num_agents):
        pos = agent_positions[i]
        if not (0 <= pos[0] < width and 0 <= pos[1] < height):
            print(f"Validation Error: Agent {i} position {pos} out of bounds {board_size}.")
            return False
        if obstacles is not None and any(np.array_equal(pos, obs) for obs in obstacles):
            print(f"Validation Error: Agent {i} position {pos} collides with an obstacle.")
            return False

    # Check bounds and obstacle collision for goal positions
    for i in range(num_agents):
        pos = goal_positions[i]
        if not (0 <= pos[0] < width and 0 <= pos[1] < height):
            print(f"Validation Error: Goal {i} position {pos} out of bounds {board_size}.")
            return False
        if obstacles is not None and any(np.array_equal(pos, obs) for obs in obstacles):
            print(f"Validation Error: Goal {i} position {pos} collides with an obstacle.")
            return False

    # Check agent-agent padding
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            dist = calculate_manhattan_distance(agent_positions[i], agent_positions[j])
            if dist < agent_padding_radius + 1:
                print(f"Validation Error: Agents {i} ({agent_positions[i]}) and {j} ({agent_positions[j]}) are too close (dist: {dist}, required: {agent_padding_radius + 1}).")
                return False

    # Check goal-goal padding
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            dist = calculate_manhattan_distance(goal_positions[i], goal_positions[j])
            if dist < goal_padding_radius + 1:
                print(f"Validation Error: Goals {i} ({goal_positions[i]}) and {j} ({goal_positions[j]}) are too close (dist: {dist}, required: {goal_padding_radius + 1}).")
                return False

    # Check start_i - goal_i minimum Manhattan distance
    for i in range(num_agents):
        dist = calculate_manhattan_distance(agent_positions[i], goal_positions[i])
        if dist < min_goal_dist_manhattan:
            print(f"Validation Error: Agent {i}'s start ({agent_positions[i]}) and goal ({goal_positions[i]}) are too close (dist: {dist}, required: {min_goal_dist_manhattan}).")
            return False

    # Check separation between any start and any goal (including non-paired ones)
    for i in range(num_agents): # Iterate over agent starts
        for j in range(num_agents): # Iterate over agent goals
            dist = calculate_manhattan_distance(agent_positions[i], goal_positions[j])
            if dist < start_goal_separation_padding + 1:
                print(f"Validation Error: Agent {i}'s start ({agent_positions[i]}) and Agent {j}'s goal ({goal_positions[j]}) are too close (dist: {dist}, required: {start_goal_separation_padding + 1}). This includes an agent's start being identical to another agent's goal or too close.")
                return False
            # Also ensure a start position is not identical to a goal position (even if for different agents)
            # This is covered by the check above if start_goal_separation_padding >= 0
            if np.array_equal(agent_positions[i], goal_positions[j]) and (start_goal_separation_padding + 1 > 0):
                 # This case should be caught by the distance check if padding > -1
                 pass # Already handled by Manhattan distance check

    return True


def torch_unique_consecutive_test(agent_positions: torch.Tensor, num_agents: int) -> bool:
    """
    Unit test to verify torch.unique_consecutive constraint.
    
    Args:
        agent_positions: Tensor of shape [batch, num_agents, 2]
        num_agents: Expected number of agents
        
    Returns:
        bool: True if constraint is satisfied
    """
    # Flatten positions and check uniqueness
    flat_positions = agent_positions.view(-1, 2)
    unique_positions = torch.unique(flat_positions, dim=0)
    
    expected_unique = agent_positions.shape[0] * num_agents
    actual_unique = unique_positions.shape[0]
    
    return actual_unique == expected_unique


# Test functions for validation
if __name__ == "__main__":
    print("Testing position generation utilities...")
    
    # Test case 1: Small grid
    board_size = (8, 8)
    num_agents = 4
    
    try:
        agent_pos, goal_pos = create_deterministic_positions_and_goals(
            board_size, num_agents, seed=42
        )
        print(f"✅ Small grid test passed")
        print(f"   Agent positions: {agent_pos}")
        print(f"   Goal positions: {goal_pos}")
        
        # Validate with torch test
        agent_tensor = torch.from_numpy(agent_pos).unsqueeze(0).float()  # [1, num_agents, 2]
        if torch_unique_consecutive_test(agent_tensor, num_agents):
            print(f"✅ Torch unique test passed")
        else:
            print(f"❌ Torch unique test failed")
            
    except ValueError as e:
        print(f"❌ Small grid test failed: {e}")
    
    # Test case 2: Larger grid with obstacles
    board_size = (16, 16)
    num_agents = 8
    obstacles = np.array([[5, 5], [6, 6], [7, 7], [8, 8]])
    
    try:
        agent_pos, goal_pos = create_deterministic_positions_and_goals(
            board_size, num_agents, obstacles, seed=42
        )
        print(f"✅ Large grid with obstacles test passed")
        print(f"   {len(agent_pos)} agents placed")
        print(f"   Average start-goal distance: {np.mean([calculate_manhattan_distance(agent_pos[i], goal_pos[i]) for i in range(num_agents)]):.2f}")
        
    except ValueError as e:
        print(f"❌ Large grid test failed: {e}")
    
    print("Position generation utility tests completed!")
