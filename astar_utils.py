"""
Fast A* implementation for MAPF pathfinding after expert trajectory ends.
"""

import heapq
from typing import Tuple, Set, List, Optional


def fast_astar_action(start_pos: Tuple[int, int], 
                     goal_pos: Tuple[int, int], 
                     obstacles: Set[Tuple[int, int]], 
                     grid_size: int = 20, 
                     max_steps: int = 20) -> Optional[int]:
    """
    Fast A* to compute optimal next action from current position to goal.
    
    Args:
        start_pos: Current position (x, y) as tuple
        goal_pos: Goal position (x, y) as tuple
        obstacles: Set of obstacle positions as tuples (or array-like that will be converted)
        grid_size: Size of the grid (assumed square)
        max_steps: Maximum search steps to limit computation
        
    Returns:
        optimal_action: Action index (0=stay, 1=right, 2=up, 3=left, 4=down) or None
    """
    # FIX: Convert obstacles to set for O(1) membership testing
    if not isinstance(obstacles, set):
        obstacles = {tuple(map(int, p)) for p in obstacles}
    
    start = tuple(start_pos)
    goal = tuple(goal_pos)
    
    # If already at goal, stay
    if start == goal:
        return 0
    
    # Action mapping: 0=stay, 1=right, 2=up, 3=left, 4=down
    # FIX: Remove stay action from neighbors to avoid bloating search
    actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # No stay action in search
    
    # FIX: Initialize with proper Manhattan heuristic instead of 0
    h0 = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    open_set = [(h0, start, [])]  # (f_score, position, path)
    closed_set = set()
    step_count = 0
    
    while open_set and step_count < max_steps:
        step_count += 1
        f_score, current_pos, path = heapq.heappop(open_set)
        
        if current_pos in closed_set:
            continue
            
        closed_set.add(current_pos)
        
        # Check if goal reached
        if current_pos == goal:
            if len(path) > 0:
                return path[0]  # Return first action in optimal path
            else:
                return 0  # Stay if somehow no path needed
        
        # Explore neighbors (skip stay action to avoid bloating search)
        for action_idx, (dx, dy) in enumerate(actions):
            # Map to actual action indices: 1=right, 2=up, 3=left, 4=down
            actual_action_idx = action_idx + 1
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Check bounds
            if (new_pos[0] < 0 or new_pos[0] >= grid_size or 
                new_pos[1] < 0 or new_pos[1] >= grid_size):
                continue
            
            # Check obstacles (now works correctly with set)
            if new_pos in obstacles:
                continue
            
            # Check if already visited
            if new_pos in closed_set:
                continue
            
            # Compute costs
            g_score = len(path) + 1  # Cost from start
            h_score = abs(new_pos[0] - goal[0]) + abs(new_pos[1] - goal[1])  # Manhattan distance
            f_score = g_score + h_score
            
            # Add to open set
            new_path = path + [actual_action_idx]
            heapq.heappush(open_set, (f_score, new_pos, new_path))
    
    # No path found - return stay action
    return 0
    
    for action_idx, (dx, dy) in enumerate(actions):
        new_pos = (start[0] + dx, start[1] + dy)
        
        # Check bounds
        if (new_pos[0] < 0 or new_pos[0] >= grid_size or 
            new_pos[1] < 0 or new_pos[1] >= grid_size):
            continue
            
        # Check obstacles
        if new_pos in obstacles:
            continue
        
        # Compute distance to goal
        distance = abs(new_pos[0] - goal[0]) + abs(new_pos[1] - goal[1])
        if distance < min_distance:
            min_distance = distance
            best_action = action_idx
    
    return best_action


def batch_astar_actions(positions: List[Tuple[int, int]], 
                       goals: List[Tuple[int, int]], 
                       obstacles: Set[Tuple[int, int]], 
                       grid_size: int = 20, 
                       max_steps: int = 20) -> List[Optional[int]]:
    """
    Compute A* actions for multiple agents in batch.
    
    Args:
        positions: List of current positions for each agent
        goals: List of goal positions for each agent
        obstacles: Set of obstacle positions
        grid_size: Size of the grid
        max_steps: Maximum search steps per agent
        
    Returns:
        List of optimal actions for each agent
    """
    actions = []
    for pos, goal in zip(positions, goals):
        action = fast_astar_action(pos, goal, obstacles, grid_size, max_steps)
        actions.append(action)
    return actions


def get_heuristic_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """
    Compute Manhattan distance between two positions.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
        
    Returns:
        Manhattan distance
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def is_valid_position(pos: Tuple[int, int], 
                     obstacles: Set[Tuple[int, int]], 
                     grid_size: int) -> bool:
    """
    Check if a position is valid (within bounds and not an obstacle).
    
    Args:
        pos: Position to check (x, y)
        obstacles: Set of obstacle positions
        grid_size: Size of the grid
        
    Returns:
        True if position is valid, False otherwise
    """
    x, y = pos
    return (0 <= x < grid_size and 
            0 <= y < grid_size and 
            pos not in obstacles)


def get_neighbors(pos: Tuple[int, int], 
                 obstacles: Set[Tuple[int, int]], 
                 grid_size: int) -> List[Tuple[int, int, int]]:
    """
    Get valid neighboring positions with their corresponding actions.
    
    Args:
        pos: Current position (x, y)
        obstacles: Set of obstacle positions
        grid_size: Size of the grid
        
    Returns:
        List of (new_x, new_y, action_idx) tuples for valid neighbors
    """
    # Action mapping: 0=stay, 1=right, 2=up, 3=left, 4=down
    actions = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
    neighbors = []
    
    for action_idx, (dx, dy) in enumerate(actions):
        new_pos = (pos[0] + dx, pos[1] + dy)
        if is_valid_position(new_pos, obstacles, grid_size):
            neighbors.append((new_pos[0], new_pos[1], action_idx))
    
    return neighbors
