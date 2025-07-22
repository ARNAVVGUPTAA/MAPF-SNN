#!/usr/bin/env python3
"""
Pre-training Module for SNN MAPF
================================

This module provides pre-training functionality to teach the SNN basic goal-seeking behavior
on simple 3x3 grids with 2 agents before moving to complex scenarios.

Pre-training helps the model learn:
1. Basic goal-oriented movement
2. Simple collision avoidance
3. Turn-taking behavior
4. Fundamental action-outcome relationships

Author: MAPF-GNN Pre-training System
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import yaml
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm

# SNN-specific imports
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.base import MemoryModule

# Import spike monitoring for network health
from spike_monitor import create_spike_monitor, SpikeTracker

# Import RL utilities for reward-based learning


def debug_snn_layers(model, spike_tracker, t, batch_idx):
    """
    Debug function to show all available SNN layers and their tracking status.
    """
    if t == 0 and batch_idx == 0 and spike_tracker.enabled and hasattr(model, 'snn_block'):
        print(f"   🔍 Available SNN Layers:")
        layer_count = 0
        for name, module in model.snn_block.named_modules():
            if hasattr(module, '_last_spike_output'):
                layer_count += 1
                spike_output = module._last_spike_output
                has_spikes = isinstance(spike_output, torch.Tensor) and spike_output.numel() > 0
                has_membrane = hasattr(module, 'v') and module.v is not None
                print(f"      {layer_count}. {name}: spikes={has_spikes}, membrane={has_membrane}")
        
        if layer_count == 0:
            print(f"      ⚠️ No SNN layers with _last_spike_output found!")
        else:
            print(f"      ✅ Total SNN layers: {layer_count}")

def safe_reset_snn(model):
    """
    Safely reset SNN state, only resetting modules that are actual memory modules.
    This avoids warnings about trying to reset non-memory modules.
    """
    for module in model.modules():
        if isinstance(module, MemoryModule):
            module.reset()

def generate_simple_scenario(grid_size: int = 3, num_agents: int = 2, num_obstacles: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a simple scenario with agents, goals, and obstacles, no overlapping.
    Ensures agents don't start at their goal positions.
    
    Args:
        grid_size: Size of the grid (3x3, 5x5, 7x7, etc.)
        num_agents: Number of agents (default 2)
        num_obstacles: Number of obstacles (default 2)
        
    Returns:
        positions: [num_agents, 2] - agent starting positions
        goals: [num_agents, 2] - goal positions
        obstacles: [num_obstacles, 2] - obstacle positions
    """
    max_attempts = 100
    for attempt in range(max_attempts):
        # Available positions in grid
        available_positions = []
        for x in range(grid_size):
            for y in range(grid_size):
                available_positions.append([x, y])
        
        # Randomly select positions for agents, goals, and obstacles (no overlap)
        total_needed = num_agents * 2 + num_obstacles  # agents + goals + obstacles
        
        if total_needed > len(available_positions):
            raise ValueError(f"Not enough positions in {grid_size}x{grid_size} grid for {num_agents} agents and {num_obstacles} obstacles")
        
        selected_positions = random.sample(available_positions, total_needed)
        
        # First num_agents are agent positions
        agent_positions = torch.tensor(selected_positions[:num_agents], dtype=torch.float32)
        
        # Next num_agents are goal positions  
        goal_positions = torch.tensor(selected_positions[num_agents:num_agents*2], dtype=torch.float32)
        
        # Remaining are obstacle positions
        obstacle_positions = torch.tensor(selected_positions[num_agents*2:], dtype=torch.float32)
        
        # Check that no agent starts at its goal (minimum distance of 1.0)
        valid_scenario = True
        for i in range(num_agents):
            distance = torch.norm(agent_positions[i] - goal_positions[i]).item()
            if distance < 1.0:  # Agent too close to its own goal
                valid_scenario = False
                break
        
        if valid_scenario:
            return agent_positions, goal_positions, obstacle_positions
    
    # If we couldn't generate a valid scenario, fall back to original method
    print(f"⚠️ Warning: Could not generate scenario with min distance constraint after {max_attempts} attempts")
    selected_positions = random.sample(available_positions, total_needed)
    agent_positions = torch.tensor(selected_positions[:num_agents], dtype=torch.float32)
    goal_positions = torch.tensor(selected_positions[num_agents:num_agents*2], dtype=torch.float32)
    obstacle_positions = torch.tensor(selected_positions[num_agents*2:], dtype=torch.float32)
    
    return agent_positions, goal_positions, obstacle_positions

def get_optimal_action(current_pos: torch.Tensor, goal_pos: torch.Tensor, other_agents: List[torch.Tensor] = None, 
                      obstacles: torch.Tensor = None, grid_size: int = 3) -> int:
    """
    Get action using simple heuristic for RL training (no A* pathfinding).
    This allows the model to learn optimal actions through RL rewards/penalties.
    
    Actions: 0=Stay, 1=Right, 2=Up, 3=Left, 4=Down
    
    Args:
        current_pos: [2] current position
        goal_pos: [2] goal position
        other_agents: List of other agent positions to avoid
        obstacles: [num_obstacles, 2] obstacle positions to avoid
        grid_size: Size of the grid for bounds checking
        
    Returns:
        action: int (0-4) - simple heuristic direction toward goal
    """
    # Simple heuristic: move toward goal
    dx = goal_pos[0] - current_pos[0]
    dy = goal_pos[1] - current_pos[1]
    
    # If at goal, stay
    if dx == 0 and dy == 0:
        return 0  # Stay
    
    # Choose action based on largest distance component
    if abs(dx) > abs(dy):
        return 1 if dx > 0 else 3  # Right or Left
    elif dy != 0:
        return 2 if dy > 0 else 4  # Up or Down
    else:
        return 0  # Stay if no clear direction

def generate_expert_trajectory(initial_positions: torch.Tensor, goals: torch.Tensor, obstacles: torch.Tensor, 
                             grid_size: int, max_steps: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate trajectory using simple heuristics for RL training (no expert supervision).
    This allows the model to learn optimal policies through trial and error with RL.
    
    Args:
        initial_positions: [num_agents, 2] starting positions
        goals: [num_agents, 2] goal positions
        obstacles: [num_obstacles, 2] obstacle positions
        grid_size: Size of the grid
        max_steps: maximum trajectory length
        
    Returns:
        trajectory_positions: [max_steps, num_agents, 2] position sequence
        trajectory_actions: [max_steps, num_agents] action sequence (heuristic-based)
    """
    num_agents = initial_positions.shape[0]
    current_positions = initial_positions.clone()
    
    trajectory_positions = []
    trajectory_actions = []
    
    for step in range(max_steps):
        trajectory_positions.append(current_positions.clone())
        
        step_actions = []
        next_positions = []
        
        # Get actions for each agent
        for agent_idx in range(num_agents):
            other_agents = [current_positions[j] for j in range(num_agents) if j != agent_idx]
            action = get_optimal_action(current_positions[agent_idx], goals[agent_idx], other_agents, obstacles, grid_size)
            step_actions.append(action)
            
            # Calculate next position
            action_deltas = torch.tensor([
                [0, 0],   # 0: Stay
                [1, 0],   # 1: Right
                [0, 1],   # 2: Up
                [-1, 0],  # 3: Left
                [0, -1],  # 4: Down
            ], dtype=torch.float32)
            
            next_pos = current_positions[agent_idx] + action_deltas[action]
            next_positions.append(next_pos)
        trajectory_actions.append(torch.tensor(step_actions))
        current_positions = torch.stack(next_positions)
        
        # Check if all agents reached goals
        all_reached = all(torch.equal(current_positions[i], goals[i]) for i in range(num_agents))
        if all_reached:
            # STOP HERE - don't pad with stay actions!
            break
    
    # Only return actual trajectory (no padding with stays)
    if len(trajectory_positions) == 0:
        # Edge case: no steps taken
        trajectory_positions = [initial_positions]
        trajectory_actions = [torch.zeros(num_agents, dtype=torch.long)]
    
    return torch.stack(trajectory_positions), torch.stack(trajectory_actions)

def create_fov_observation(position: torch.Tensor, goal: torch.Tensor, other_agents: List[torch.Tensor], 
                          obstacles: torch.Tensor, grid_size: int = 3, fov_size: int = 7, 
                          all_goals: torch.Tensor = None, agent_idx: int = None) -> torch.Tensor:
    """
    Create field-of-view observation for an agent with global goal awareness.
    
    Args:
        position: [2] agent position
        goal: [2] agent goal (THIS agent's specific goal)
        other_agents: List of other agent positions
        obstacles: [num_obstacles, 2] obstacle positions
        grid_size: size of the grid
        fov_size: size of the field of view
        all_goals: [num_agents, 2] ALL agent goals for global awareness
        agent_idx: index of this agent (to identify which goal is theirs)
        
    Returns:
        fov: [fov_size*fov_size*3] flattened FOV observation with 3 channels:
             Channel 0: obstacles/agents, Channel 1: MY goal, Channel 2: OTHER goals
    """
    # Create FOV grid centered on agent
    fov_grid = torch.zeros(fov_size, fov_size, 3)  # 3 channels: obstacles/agents, MY goal, OTHER goals
    
    center = fov_size // 2
    agent_x, agent_y = int(position[0].item()), int(position[1].item())
    
    # Fill FOV grid
    for fov_x in range(fov_size):
        for fov_y in range(fov_size):
            # Calculate world coordinates
            world_x = agent_x + (fov_x - center)
            world_y = agent_y + (fov_y - center)
            
            # Check if within grid bounds
            if 0 <= world_x < grid_size and 0 <= world_y < grid_size:
                world_pos = torch.tensor([world_x, world_y], dtype=torch.float32)
                
                # Check for obstacles
                for obstacle in obstacles:
                    if torch.equal(world_pos, obstacle):
                        fov_grid[fov_x, fov_y, 0] = 1.0
                        break
                
                # Check for other agents
                for other_pos in other_agents:
                    if torch.equal(world_pos, other_pos):
                        fov_grid[fov_x, fov_y, 0] = 0.5  # Different value for agents
                        break
                
                # Check for MY specific goal
                if torch.equal(world_pos, goal):
                    fov_grid[fov_x, fov_y, 1] = 1.0  # Channel 1: MY goal
                
                # Check for OTHER agents' goals (if global goal info is provided)
                if all_goals is not None and agent_idx is not None:
                    for other_agent_idx, other_goal in enumerate(all_goals):
                        if other_agent_idx != agent_idx and torch.equal(world_pos, other_goal):
                            fov_grid[fov_x, fov_y, 2] = 1.0  # Channel 2: OTHER goals
                            break
            else:
                # Out of bounds - mark as obstacle
                fov_grid[fov_x, fov_y, 0] = 1.0
    
    return fov_grid.flatten()

def generate_pretraining_batch(batch_size: int = 32, sequence_length: int = 10, grid_size: int = 3, 
                              num_agents: int = 2, fov_size: int = 7, num_obstacles: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List]:
    """
    Generate a batch of pre-training data.
    
    Args:
        batch_size: number of scenarios in batch
        sequence_length: length of each trajectory
        grid_size: size of the grid
        num_agents: number of agents
        fov_size: size of field of view
        num_obstacles: number of obstacles
        
    Returns:
        batch_fovs: [batch_size, sequence_length, num_agents, fov_size*fov_size*2]
        batch_actions: [batch_size, sequence_length, num_agents]
        batch_positions: [batch_size, sequence_length, num_agents, 2]
        batch_goals: [batch_size, num_agents, 2]
        batch_obstacles: List of obstacle tensors
    """
    batch_fovs = []
    batch_actions = []
    batch_positions = []
    batch_goals = []
    batch_obstacles = []
    
    for _ in range(batch_size):
        # Generate scenario
        positions, goals, obstacles = generate_simple_scenario(grid_size, num_agents)
        
        # Generate heuristic trajectory for RL training
        traj_positions, traj_actions = generate_expert_trajectory(positions, goals, obstacles, grid_size, max_steps=sequence_length)
        
        # If trajectory is shorter than sequence_length, pad with final positions
        actual_length = traj_positions.shape[0]
        if actual_length < sequence_length:
            # Pad positions (repeat final position)
            final_pos = traj_positions[-1:].repeat(sequence_length - actual_length, 1, 1)
            traj_positions = torch.cat([traj_positions, final_pos], dim=0)
            
            # Pad actions with -1 as "episode done" marker
            done_actions = torch.full((sequence_length - actual_length, num_agents), -1, dtype=torch.long)
            traj_actions = torch.cat([traj_actions, done_actions], dim=0)
        
        # Generate FOV observations for each timestep
        scenario_fovs = []
        for t in range(sequence_length):
            timestep_fovs = []
            for agent_idx in range(num_agents):
                other_agents = [traj_positions[t, j] for j in range(num_agents) if j != agent_idx]
                fov = create_fov_observation(
                    traj_positions[t, agent_idx], 
                    goals[agent_idx], 
                    other_agents, 
                    obstacles, 
                    grid_size, 
                    fov_size,
                    all_goals=goals,  # Pass ALL goals for global awareness
                    agent_idx=agent_idx  # Pass agent index to identify their goal
                )
                timestep_fovs.append(fov)
            scenario_fovs.append(torch.stack(timestep_fovs))
        
        batch_fovs.append(torch.stack(scenario_fovs))
        batch_actions.append(traj_actions)
        batch_positions.append(traj_positions)
        batch_goals.append(goals)
        batch_obstacles.append(obstacles)
    
    return (
        torch.stack(batch_fovs),
        torch.stack(batch_actions),
        torch.stack(batch_positions),
        torch.stack(batch_goals),
        batch_obstacles  # List of empty tensors
    )

def generate_progressive_batch(batch_size: int = 32, sequence_length: int = 10, grid_size: int = 3, 
                             num_agents: int = 2, num_obstacles: int = 2, fov_size: int = 7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List]:
    """
    Generate a batch of progressive pre-training data with obstacles.
    
    Args:
        batch_size: number of scenarios in batch
        sequence_length: length of each trajectory  
        grid_size: size of the grid (3, 5, or 7)
        num_agents: number of agents (2)
        num_obstacles: number of obstacles (2)
        fov_size: size of field of view
        
    Returns:
        batch_fovs: [batch_size, sequence_length, num_agents, fov_size*fov_size*2]
        batch_actions: [batch_size, sequence_length, num_agents]
        batch_positions: [batch_size, sequence_length, num_agents, 2]
        batch_goals: [batch_size, num_agents, 2]
        batch_obstacles: List of obstacle tensors [num_obstacles, 2]
    """
    batch_fovs = []
    batch_actions = []
    batch_positions = []
    batch_goals = []
    batch_obstacles = []
    
    for batch_idx in range(batch_size):
        # Generate scenario with obstacles
        agent_positions, goal_positions, obstacles = generate_scenario_with_obstacles(
            grid_size=grid_size, 
            num_agents=num_agents, 
            num_obstacles=num_obstacles
        )
        
        # Generate heuristic trajectory for RL training with obstacles
        traj_positions, traj_actions = generate_expert_trajectory_with_obstacles(
            agent_positions, goal_positions, obstacles, grid_size, max_steps=sequence_length
        )
        
        # If trajectory is shorter than sequence_length, pad with final positions (not "stay" actions)
        actual_length = traj_positions.shape[0]
        if actual_length < sequence_length:
            # Pad positions (repeat final position)
            final_pos = traj_positions[-1:].repeat(sequence_length - actual_length, 1, 1)
            traj_positions = torch.cat([traj_positions, final_pos], dim=0)
            
            # Pad actions (use -1 as "episode done" marker, we'll mask these in loss)
            done_actions = torch.full((sequence_length - actual_length, num_agents), -1, dtype=torch.long)
            traj_actions = torch.cat([traj_actions, done_actions], dim=0)
        
        # Generate FOV observations for each timestep
        scenario_fovs = []
        for t in range(sequence_length):
            timestep_fovs = []
            for agent_idx in range(num_agents):
                # Get other agent positions at this timestep
                other_agents = [traj_positions[t, j] for j in range(num_agents) if j != agent_idx]
                
                # Create FOV observation
                fov = create_fov_observation(
                    traj_positions[t, agent_idx], 
                    goal_positions[agent_idx], 
                    other_agents, 
                    obstacles, 
                    grid_size, 
                    fov_size,
                    all_goals=goal_positions,  # Pass ALL goals for global awareness
                    agent_idx=agent_idx  # Pass agent index to identify their goal
                )
                timestep_fovs.append(fov)
            scenario_fovs.append(torch.stack(timestep_fovs))
        
        batch_fovs.append(torch.stack(scenario_fovs))
        batch_actions.append(traj_actions)
        batch_positions.append(traj_positions)
        batch_goals.append(goal_positions)
        batch_obstacles.append(obstacles)
    
    return (
        torch.stack(batch_fovs),
        torch.stack(batch_actions),
        torch.stack(batch_positions),
        torch.stack(batch_goals),
        batch_obstacles  # List of obstacle tensors
    )

def generate_scenario_with_obstacles(grid_size: int = 3, num_agents: int = 2, num_obstacles: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a scenario with agents, goals, and obstacles that don't overlap.
    Ensures agents don't start at their goal positions.
    
    Args:
        grid_size: Size of the grid
        num_agents: Number of agents 
        num_obstacles: Number of obstacles
        
    Returns:
        positions: [num_agents, 2] - agent starting positions
        goals: [num_agents, 2] - goal positions
        obstacles: [num_obstacles, 2] - obstacle positions
    """
    max_attempts = 100
    for attempt in range(max_attempts):
        # Get all possible positions
        all_positions = []
        for x in range(grid_size):
            for y in range(grid_size):
                all_positions.append([x, y])
        
        # Randomly select positions for agents, goals, and obstacles (no overlaps)
        total_needed = num_agents * 2 + num_obstacles  # agents + goals + obstacles
        if total_needed > len(all_positions):
            raise ValueError(f"Not enough positions in {grid_size}x{grid_size} grid for {total_needed} entities")
        
        selected_positions = random.sample(all_positions, total_needed)
        
        # Split positions
        agent_positions = torch.tensor(selected_positions[:num_agents], dtype=torch.float32)
        goal_positions = torch.tensor(selected_positions[num_agents:num_agents*2], dtype=torch.float32)
        obstacle_positions = torch.tensor(selected_positions[num_agents*2:num_agents*2+num_obstacles], dtype=torch.float32)
        
        # Check that no agent starts at its goal (minimum distance of 1.0)
        valid_scenario = True
        for i in range(num_agents):
            distance = torch.norm(agent_positions[i] - goal_positions[i]).item()
            if distance < 1.0:  # Agent too close to its own goal
                valid_scenario = False
                break
        
        if valid_scenario:
            return agent_positions, goal_positions, obstacle_positions
    
    # If we couldn't generate a valid scenario, fall back to original method
    print(f"⚠️ Warning: Could not generate scenario with min distance constraint after {max_attempts} attempts")
    selected_positions = random.sample(all_positions, total_needed)
    agent_positions = torch.tensor(selected_positions[:num_agents], dtype=torch.float32)
    goal_positions = torch.tensor(selected_positions[num_agents:num_agents*2], dtype=torch.float32)
    obstacle_positions = torch.tensor(selected_positions[num_agents*2:num_agents*2+num_obstacles], dtype=torch.float32)
    
    return agent_positions, goal_positions, obstacle_positions

def generate_expert_trajectory_with_obstacles(initial_positions: torch.Tensor, goals: torch.Tensor, 
                                            obstacles: torch.Tensor, grid_size: int, max_steps: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate trajectory using simple heuristics for RL training (no A* expert).
    This allows the model to learn optimal trajectories through trial and error with RL.
    
    Args:
        initial_positions: [num_agents, 2] starting positions
        goals: [num_agents, 2] goal positions
        obstacles: [num_obstacles, 2] obstacle positions
        grid_size: size of the grid
        max_steps: maximum trajectory length
        
    Returns:
        trajectory_positions: [max_steps, num_agents, 2] position sequence
        trajectory_actions: [max_steps, num_agents] action sequence (heuristic-based)
    """
    num_agents = initial_positions.shape[0]
    current_positions = initial_positions.clone()
    
    trajectory_positions = []
    trajectory_actions = []
    
    for step in range(max_steps):
        trajectory_positions.append(current_positions.clone())
        
        step_actions = []
        next_positions = []
        
        # Get actions for each agent
        for agent_idx in range(num_agents):
            other_agents = [current_positions[j] for j in range(num_agents) if j != agent_idx]
            action = get_optimal_action_with_obstacles(
                current_positions[agent_idx], 
                goals[agent_idx], 
                other_agents, 
                obstacles, 
                grid_size
            )
            step_actions.append(action)
            
            # Calculate next position
            action_deltas = torch.tensor([
                [0, 0],   # 0: Stay
                [1, 0],   # 1: Right
                [0, 1],   # 2: Up
                [-1, 0],  # 3: Left
                [0, -1],  # 4: Down
            ], dtype=torch.float32)
            
            next_pos = current_positions[agent_idx] + action_deltas[action]
            next_positions.append(next_pos)
        
        trajectory_actions.append(torch.tensor(step_actions))
        current_positions = torch.stack(next_positions)
        
        # Check if all agents reached goals - STOP EARLY!
        goal_distances = torch.norm(current_positions - goals, dim=-1)
        agents_at_goal = goal_distances <= 0.1  # Small tolerance
        if torch.all(agents_at_goal).item():  # Convert to Python bool
            break
    
    # Only return actual steps, don't pad with stays
    if len(trajectory_positions) == 0:
        trajectory_positions = [initial_positions]
        trajectory_actions = [torch.zeros(num_agents, dtype=torch.long)]
    
    return torch.stack(trajectory_positions), torch.stack(trajectory_actions)

def get_optimal_action_with_obstacles(current_pos: torch.Tensor, goal_pos: torch.Tensor, 
                                     other_agents: List[torch.Tensor] = None, 
                                     obstacles: torch.Tensor = None, grid_size: int = 3) -> int:
    """
    Get action using simple heuristic with obstacle avoidance for RL training.
    This allows the model to learn optimal paths through RL rewards/penalties.
    
    Args:
        current_pos: [2] current position
        goal_pos: [2] goal position
        other_agents: List of other agent positions to avoid
        obstacles: [num_obstacles, 2] obstacle positions
        grid_size: size of the grid
        
    Returns:
        action: int (0-4) - heuristic action with basic obstacle avoidance
    """
    # Use the same heuristic logic as get_optimal_action
    return get_optimal_action(current_pos, goal_pos, other_agents, obstacles, grid_size)

def pretrain_model(model, optimizer, config: Dict[str, Any], device: str = 'cpu', 
                  epochs: int = 2, batches_per_epoch: int = 50) -> Dict[str, List[float]]:
    """
    Pre-train the SNN model on simple 3x3 scenarios using reinforcement learning.
    
    Args:
        model: SNN model to train
        optimizer: optimizer for training
        config: configuration dictionary
        device: device to train on
        epochs: number of pre-training epochs
        batches_per_epoch: number of batches per epoch
        
    Returns:
        training_history: dictionary with training metrics
    """
    print("🎯 Starting SNN Pre-training with RL on 3x3 grids...")
    print(f"   📊 Epochs: {epochs}, Batches per epoch: {batches_per_epoch}")
    print(f"   🎮 Scenario: 3x3 grid, 2 agents, 2 goals, no obstacles")
    print("   🚀 Using Reinforcement Learning instead of Imitation Learning")
    
    # Initialize spike monitor for network health tracking
    spike_monitor = create_spike_monitor(config)
    spike_tracker = SpikeTracker(spike_monitor)
    
    model.train()
    history = {
        'loss': [],
        'agents_reached_goal': [],
        'goal_progress': [],
        'collision_rate': [],
        'network_health': []
    }
    
    # Pre-training specific config
    pretrain_config = config.copy()
    pretrain_config.update({
        'board_size': [3, 3],
        'num_agents': 2,
        'sequence_length': 10,
        'batch_size': 32,
        'learning_rate': config.get('learning_rate', 1e-4) * 2,  # Slightly higher LR for pre-training
    })
    
    batch_size = pretrain_config['batch_size']
    sequence_length = pretrain_config['sequence_length']
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_goal_rates = []
        epoch_goal_progress = []
        epoch_collisions = []
        
        print(f"\n🏃 Pre-training Epoch {epoch + 1}/{epochs}")
        
        # Create progress bar
        pbar = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch + 1}")
        
        for batch_idx in pbar:
            # Generate batch
            batch_fovs, batch_actions, batch_positions, batch_goals, _ = generate_pretraining_batch(
                batch_size=batch_size,
                sequence_length=sequence_length,
                grid_size=3,
                num_agents=2,
                fov_size=7
            )
            
            # Move to device
            batch_fovs = batch_fovs.to(device)
            batch_actions = batch_actions.to(device)
            batch_positions = batch_positions.to(device)
            batch_goals = batch_goals.to(device)  # Already stacked in generate_pretraining_batch
            
            optimizer.zero_grad()
            
            # Reset SNN state
            safe_reset_snn(model)
            
            total_loss = 0.0
            agents_reached_goal = 0
            total_agents = 0
            goal_tolerance = 0.5  # Distance threshold to consider agent reached goal
            
            # Process sequence
            for t in range(sequence_length):
                # Get model output
                timestep_fovs = batch_fovs[:, t]  # [batch_size, num_agents, fov_features]
                target_actions = batch_actions[:, t]  # [batch_size, num_agents]
                
                # Get actual number of agents from the batch
                actual_num_agents = timestep_fovs.shape[1]
                
                # Flatten for model input
                model_input = timestep_fovs.reshape(batch_size * actual_num_agents, -1)  # [batch_size * num_agents, features]
                
                # Forward pass with spike monitoring
                current_positions = batch_positions[:, t]  # [batch_size, num_agents, 2]
                current_goals = batch_goals  # [batch_size, num_agents, 2]
                
                if spike_tracker.enabled:
                    # Get model output with spike information
                    outputs, spike_info = model(model_input, positions=current_positions, goals=current_goals, return_spikes=True)
                    
                    # Record spike activity for specific layers
                    if spike_info:
                        # Debug print to see what's in spike_info
                        if batch_idx == 0 and t == 0:
                            print(f"   🔍 [PRETRAINING] spike_info keys: {list(spike_info.keys())}")
                            for key, value in spike_info.items():
                                if isinstance(value, torch.Tensor):
                                    print(f"   🔍 [PRETRAINING] {key}: shape={value.shape}, mean={value.mean().item():.6f}, max={value.max().item():.6f}")
                        
                        # Track the three critical layers
                        if 'global_features' in spike_info:
                            spike_tracker.monitor.register_layer('global_features')
                            spike_tracker('global_features', spike_info['global_features'])
                        
                        if 'snn_output' in spike_info:
                            spike_tracker.monitor.register_layer('snn_output')
                            spike_tracker('snn_output', spike_info['snn_output'])
                        
                        if 'action_logits' in spike_info:
                            spike_tracker.monitor.register_layer('action_logits')
                            spike_tracker('action_logits', spike_info['action_logits'])
                else:
                    # Standard forward pass without spike monitoring
                    outputs = model(model_input, positions=current_positions, goals=current_goals)  # [batch_size * num_agents, num_actions]
                
                # CRITICAL DEBUG: Check model output magnitude and distribution (pretraining)
                if t == 0 and batch_idx == 0:  # First timestep of first batch
                    output_magnitude = torch.abs(outputs).max().item()
                    output_mean = outputs.mean().item()
                    output_std = outputs.std().item()
                    output_range = [outputs.min().item(), outputs.max().item()]
                    
                    print(f"   🔍 [PRETRAINING] MODEL OUTPUT DEBUG:")
                    print(f"      📊 Shape: {outputs.shape}")
                    print(f"      📊 Magnitude: {output_magnitude:.6f}")
                    print(f"      📊 Mean: {output_mean:.6f}")
                    print(f"      📊 Std: {output_std:.6f}")
                    print(f"      📊 Range: [{output_range[0]:.6f}, {output_range[1]:.6f}]")
                    
                    # Check if output is all zeros
                    if output_magnitude < 1e-10:
                        print(f"      ❌ [PRETRAINING] MODEL OUTPUT IS ESSENTIALLY ZERO!")
                        print(f"      🔍 Checking for gradient flow issues...")
                        
                        # Check if gradients are flowing
                        if outputs.requires_grad:
                            print(f"      ✅ Output requires gradients")
                        else:
                            print(f"      ❌ Output does NOT require gradients")
                    else:
                        print(f"      ✅ [PRETRAINING] Model output has magnitude")
                
                # Track model output for debugging (but don't treat as SNN layer)
                if spike_tracker.enabled:
                    try:
                        # Register the layer first if needed
                        spike_tracker.monitor.register_layer('model_output')
                        # Track raw output values to see actual magnitudes
                        spike_tracker('model_output', outputs)
                        
                        # Debug: Check output magnitude
                        if t == 0 and batch_idx == 0:
                            output_magnitude = torch.abs(outputs).mean().item()
                            print(f"   � Output magnitude: {output_magnitude:.6f}")
                            
                        # Debug: Log real output magnitudes every few batches
                        if batch_idx % 10 == 0:
                            print(f"🔍 Output range: {outputs.abs().min().item():.3f}-{outputs.abs().max().item():.3f}, mean: {outputs.abs().mean().item():.3f}")
                    except Exception as e:
                        if t == 0 and batch_idx == 0:
                            print(f"   ⚠️ Output tracking warning: {e}")
                
                # Debug: Show available SNN layers 
                debug_snn_layers(model, spike_tracker, t, batch_idx)
                
                # Comprehensive SNN membrane potential and spike tracking (AdaptiveLIFNode debugging)
                if spike_tracker.enabled and hasattr(model, 'snn_block'):
                    for name, module in model.snn_block.named_modules():
                        if hasattr(module, '_last_spike_output'):
                            try:
                                # Get spike output from AdaptiveLIFNode
                                membrane_spikes = module._last_spike_output
                                if isinstance(membrane_spikes, torch.Tensor) and membrane_spikes.numel() > 0:
                                    # Register layer and record spikes
                                    layer_name = f'snn_{name}'
                                    spike_tracker.monitor.register_layer(layer_name)
                                    spike_tracker(layer_name, membrane_spikes)
                                    
                                    # Detailed membrane potential and spike analysis
                                    if t == 0 and batch_idx == 0:  # Log first timestep of first batch
                                        total_neurons = membrane_spikes.numel()
                                        total_spikes = membrane_spikes.sum().item()
                                        spike_rate = total_spikes / total_neurons if total_neurons > 0 else 0.0
                                        
                                        # Get membrane potential if available
                                        membrane_potential = "N/A"
                                        threshold_value = "N/A"
                                        if hasattr(module, 'v'):
                                            try:
                                                if module.v is not None and isinstance(module.v, torch.Tensor):
                                                    membrane_potential = f"{module.v.mean().item():.6f}"
                                            except Exception:
                                                membrane_potential = "Error"
                                        
                                        # Get threshold value
                                        if hasattr(module, 'v_threshold'):
                                            try:
                                                if module.v_threshold is not None and isinstance(module.v_threshold, torch.Tensor):
                                                    threshold_value = f"{module.v_threshold.mean().item():.6f}"
                                            except Exception:
                                                threshold_value = "Error"
                                        
                                        print(f"   🧠 [PRETRAINING] {name}: spikes={total_spikes}/{total_neurons} (rate={spike_rate:.3f})")
                                        print(f"      🧠 membrane potential: {membrane_potential}")
                                        print(f"      🎯 threshold: {threshold_value}")
                                        print(f"      ⚡ using actual spikes from SpikingJelly")
                            except Exception as e:
                                # Handle temporal_lif IndexError and other membrane tracking issues
                                if "temporal_lif" in name and "tuple index out of range" in str(e):
                                    if t == 0 and batch_idx == 0:
                                        print(f"   ⚠️ Membrane tracking warning for {name}: {e}")
                                elif t == 0 and batch_idx == 0:
                                    print(f"   ⚠️ SNN layer {name} tracking error: {e}")
                
                # Comprehensive SNN membrane potential and spike tracking
                if spike_tracker.enabled and hasattr(model, 'snn_block'):
                    for name, module in model.snn_block.named_modules():
                        if hasattr(module, '_last_spike_output'):
                            try:
                                # Get spike output from AdaptiveLIFNode
                                membrane_spikes = module._last_spike_output
                                if isinstance(membrane_spikes, torch.Tensor) and membrane_spikes.numel() > 0:
                                    # Register layer and record spikes
                                    layer_name = f'snn_{name}'
                                    spike_tracker.monitor.register_layer(layer_name)
                                    spike_tracker(layer_name, membrane_spikes)
                                    
                                    # Detailed membrane potential and spike analysis
                                    if t == 0 and batch_idx == 0:  # Log first timestep of first batch
                                        total_neurons = membrane_spikes.numel()
                                        total_spikes = membrane_spikes.sum().item()
                                        spike_rate = total_spikes / total_neurons if total_neurons > 0 else 0.0
                                        
                                        # Get membrane potential if available
                                        membrane_potential = "N/A"
                                        if hasattr(module, 'v'):
                                            try:
                                                if module.v is not None and isinstance(module.v, torch.Tensor):
                                                    membrane_potential = f"{module.v.mean().item():.6f}"
                                            except Exception:
                                                membrane_potential = "Error"
                                        
                                        print(f"   🧠 {name}: spikes={total_spikes}/{total_neurons} (rate={spike_rate:.3f})")
                                        print(f"      🧠 membrane potential: {membrane_potential}")
                                        print(f"      ⚡ using actual spikes from SpikingJelly")
                            except Exception as e:
                                # Handle temporal_lif IndexError and other membrane tracking issues
                                if "temporal_lif" in name and "tuple index out of range" in str(e):
                                    if t == 0 and batch_idx == 0:
                                        print(f"   ⚠️ Membrane tracking warning for {name}: {e}")
                                elif t == 0 and batch_idx == 0:
                                    print(f"   ⚠️ SNN layer {name} tracking error: {e}")
                
                # Track SNN spikes using actual outputs from AdaptiveLIFNode (fallback)
                if spike_tracker.enabled and hasattr(model, 'snn_block'):
                    for name, module in model.snn_block.named_modules():
                        # Only record if module has stored spike output
                        if hasattr(module, '_last_spike_output'):
                            membrane_spikes = module._last_spike_output
                            if isinstance(membrane_spikes, torch.Tensor) and membrane_spikes.numel() > 0:
                                # Register layer and record spikes
                                layer_name = f'snn_{name}'
                                spike_tracker.monitor.register_layer(layer_name)
                                spike_tracker(layer_name, membrane_spikes)
                
                # Legacy spike counting (for reference)
                if hasattr(model, 'snn_block'):
                    total_spikes = 0
                    try:
                        for name, module in model.snn_block.named_modules():
                            if not isinstance(name, str):
                                continue
                            if hasattr(module, 'spike_count') and module.spike_count is not None:
                                total_spikes += module.spike_count.sum().item()
                    except Exception as e:
                        if t == 0 and batch_idx == 0:
                            print(f"   ⚠️ Legacy spike counting error: {e}")
                    if t == 0 and batch_idx == 0:  # Log once per epoch
                        print(f"   🧠 SNN Spikes this timestep: {total_spikes}")
                
                # Reshape outputs
                outputs = outputs.reshape(batch_size, actual_num_agents, -1)  # [batch_size, num_agents, num_actions]
                
                # Compute loss (cross-entropy) with masking for episode done actions
                flat_outputs = outputs.reshape(-1, outputs.shape[-1])
                flat_targets = target_actions.reshape(-1).long()
                
                # Mask out "episode done" actions (-1)
                valid_mask = flat_targets >= 0
                if valid_mask.sum() > 0:  # Only compute loss if there are valid actions
                    timestep_loss = F.cross_entropy(
                        flat_outputs[valid_mask], 
                        flat_targets[valid_mask]
                    )
                else:
                    timestep_loss = torch.tensor(0.0, device=outputs.device)
                
                # Track RL-style goal reaching (excluding collided agents)
                batch_final_positions = batch_positions[:, -1]  # [batch_size, num_agents, 2]
                batch_final_distances = torch.norm(batch_final_positions - batch_goals, dim=-1)
                batch_agents_at_goal = batch_final_distances <= goal_tolerance  # [batch_size, num_agents]
                
                # Check for collisions during the episode to exclude collided agents from goal count
                collision_detected = torch.zeros((batch_size, actual_num_agents), dtype=torch.bool)
                for b in range(batch_size):
                    for t in range(sequence_length):
                        # Check collisions between all agent pairs at timestep t
                        for i in range(actual_num_agents):
                            for j in range(i + 1, actual_num_agents):
                                pos_i = batch_positions[b, t, i]
                                pos_j = batch_positions[b, t, j]
                                if torch.equal(pos_i, pos_j):
                                    collision_detected[b, i] = True
                                    collision_detected[b, j] = True
                
                # Only count goal reaching for non-collided agents
                valid_goal_reaching = batch_agents_at_goal & (~collision_detected)
                agents_reached_goal += torch.sum(valid_goal_reaching).item()
                
                # Count ALL agents in denominator for honest goal rate percentage
                total_agents += batch_size * actual_num_agents
                
                # Removed spike sparsity regularization - was overwhelming the model
                # Using adaptive thresholds in AdaptiveLIFNode for spike regulation instead
                
                # Add enhanced oscillation penalty for pre-training
                if config.get('use_enhanced_oscillation_detection', True):
                    try:
                        from loss_compute import compute_enhanced_oscillation_penalty
                        oscillation_penalty = compute_enhanced_oscillation_penalty(
                            batch_positions[:, :t+1], batch_id=batch_idx, 
                            oscillation_weight=config.get('oscillation_penalty_weight', 7.0),  # Same default as main training
                            goal_positions=batch_goals
                        )
                        timestep_loss += oscillation_penalty
                    except ImportError:
                        # Fallback to simple oscillation penalty
                        from loss_compute import compute_simple_oscillation_penalty
                        oscillation_penalty = compute_simple_oscillation_penalty(
                            batch_positions[:, :t+1], batch_id=batch_idx,
                            oscillation_weight=config.get('oscillation_penalty_weight', 7.0),  # Same default as main training
                            goal_positions=batch_goals
                        )
                        timestep_loss += oscillation_penalty
                
                total_loss += timestep_loss
            
            # Average loss over sequence
            total_loss = total_loss / sequence_length
            
            # Add progressive goal rewards (bigger rewards for more agents reaching goals)
            final_positions = batch_positions[:, -1]  # [batch_size, num_agents, 2]
            initial_distances = torch.norm(batch_positions[:, 0] - batch_goals, dim=-1)
            final_distances = torch.norm(final_positions - batch_goals, dim=-1)
            progress = torch.mean(initial_distances - final_distances)
            
            # Count how many agents reached their goals (within tolerance)
            goal_tolerance = 0.5
            agents_at_goal = final_distances <= goal_tolerance  # [batch_size, num_agents]
            goals_reached_per_batch = torch.sum(agents_at_goal, dim=1).float()  # [batch_size]
            
            # Progressive rewards: 1 goal = small, 2 goals = big bonus!
            goal_rewards = torch.zeros_like(goals_reached_per_batch)
            goal_rewards += (goals_reached_per_batch >= 1).float() * 0.5  # 1 goal reached
            goal_rewards += (goals_reached_per_batch >= 2).float() * 1.5  # Both goals reached - BIG BONUS!
            
            average_goal_reward = torch.mean(goal_rewards)
            goal_progress_loss = -progress * 0.1 - average_goal_reward * 0.2  # Stronger reward signal
            
            total_loss += goal_progress_loss
            
            # Backward pass
            total_loss.backward()
            
            # Adaptive gradient clipping with proper norm measurement
            max_norm = config.get('gradient_clip_max_norm', 2.0)
            
            # Measure gradient norm WITHOUT modifying gradients
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.detach().data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            unclipped_grad_norm = total_grad_norm ** 0.5
            
            # Only apply clipping if gradients are significantly large
            explosion_threshold = max_norm * 3.0  # Only clip if > 6.0
            if unclipped_grad_norm > explosion_threshold:
                # Apply clipping for true gradient explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                # Recompute gradient norm after clipping
                clipped_norm = sum(
                    p.grad.detach().data.norm(2).item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5
                grad_norm = clipped_norm
                if batch_idx % 10 == 0:
                    print(f"   🚨 EXPLOSION! Gradient clipped: {unclipped_grad_norm:.3f} → {grad_norm:.3f}")
            else:
                # No clipping - let gradients flow naturally
                grad_norm = unclipped_grad_norm
                if batch_idx % 20 == 0:
                    print(f"   📊 Gradient norm: {grad_norm:.3f} (no clipping needed)")
            
            # Log detailed gradient info
            if batch_idx % 20 == 0:
                # Check specific layer gradients that might be getting suppressed
                for name, param in model.named_parameters():
                    if param.grad is not None and 'global_features' in name:
                        layer_grad_norm = param.grad.norm().item()
                        print(f"   🧠 {name} grad norm: {layer_grad_norm:.6f}")
            
            optimizer.step()
            
            # Calculate RL-style goal reaching rate instead of accuracy
            goal_reaching_rate = agents_reached_goal / total_agents if total_agents > 0 else 0.0
            goal_progress_metric = progress.item()
            
            # Calculate collision rate (simplified)
            collision_count = 0
            for b in range(batch_size):
                for t in range(sequence_length):
                    pos1 = batch_positions[b, t, 0]
                    pos2 = batch_positions[b, t, 1]
                    if torch.equal(pos1, pos2):
                        collision_count += 1
            collision_rate = collision_count / (batch_size * sequence_length)
            
            # Store metrics
            epoch_losses.append(total_loss.item())
            epoch_goal_rates.append(goal_reaching_rate)
            epoch_goal_progress.append(goal_progress_metric)
            epoch_collisions.append(collision_rate)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Goal Rate': f'{goal_reaching_rate:.3f}',
                'Progress': f'{goal_progress_metric:.3f}',
                'Collisions': f'{collision_rate:.3f}'
            })
        
        # Store epoch metrics
        history['loss'].append(np.mean(epoch_losses))
        history['agents_reached_goal'].append(np.mean(epoch_goal_rates))
        history['goal_progress'].append(np.mean(epoch_goal_progress))
        history['collision_rate'].append(np.mean(epoch_collisions))
        
        # Calculate network health score
        if spike_tracker.enabled:
            network_health = spike_tracker.summary(epoch + 1)
            history['network_health'].append(network_health)
            print(f"🧠 Network Health Score: {network_health:.3f}/1.0 {'✅' if network_health > 0.5 else '⚠️' if network_health > 0.2 else '❌'}")
        else:
            history['network_health'].append(1.0)  # Default if not tracking
        
        # Print epoch summary
        print(f"📊 Epoch {epoch + 1} Summary:")
        print(f"   Loss: {history['loss'][-1]:.4f}")
        print(f"   Goal Reaching Rate: {history['agents_reached_goal'][-1]:.3f}")
        print(f"   Goal Progress: {history['goal_progress'][-1]:.3f}")
        print(f"   Collisions: {history['collision_rate'][-1]:.3f}")
        print(f"   Network Health: {history['network_health'][-1]:.3f}")
        
        # Print epoch summary
        print(f"   📈 Epoch {epoch + 1} Summary:")
        print(f"      Loss: {history['loss'][-1]:.4f}")
        print(f"      Agents Reached Goal: {history['agents_reached_goal'][-1]:.3f}")
        print(f"      Goal Progress: {history['goal_progress'][-1]:.3f}")
        print(f"      Collision Rate: {history['collision_rate'][-1]:.3f}")
    
    print("\n✅ Pre-training completed!")
    print(f"   🎯 Final Goal Rate: {history['agents_reached_goal'][-1]:.3f}")
    print(f"   📈 Final Goal Progress: {history['goal_progress'][-1]:.3f}")
    print(f"   🚫 Final Collision Rate: {history['collision_rate'][-1]:.3f}")
    
    return history

def progressive_pretrain_model(model, optimizer, config: Dict[str, Any], device: str = 'cpu') -> Dict[str, List[float]]:
    """
    Progressive pre-training: 4 epochs 3x3, 3 epochs 5x5, 3 epochs 7x7.
    Each stage uses 2 agents, 2 obstacles, appropriate sequence lengths to avoid excessive STAY actions.
    
    Args:
        model: SNN model to train
        optimizer: optimizer for training
        config: configuration dictionary
        device: device to train on
        
    Returns:
        training_history: dictionary with training metrics for all stages
    """
    print("🎯 Starting Progressive SNN Pre-training...")
    print("   📊 Stage 1: 4 epochs on 3x3 grids")
    print("   📊 Stage 2: 3 epochs on 5x5 grids") 
    print("   📊 Stage 3: 3 epochs on 7x7 grids")
    print("   🎮 Each stage: 2 agents, 2 obstacles, adaptive sequence length")
    
    # Initialize spike monitor for network health tracking
    spike_monitor = create_spike_monitor(config)
    spike_tracker = SpikeTracker(spike_monitor)
    
    model.train()
    
    # Get progressive training stages from config
    pretraining_schedule = config.get('pretraining_schedule', [])
    if not pretraining_schedule:
        # Fallback to hard-coded stages if config schedule not found
        print("⚠️  No pretraining_schedule found in config, using fallback stages")
        stages = [
            (3, 8, 8, 50),   # 3x3: short sequences to avoid too many STAYs
            (5, 15, 25, 40), # 5x5: medium sequences - UPDATED to use config values
            (7, 5, 25, 30),  # 7x7: longer sequences
        ]
    else:
        # Use config schedule: (grid_size, epochs, sequence_length, batches_per_epoch)
        stages = []
        for stage in pretraining_schedule:
            grid_size = stage['grid_size']
            epochs = stage['epochs']
            sequence_length = stage['sequence_length']
            batches_per_epoch = config.get('pretraining_batches_per_epoch', 50)  # Default batches per epoch
            stages.append((grid_size, epochs, sequence_length, batches_per_epoch))
        print(f"📋 Using config pretraining_schedule: {len(stages)} stages")
        for i, (gs, ep, sl, bpe) in enumerate(stages):
            print(f"   Stage {i+1}: {gs}x{gs} grid, {ep} epochs, seq_len={sl}, {bpe} batches/epoch")
    
    # Combined history for all stages
    history = {
        'loss': [],
        'agents_reached_goal': [],
        'goal_progress': [],
        'collision_rate': [],
        'network_health': [],
        'stage_info': []
    }
    
    total_epochs = sum(stage[1] for stage in stages)
    current_epoch = 0
    
    for stage_idx, (grid_size, stage_epochs, sequence_length, batches_per_epoch) in enumerate(stages):
        print(f"\n🏃 === STAGE {stage_idx + 1}: {grid_size}x{grid_size} Grid ===")
        print(f"   📏 Sequence length: {sequence_length}")
        print(f"   📊 Batches per epoch: {batches_per_epoch}")
        print(f"   🎯 Epochs: {stage_epochs}")
        
        # Stage-specific config
        stage_config = config.copy()
        stage_config.update({
            'board_size': [grid_size, grid_size],
            'num_agents': 2,
            'sequence_length': sequence_length,
            'batch_size': 32,
        })
        
        batch_size = stage_config['batch_size']
        
        for epoch in range(stage_epochs):
            current_epoch += 1
            epoch_losses = []
            epoch_goal_rates = []
            epoch_goal_progress = []
            epoch_collisions = []
            
            print(f"\n🏃 Progressive Epoch {current_epoch}/{total_epochs} (Stage {stage_idx + 1}, Grid {grid_size}x{grid_size})")
            
            # Create progress bar
            pbar = tqdm(range(batches_per_epoch), desc=f"Stage {stage_idx + 1} Epoch {epoch + 1}")
            
            for batch_idx in pbar:
                # Generate batch for current stage
                batch_fovs, batch_actions, batch_positions, batch_goals, batch_obstacles = generate_progressive_batch(
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    grid_size=grid_size,
                    num_agents=2,
                    num_obstacles=2,
                    fov_size=7
                )
                
                # Move to device
                batch_fovs = batch_fovs.to(device)
                batch_actions = batch_actions.to(device)
                batch_positions = batch_positions.to(device)
                batch_goals = batch_goals.to(device)
                
                optimizer.zero_grad()
                
                # Reset SNN state
                safe_reset_snn(model)
                
                total_loss = 0.0
                agents_reached_goal = 0
                total_agents = 0
                goal_tolerance = 0.5  # Distance threshold to consider agent reached goal
                
                # Process sequence
                for t in range(sequence_length):
                    # Get model output
                    timestep_fovs = batch_fovs[:, t]  # [batch_size, num_agents, fov_features]
                    target_actions = batch_actions[:, t]  # [batch_size, num_agents]
                    
                    # Get actual number of agents from the batch
                    actual_num_agents = timestep_fovs.shape[1]
                    
                    # Flatten for model input
                    model_input = timestep_fovs.reshape(batch_size * actual_num_agents, -1)
                    
                    # Forward pass with spike monitoring
                    current_positions = batch_positions[:, t]  # [batch_size, num_agents, 2]
                    current_goals = batch_goals  # [batch_size, num_agents, 2]
                    
                    if spike_tracker.enabled:
                        # Get model output with spike information
                        outputs, spike_info = model(model_input, positions=current_positions, goals=current_goals, return_spikes=True)
                        
                        # Record spike activity for specific layers
                        if spike_info:
                            # Debug print to see what's in spike_info
                            if batch_idx == 0 and t == 0:
                                print(f"   🔍 [PROGRESSIVE PRETRAINING] spike_info keys: {list(spike_info.keys())}")
                                for key, value in spike_info.items():
                                    if isinstance(value, torch.Tensor):
                                        print(f"   🔍 [PROGRESSIVE PRETRAINING] {key}: shape={value.shape}, mean={value.mean().item():.6f}, max={value.max().item():.6f}")
                            
                            # Track the three critical layers
                            if 'global_features' in spike_info:
                                spike_tracker.monitor.register_layer('global_features')
                                spike_tracker('global_features', spike_info['global_features'])
                            
                            if 'snn_output' in spike_info:
                                spike_tracker.monitor.register_layer('snn_output')
                                spike_tracker('snn_output', spike_info['snn_output'])
                            
                            if 'action_logits' in spike_info:
                                spike_tracker.monitor.register_layer('action_logits')
                                spike_tracker('action_logits', spike_info['action_logits'])
                    else:
                        # Standard forward pass without spike monitoring
                        outputs = model(model_input, positions=current_positions, goals=current_goals)
                    
                    # CRITICAL DEBUG: Check model output magnitude and distribution (progressive pretraining)
                    if t == 0 and batch_idx == 0:  # First timestep of first batch
                        output_magnitude = torch.abs(outputs).max().item()
                        output_mean = outputs.mean().item()
                        output_std = outputs.std().item()
                        output_range = [outputs.min().item(), outputs.max().item()]
                        
                        print(f"   🔍 [PROGRESSIVE PRETRAINING] MODEL OUTPUT DEBUG:")
                        print(f"      📊 Shape: {outputs.shape}")
                        print(f"      📊 Magnitude: {output_magnitude:.6f}")
                        print(f"      📊 Mean: {output_mean:.6f}")
                        print(f"      📊 Std: {output_std:.6f}")
                        print(f"      📊 Range: [{output_range[0]:.6f}, {output_range[1]:.6f}]")
                        
                        # Check if output is all zeros
                        if output_magnitude < 1e-10:
                            print(f"      ❌ [PROGRESSIVE PRETRAINING] MODEL OUTPUT IS ESSENTIALLY ZERO!")
                            print(f"      🔍 Checking for gradient flow issues...")
                            
                            # Check if gradients are flowing
                            if outputs.requires_grad:
                                print(f"      ✅ Output requires gradients")
                            else:
                                print(f"      ❌ Output does NOT require gradients")
                        else:
                            print(f"      ✅ [PROGRESSIVE PRETRAINING] Model output has magnitude")
                    
                    # Track spike activity for network health (basic tracking using model output)
                    if spike_tracker.enabled:
                        # Track output spikes (assuming outputs represent spike activity)
                        try:
                            # Use separate threshold for model output (much lower than SNN internals)
                            model_output_threshold = config.get('model_output_spike_threshold', 0.1)  # Much lower threshold
                            output_spikes = (torch.abs(outputs) > model_output_threshold).float()
                            actual_spike_rate = output_spikes.mean().item()
                            
                            # Always track activity to ensure layer registration
                            spike_tracker.monitor.register_layer('model_output')
                            # Track raw output values (not spike mask) to see actual magnitudes
                            spike_tracker('model_output', outputs)
                        except Exception as e:
                            if t == 0 and batch_idx == 0:
                                print(f"   ⚠️ Output tracking warning: {e}")
                        
                        # Track AdaptiveLIFNode layers (same as main training)
                        if hasattr(model, 'snn_block'):
                            for name, module in model.snn_block.named_modules():
                                if hasattr(module, '_last_spike_output'):
                                    try:
                                        # Get spike output from AdaptiveLIFNode
                                        membrane_spikes = module._last_spike_output
                                        if isinstance(membrane_spikes, torch.Tensor) and membrane_spikes.numel() > 0:
                                            # Register layer and record spikes
                                            layer_name = f'snn_{name}'
                                            spike_tracker.monitor.register_layer(layer_name)
                                            spike_tracker(layer_name, membrane_spikes)
                                            
                                            # Detailed membrane potential and spike analysis
                                            if t == 0 and batch_idx == 0:  # Log first timestep of first batch
                                                total_neurons = membrane_spikes.numel()
                                                total_spikes = membrane_spikes.sum().item()
                                                spike_rate = total_spikes / total_neurons if total_neurons > 0 else 0.0
                                                
                                                # Get membrane potential if available
                                                membrane_potential = "N/A"
                                                threshold_value = "N/A"
                                                if hasattr(module, 'v'):
                                                    try:
                                                        if module.v is not None and isinstance(module.v, torch.Tensor):
                                                            membrane_potential = f"{module.v.mean().item():.6f}"
                                                    except Exception:
                                                        membrane_potential = "Error"
                                                
                                                # Get threshold value
                                                if hasattr(module, 'v_threshold'):
                                                    try:
                                                        if module.v_threshold is not None and isinstance(module.v_threshold, torch.Tensor):
                                                            threshold_value = f"{module.v_threshold.mean().item():.6f}"
                                                    except Exception:
                                                        threshold_value = "Error"
                                                
                                                print(f"   🧠 [PROGRESSIVE PRETRAINING] {name}: spikes={total_spikes}/{total_neurons} (rate={spike_rate:.3f})")
                                                print(f"      🧠 membrane potential: {membrane_potential}")
                                                print(f"      🎯 threshold: {threshold_value}")
                                                print(f"      ⚡ using actual spikes from SpikingJelly")
                                    except Exception as e:
                                        # Handle temporal_lif IndexError and other membrane tracking issues
                                        if "temporal_lif" in name and "tuple index out of range" in str(e):
                                            if t == 0 and batch_idx == 0:
                                                print(f"   ⚠️ Membrane tracking warning for {name}: {e}")
                                        elif t == 0 and batch_idx == 0:
                                            print(f"   ⚠️ SNN layer {name} tracking error: {e}")
                        
                        if t == 0 and batch_idx == 0:
                            output_magnitude = torch.abs(outputs).mean().item()
                            spike_rate = (torch.abs(outputs) > model_output_threshold).float().mean().item()
                            print(f"   🔍 Output magnitude: {output_magnitude:.6f}, Spike rate: {spike_rate:.3f}")
                    
                    # Debug: Show available SNN layers in progressive pretraining
                    debug_snn_layers(model, spike_tracker, t, batch_idx)
                    
                    # Comprehensive SNN membrane potential and spike tracking for progressive pretraining
                    if spike_tracker.enabled and hasattr(model, 'snn_block'):
                        for name, module in model.snn_block.named_modules():
                            if hasattr(module, '_last_spike_output'):
                                try:
                                    # Get spike output from AdaptiveLIFNode
                                    membrane_spikes = module._last_spike_output
                                    if isinstance(membrane_spikes, torch.Tensor) and membrane_spikes.numel() > 0:
                                        # Register layer and record spikes
                                        layer_name = f'snn_{name}'
                                        spike_tracker.monitor.register_layer(layer_name)
                                        spike_tracker(layer_name, membrane_spikes)
                                        
                                        # Detailed membrane potential and spike analysis
                                        if t == 0 and batch_idx == 0:  # Log first timestep of first batch
                                            total_neurons = membrane_spikes.numel()
                                            total_spikes = membrane_spikes.sum().item()
                                            spike_rate = total_spikes / total_neurons if total_neurons > 0 else 0.0
                                            
                                            # Get membrane potential if available
                                            membrane_potential = "N/A"
                                            if hasattr(module, 'v'):
                                                try:
                                                    if module.v is not None and isinstance(module.v, torch.Tensor):
                                                        membrane_potential = f"{module.v.mean().item():.6f}"
                                                except Exception:
                                                    membrane_potential = "Error"
                                            
                                            print(f"   🧠 {name}: spikes={total_spikes}/{total_neurons} (rate={spike_rate:.3f})")
                                            print(f"      🧠 membrane potential: {membrane_potential}")
                                            print(f"      ⚡ using actual spikes from SpikingJelly")
                                except Exception as e:
                                    # Handle temporal_lif IndexError and other membrane tracking issues
                                    if "temporal_lif" in name and "tuple index out of range" in str(e):
                                        if t == 0 and batch_idx == 0:
                                            print(f"   ⚠️ Membrane tracking warning for {name}: {e}")
                                    elif t == 0 and batch_idx == 0:
                                        print(f"   ⚠️ SNN layer {name} tracking error: {e}")
                    
                    # Legacy spike counting (for reference)
                    if hasattr(model, 'snn_block'):
                        total_spikes = 0
                        try:
                            for name, module in model.snn_block.named_modules():
                                if not isinstance(name, str):
                                    continue
                                if hasattr(module, 'spike_count') and module.spike_count is not None:
                                    total_spikes += module.spike_count.sum().item()
                        except Exception as e:
                            if t == 0 and batch_idx == 0:
                                print(f"   ⚠️ Legacy spike counting error: {e}")
                        if t == 0 and batch_idx == 0:  # Log once per epoch
                            print(f"   🧠 SNN Spikes this timestep: {total_spikes}")
                    
                    
                    # Reshape outputs
                    outputs = outputs.reshape(batch_size, actual_num_agents, -1)
                    
                    # Compute loss (cross-entropy) with masking for episode done actions
                    flat_outputs = outputs.reshape(-1, outputs.shape[-1])
                    flat_targets = target_actions.reshape(-1).long()
                    
                    # Mask out "episode done" actions (-1)
                    valid_mask = flat_targets >= 0
                    if valid_mask.sum() > 0:  # Only compute loss if there are valid actions
                        timestep_loss = F.cross_entropy(
                            flat_outputs[valid_mask], 
                            flat_targets[valid_mask]
                        )
                    else:
                        timestep_loss = torch.tensor(0.0, device=outputs.device)
                    
                    # Track RL-style goal reaching (excluding collided agents)
                    batch_final_positions = batch_positions[:, -1]  # [batch_size, num_agents, 2]
                    batch_final_distances = torch.norm(batch_final_positions - batch_goals, dim=-1)
                    batch_agents_at_goal = batch_final_distances <= 0.5  # goal_tolerance
                    
                    # Check for collisions during the episode to exclude collided agents from goal count
                    collision_detected = torch.zeros((batch_size, actual_num_agents), dtype=torch.bool)
                    for b in range(batch_size):
                        for t in range(sequence_length):
                            # Check collisions between all agent pairs at timestep t
                            for i in range(actual_num_agents):
                                for j in range(i + 1, actual_num_agents):
                                    pos_i = batch_positions[b, t, i]
                                    pos_j = batch_positions[b, t, j]
                                    if torch.equal(pos_i, pos_j):
                                        collision_detected[b, i] = True
                                        collision_detected[b, j] = True
                    
                    # Only count goal reaching for non-collided agents
                    valid_goal_reaching = batch_agents_at_goal & (~collision_detected)
                    agents_reached_goal += torch.sum(valid_goal_reaching).item()
                    
                    # Count ALL agents in denominator for honest goal rate percentage
                    total_agents += batch_size * actual_num_agents
                    
                    # Removed spike sparsity regularization - was overwhelming the model
                    # Using adaptive thresholds in AdaptiveLIFNode for spike regulation instead
                    
                    # Add enhanced oscillation penalty for progressive pre-training
                    if config.get('use_enhanced_oscillation_detection', True):
                        try:
                            from loss_compute import compute_enhanced_oscillation_penalty
                            oscillation_penalty = compute_enhanced_oscillation_penalty(
                                batch_positions[:, :t+1], batch_id=batch_idx, 
                                oscillation_weight=config.get('oscillation_penalty_weight', 7.0),  # Same default as main training
                                goal_positions=batch_goals
                            )
                            timestep_loss += oscillation_penalty
                        except ImportError:
                            # Fallback to simple oscillation penalty
                            from loss_compute import compute_simple_oscillation_penalty
                            oscillation_penalty = compute_simple_oscillation_penalty(
                                batch_positions[:, :t+1], batch_id=batch_idx,
                                oscillation_weight=config.get('oscillation_penalty_weight', 7.0),  # Same default as main training
                                goal_positions=batch_goals
                            )
                            timestep_loss += oscillation_penalty
                    
                    total_loss += timestep_loss
                
                # Average loss over sequence
                total_loss = total_loss / sequence_length
                
                # Add progressive goal rewards (bigger rewards for more agents reaching goals)
                final_positions = batch_positions[:, -1]
                initial_distances = torch.norm(batch_positions[:, 0] - batch_goals, dim=-1)
                final_distances = torch.norm(final_positions - batch_goals, dim=-1)
                progress = torch.mean(initial_distances - final_distances)
                
                # Count how many agents reached their goals (within tolerance)
                goal_tolerance = 0.5
                agents_at_goal = final_distances <= goal_tolerance  # [batch_size, num_agents]
                goals_reached_per_batch = torch.sum(agents_at_goal, dim=1).float()  # [batch_size]
                
                # Progressive rewards: 1 goal = small, 2 goals = big bonus!
                goal_rewards = torch.zeros_like(goals_reached_per_batch)
                goal_rewards += (goals_reached_per_batch >= 1).float() * 0.5  # 1 goal reached
                goal_rewards += (goals_reached_per_batch >= 2).float() * 1.5  # Both goals reached - BIG BONUS!
                
                average_goal_reward = torch.mean(goal_rewards)
                goal_progress_loss = -progress * 0.1 - average_goal_reward * 0.2  # Stronger reward signal
                
                total_loss += goal_progress_loss
                
                # Backward pass
                total_loss.backward()
                
                # Adaptive gradient clipping with proper norm measurement
                max_norm = config.get('gradient_clip_max_norm', 2.0)
                
                # Measure gradient norm WITHOUT modifying gradients
                total_grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.detach().data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                unclipped_grad_norm = total_grad_norm ** 0.5
                
                # Only apply clipping if gradients are significantly large
                explosion_threshold = max_norm * 3.0  # Only clip if > 6.0
                if unclipped_grad_norm > explosion_threshold:
                    # Apply clipping for true gradient explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    # Recompute gradient norm after clipping
                    clipped_norm = sum(
                        p.grad.detach().data.norm(2).item() ** 2
                        for p in model.parameters() if p.grad is not None
                    ) ** 0.5
                    grad_norm = clipped_norm
                    if batch_idx % 10 == 0:
                        print(f"   🚨 PROGRESSIVE EXPLOSION! Gradient clipped: {unclipped_grad_norm:.3f} → {grad_norm:.3f}")
                else:
                    # No clipping - let gradients flow naturally
                    grad_norm = unclipped_grad_norm
                    if batch_idx % 20 == 0:
                        print(f"   📊 Progressive gradient norm: {grad_norm:.3f} (no clipping needed)")
                
                # Log detailed gradient info
                if batch_idx % 20 == 0:
                    # Check specific layer gradients that might be getting suppressed
                    for name, param in model.named_parameters():
                        if param.grad is not None and 'global_features' in name:
                            layer_grad_norm = param.grad.norm().item()
                            print(f"   🧠 {name} grad norm: {layer_grad_norm:.6f}")
                
                optimizer.step()
                
                # Calculate RL-style goal reaching rate instead of accuracy
                goal_reaching_rate = agents_reached_goal / total_agents if total_agents > 0 else 0.0
                goal_progress_metric = progress.item()
                
                # Calculate collision rate
                collision_count = 0
                for b in range(batch_size):
                    for t in range(sequence_length):
                        positions_t = batch_positions[b, t]  # [num_agents, 2]
                        for i in range(actual_num_agents):
                            for j in range(i + 1, actual_num_agents):
                                if torch.equal(positions_t[i], positions_t[j]):
                                    collision_count += 1
                collision_rate = collision_count / (batch_size * sequence_length)
                
                # Store metrics
                epoch_losses.append(total_loss.item())
                epoch_goal_rates.append(goal_reaching_rate)
                epoch_goal_progress.append(goal_progress_metric)
                epoch_collisions.append(collision_rate)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Goal Rate': f'{goal_reaching_rate:.3f}',
                    'Progress': f'{goal_progress_metric:.3f}',
                    'Collisions': f'{collision_rate:.3f}',
                    'Grid': f'{grid_size}x{grid_size}'
                })
            
            # Store epoch metrics
            epoch_loss = np.mean(epoch_losses)
            epoch_goal_rate = np.mean(epoch_goal_rates)
            epoch_prog = np.mean(epoch_goal_progress)
            epoch_coll = np.mean(epoch_collisions)
            
            # Calculate network health score
            if spike_tracker.enabled:
                network_health = spike_tracker.summary(current_epoch)
                history['network_health'].append(network_health)
            else:
                network_health = 1.0  # Default if not tracking
                history['network_health'].append(network_health)
            
            history['loss'].append(epoch_loss)
            history['agents_reached_goal'].append(epoch_goal_rate)
            history['goal_progress'].append(epoch_prog)
            history['collision_rate'].append(epoch_coll)
            history['stage_info'].append({
                'stage': stage_idx + 1,
                'grid_size': grid_size,
                'epoch_in_stage': epoch + 1,
                'global_epoch': current_epoch
            })
            
            print(f"   📈 Stage {stage_idx + 1} Epoch {epoch + 1} Summary:")
            print(f"      Loss: {epoch_loss:.4f}")
            print(f"      Goal Reaching Rate: {epoch_goal_rate:.4f}")
            print(f"      Goal Progress: {epoch_prog:.4f}")
            print(f"      Collision Rate: {epoch_coll:.4f}")
            print(f"      🧠 Network Health: {network_health:.3f}/1.0 {'✅' if network_health > 0.5 else '⚠️' if network_health > 0.2 else '❌'}")
    
    print(f"\n✅ Progressive Pre-training completed!")
    print(f"   📊 Final Loss: {history['loss'][-1]:.4f}")
    print(f"   🎯 Final Goal Reaching Rate: {history['agents_reached_goal'][-1]:.4f}")
    print(f"   📈 Final Goal Progress: {history['goal_progress'][-1]:.4f}")
    print(f"   💥 Final Collision Rate: {history['collision_rate'][-1]:.4f}")
    print(f"   🧠 Final Network Health: {history['network_health'][-1]:.3f}/1.0 {'✅' if history['network_health'][-1] > 0.5 else '⚠️' if history['network_health'][-1] > 0.2 else '❌'}")
    
    return history

def visualize_pretrain_scenario(positions: torch.Tensor, goals: torch.Tensor, trajectory: torch.Tensor = None, 
                               title: str = "Pre-training Scenario", save_path: str = None):
    """
    Visualize a simple pre-training scenario.
    
    Args:
        positions: [num_agents, 2] initial positions
        goals: [num_agents, 2] goal positions  
        trajectory: [timesteps, num_agents, 2] optional trajectory to animate
        title: plot title
        save_path: path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Draw grid
    for i in range(4):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        ax.axvline(i - 0.5, color='gray', linewidth=0.5)
    
    # Plot agent start positions
    colors = ['red', 'blue']
    for i, (pos, goal) in enumerate(zip(positions, goals)):
        ax.scatter(pos[0], pos[1], c=colors[i], s=200, marker='o', 
                  label=f'Agent {i+1} Start', edgecolors='black', linewidth=2)
        ax.scatter(goal[0], goal[1], c=colors[i], s=200, marker='*', 
                  label=f'Agent {i+1} Goal', edgecolors='black', linewidth=2)
    
    # If trajectory provided, draw path
    if trajectory is not None:
        for i in range(positions.shape[0]):
            path_x = trajectory[:, i, 0].numpy()
            path_y = trajectory[:, i, 1].numpy()
            ax.plot(path_x, path_y, color=colors[i], linewidth=2, alpha=0.7, 
                   linestyle='--', label=f'Agent {i+1} Path')
    
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved visualization to {save_path}")
    
    plt.show()

def test_pretrain_generation():
    """Test the pre-training data generation."""
    print("🧪 Testing pre-training data generation...")
    
    # Generate a simple scenario
    positions, goals, obstacles = generate_simple_scenario()
    print(f"   Agent positions: {positions}")
    print(f"   Goal positions: {goals}")
    print(f"   Obstacles: {obstacles}")
    
    # Generate heuristic trajectory for RL learning
    trajectory_pos, trajectory_actions = generate_expert_trajectory(positions, goals, max_steps=8)
    print(f"   Trajectory shape: {trajectory_pos.shape}")
    print(f"   Actions shape: {trajectory_actions.shape}")
    
    # Generate batch
    batch_fovs, batch_actions, batch_positions, batch_goals, batch_obstacles = generate_pretraining_batch(
        batch_size=4, sequence_length=8
    )
    
    # Visualize one scenario
    visualize_pretrain_scenario(positions, goals, trajectory_pos, 
                               title="Test Pre-training Scenario")
    
    print("✅ Pre-training data generation test completed!")

if __name__ == "__main__":
    test_pretrain_generation()
