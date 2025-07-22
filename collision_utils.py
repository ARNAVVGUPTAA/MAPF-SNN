"""
Collision detection utilities for Multi-Agent Path Finding (MAPF).
Provides functions to detect and penalize different types of collisions.
"""

import torch
import numpy as np
import yaml
import os
from typing import Dict, List, Tuple, Optional


def generate_safe_dummy_positions(num_agents: int = 5, board_size: int = 28, 
                                 position_type: str = "start") -> torch.Tensor:
    """
    Generate safe dummy positions for agents with maximum spread to minimize collisions.
    No obstacles are generated, just spread-out positions.
    
    Args:
        num_agents: Number of agents to generate positions for
        board_size: Size of the board
        position_type: Type of positions ("start" or "goal")
        
    Returns:
        safe_positions: Safe dummy positions [num_agents, 2]
    """
    dummy_positions = []
    
    if position_type == "start":
        # For start positions, spread agents across the map
        for agent_id in range(num_agents):
            x = (agent_id * 5) % board_size  # Spread horizontally every 5 units
            y = (agent_id * 3) % board_size  # Spread vertically every 3 units
            dummy_positions.append([float(x), float(y)])
    else:  # goal positions
        # For goal positions, place them on opposite side of the map from starts
        for agent_id in range(num_agents):
            x = (board_size - 5 - agent_id * 2) % board_size  # Goals on opposite side
            y = (board_size - 3 - agent_id * 2) % board_size  # Goals spread diagonally
            dummy_positions.append([float(x), float(y)])
    
    return torch.tensor(dummy_positions, dtype=torch.float32)


def detect_collisions(positions: torch.Tensor, prev_positions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """
    Detect various types of collisions between agents.
    
    Args:
        positions: Current agent positions [batch, num_agents, 2] (x, y coordinates)
        prev_positions: Previous agent positions [batch, num_agents, 2] for edge collision detection
        
    Returns:
        Dict containing collision indicators:
        - 'vertex_collisions': [batch, num_agents] - agents in same position
        - 'edge_collisions': [batch, num_agents] - agents swapping positions
        - 'proximity_collisions': [batch, num_agents] - agents too close
    """
    batch_size, num_agents, _ = positions.shape
    device = positions.device
    
    # Initialize collision indicators
    vertex_collisions = torch.zeros(batch_size, num_agents, device=device)
    edge_collisions = torch.zeros(batch_size, num_agents, device=device)
    proximity_collisions = torch.zeros(batch_size, num_agents, device=device)
    
    # Detect vertex collisions (same position)
    for b in range(batch_size):
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # Check if agents i and j are at the same position
                same_position = torch.all(positions[b, i] == positions[b, j])
                if same_position:
                    vertex_collisions[b, i] = 1.0
                    vertex_collisions[b, j] = 1.0
                
                # Check proximity (within collision radius)
                distance = torch.norm(positions[b, i] - positions[b, j])
                if distance <= 1.0:  # Adjacent cells or same cell
                    proximity_collisions[b, i] = 1.0
                    proximity_collisions[b, j] = 1.0
    
    # Detect edge collisions (swapping positions)
    if prev_positions is not None:
        for b in range(batch_size):
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    # Check if agents i and j swapped positions
                    i_to_j = torch.all(positions[b, i] == prev_positions[b, j])
                    j_to_i = torch.all(positions[b, j] == prev_positions[b, i])
                    if i_to_j and j_to_i:
                        edge_collisions[b, i] = 1.0
                        edge_collisions[b, j] = 1.0
    
    return {
        'vertex_collisions': vertex_collisions,
        'edge_collisions': edge_collisions,
        'proximity_collisions': proximity_collisions
    }


def compute_per_agent_collision_losses(logits: torch.Tensor, 
                                     immediate_penalty: torch.Tensor,
                                     collision_config: Dict) -> torch.Tensor:
    """
    Compute per-agent collision losses that can be applied to individual agent predictions.
    
    Args:
        logits: Action logits [batch * num_agents, num_actions]
        immediate_penalty: Collision penalties per agent [batch, num_agents]
        collision_config: Collision configuration dictionary
        
    Returns:
        per_agent_losses: Collision losses per agent [batch * num_agents]
    """
    batch_size, num_agents = immediate_penalty.shape
    device = immediate_penalty.device
    
    # Reshape penalty to match logits shape [batch * num_agents]
    penalty_flat = immediate_penalty.reshape(-1)  # [batch * num_agents]
    
    # Compute loss based on collision type
    collision_loss_type = collision_config.get('collision_loss_type', 'l2')
    
    if collision_loss_type == 'l1':
        per_agent_losses = penalty_flat
    elif collision_loss_type == 'l2':
        per_agent_losses = penalty_flat ** 2
    elif collision_loss_type == 'exponential':
        per_agent_losses = torch.exp(penalty_flat) - 1
    else:
        per_agent_losses = penalty_flat
    
    return per_agent_losses


def compute_collision_loss(logits: torch.Tensor, 
                          positions: torch.Tensor,
                          prev_positions: Optional[torch.Tensor] = None,
                          collision_config: Dict = None) -> Tuple[torch.Tensor, Dict]:
    """
    Compute collision loss to penalize actions that lead to collisions.
    Uses continuous inverse Manhattan distance for smooth gradients.
    
    Args:
        logits: Action logits [batch * num_agents, num_actions]
        positions: Current agent positions [batch, num_agents, 2]
        prev_positions: Previous agent positions [batch, num_agents, 2]
        collision_config: Configuration parameters for collision loss
        
    Returns:
        Tuple of (collision_loss, collision_info)
    """
    if collision_config is None:
        collision_config = {
            'vertex_collision_weight': 5.0,  # Increased from 1.0 for balance
            'edge_collision_weight': 3.0,    # Increased from 0.5 for balance
            'collision_loss_type': 'l2',
            'use_future_collision_penalty': False,
            'future_collision_steps': 2,
            'future_collision_weight': 0.8,
            'future_step_decay': 0.7,
            'separate_collision_types': True,
            'per_agent_loss': True,
            'collision_threshold': 2.0,  # Distance threshold for collision penalty
            'use_continuous_collision': True  # Use continuous inverse distance
        }
    
    batch_size, num_agents = positions.shape[:2]
    device = positions.device
    
    # Use continuous collision loss for better gradients
    if collision_config.get('use_continuous_collision', True):
        collision_penalty = compute_continuous_collision_penalty(
            positions, collision_config
        )
    else:
        # Fallback to binary collision detection
        collisions = detect_collisions(positions, prev_positions)
        vertex_penalty = collisions['vertex_collisions'] * collision_config['vertex_collision_weight']
        edge_penalty = collisions['edge_collisions'] * collision_config['edge_collision_weight']
        collision_penalty = vertex_penalty + edge_penalty
    
    # Check if we should compute per-agent losses
    use_per_agent_loss = collision_config.get('per_agent_loss', True)
    
    if use_per_agent_loss:
        # Compute per-agent collision losses that can be applied to individual agent predictions
        per_agent_collision_losses = compute_per_agent_collision_losses(
            logits, collision_penalty, collision_config
        )
        
        # Also compute aggregate statistics for logging
        if collision_config['collision_loss_type'] == 'l1':
            aggregate_collision_loss = torch.mean(collision_penalty)
        elif collision_config['collision_loss_type'] == 'l2':
            aggregate_collision_loss = torch.mean(collision_penalty ** 2)
        elif collision_config['collision_loss_type'] == 'exponential':
            aggregate_collision_loss = torch.mean(torch.exp(collision_penalty) - 1)
        else:
            aggregate_collision_loss = torch.mean(collision_penalty)
        
        total_collision_loss = per_agent_collision_losses
        real_collision_loss = aggregate_collision_loss
        
    else:
        # Legacy behavior: compute single aggregate loss
        # Initialize collision losses
        real_collision_loss = torch.tensor(0.0, device=device)
        
        # Compute real collision loss
        if collision_config['collision_loss_type'] == 'l1':
            real_collision_loss = torch.mean(collision_penalty)
        elif collision_config['collision_loss_type'] == 'l2':
            real_collision_loss = torch.mean(collision_penalty ** 2)
        elif collision_config['collision_loss_type'] == 'exponential':
            real_collision_loss = torch.mean(torch.exp(collision_penalty) - 1)
        else:
            real_collision_loss = torch.mean(collision_penalty)
        
        total_collision_loss = real_collision_loss
    
    # Future collision penalty is not implemented for per-agent losses yet
    # (can be added later if needed)
    future_collision_loss = torch.tensor(0.0, device=device)
    future_penalty_value = 0.0
    
    if not use_per_agent_loss and collision_config.get('use_future_collision_penalty', False):
        future_steps = collision_config.get('future_collision_steps', 2)
        step_decay = collision_config.get('future_step_decay', 0.7)
        
        # Only penalize future collisions if there are no current collisions
        # This prevents double penalization
        if collision_config.get('separate_collision_types', True):
            # Create a mask for agents that are NOT currently colliding
            no_current_collision_mask = (collision_penalty == 0).float()
            
            future_penalty_tensor = predict_future_collisions(
                positions, logits, 
                steps=future_steps, 
                board_size=28,  # TODO: make this configurable
                step_decay=step_decay,
                collision_mask=no_current_collision_mask  # Only predict for non-colliding agents
            )
        else:
            # Old behavior: predict for all agents regardless of current collision
            future_penalty_tensor = predict_future_collisions(
                positions, logits, 
                steps=future_steps, 
                board_size=28,
                step_decay=step_decay
            )
        
        future_penalty_value = float(future_penalty_tensor.item())
        future_collision_loss = collision_config.get('future_collision_weight', 0.8) * future_penalty_tensor
        
        # Total collision loss - combine both real and future losses
        if collision_config.get('separate_collision_types', True):
            # Blend real and future collision losses with configurable factor
            alpha = collision_config.get('future_blend_factor', 0.5)  # Default blend factor
            total_collision_loss = real_collision_loss + alpha * future_collision_loss
        else:
            # Add both losses with equal weight (legacy behavior)
            total_collision_loss = real_collision_loss + future_collision_loss
    
    # Collect collision information for logging
    collision_info = {
        'total_real_collisions': torch.sum(collision_penalty > 0).item(),
        'vertex_collisions': 0,  # Not directly available with continuous collision
        'edge_collisions': 0,    # Not directly available with continuous collision
        'real_collision_rate': torch.mean((collision_penalty > 0).float()).item(),
        'avg_real_collision_penalty': torch.mean(collision_penalty).item(),
        'future_collision_penalty': future_penalty_value,
        'real_collision_loss': float(real_collision_loss.item()) if isinstance(real_collision_loss, torch.Tensor) else real_collision_loss,
        'future_collision_loss': float(future_collision_loss.item()),
        'using_real_collision_loss': True,  # Always using both losses now
        'future_blend_factor': collision_config.get('future_blend_factor', 0.5) if collision_config.get('separate_collision_types', True) else 1.0,
        'per_agent_loss': use_per_agent_loss
    }
    
    return total_collision_loss, collision_info


def extract_positions_from_actions(current_positions: torch.Tensor, 
                                  actions: torch.Tensor,
                                  board_size: int = 28) -> torch.Tensor:
    """
    Predict next positions based on current positions and actions.
    
    Args:
        current_positions: Current agent positions [batch, num_agents, 2]
        actions: Predicted actions [batch, num_agents] 
        board_size: Size of the board for boundary checking
        
    Returns:
        next_positions: Predicted next positions [batch, num_agents, 2]
    """
    # Action mapping: 0=stay, 1=right, 2=up, 3=left, 4=down
    action_deltas = torch.tensor([
        [0, 0],   # 0: Stay
        [1, 0],   # 1: Right
        [0, 1],   # 2: Up
        [-1, 0],  # 3: Left
        [0, -1]   # 4: Down
    ], device=current_positions.device, dtype=current_positions.dtype)
    
    batch_size, num_agents = actions.shape
    
    # Get action deltas for each agent
    deltas = action_deltas[actions]  # [batch, num_agents, 2]
    
    # Compute next positions
    next_positions = current_positions + deltas
    
    # Apply boundary constraints
    next_positions = torch.clamp(next_positions, 0, board_size - 1)
    
    return next_positions


def predict_future_collisions(current_positions: torch.Tensor,
                             action_logits: torch.Tensor,
                             steps: int = 2,
                             board_size: int = 28,
                             step_decay: float = 0.7,
                             collision_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Predict potential collisions in future steps based on current action tendencies.
    
    Args:
        current_positions: Current agent positions [batch, num_agents, 2]
        action_logits: Action logits [batch * num_agents, num_actions]
        steps: Number of future steps to predict
        board_size: Size of the board
        step_decay: Decay factor for future steps (closer steps have higher weight)
        collision_mask: Mask indicating which agents to predict for [batch, num_agents] (1.0 = predict, 0.0 = ignore)
        
    Returns:
        future_collision_penalty: Penalty for predicted future collisions
    """
    batch_size, num_agents = current_positions.shape[:2]
    device = current_positions.device
    
    # Convert logits to action probabilities
    action_probs = torch.softmax(action_logits.reshape(batch_size, num_agents, -1), dim=-1)
    
    total_future_penalty = torch.tensor(0.0, device=device)
    positions = current_positions.clone()
    
    for step in range(steps):
        # Use expected position computation instead of sampling for differentiability
        # Action mapping: 0=stay, 1=right, 2=up, 3=left, 4=down
        action_deltas = torch.tensor([
            [0, 0],   # 0: Stay
            [1, 0],   # 1: Right
            [0, 1],   # 2: Up
            [-1, 0],  # 3: Left
            [0, -1]   # 4: Down
        ], device=device, dtype=torch.float32)
        
        # Compute expected deltas: action_probs @ action_deltas [batch, agents, 2]
        expected_deltas = torch.einsum('ban,nd->bad', action_probs, action_deltas)  # [batch, agents, 2]
        
        # Predict next positions using expected movement
        next_positions = positions + expected_deltas
        
        # Apply boundary constraints
        next_positions = torch.clamp(next_positions, 0, board_size - 1)
        
        # Detect collisions at this future step
        collisions = detect_collisions(next_positions, positions)
        
        # Calculate step weight (exponential decay for future steps)
        step_weight = step_decay ** step
        
        # Apply collision mask if provided (only count collisions for non-currently-colliding agents)
        vertex_penalty = collisions['vertex_collisions']
        edge_penalty = collisions['edge_collisions']
        
        if collision_mask is not None:
            vertex_penalty = vertex_penalty * collision_mask
            edge_penalty = edge_penalty * collision_mask
        
        # Add penalty (weighted by distance into future)
        step_penalty = (torch.sum(vertex_penalty) + 0.5 * torch.sum(edge_penalty)) * step_weight
        total_future_penalty += step_penalty
        
        # Update positions for next iteration
        positions = next_positions
    
    # Normalize by number of agents and steps
    normalization_factor = batch_size * num_agents * steps
    if collision_mask is not None:
        # If using mask, normalize by the number of agents we're actually predicting for
        active_agents = torch.sum(collision_mask).item()
        if active_agents > 0:
            normalization_factor = active_agents * steps
    
    return total_future_penalty / normalization_factor if normalization_factor > 0 else total_future_penalty


def extract_positions_from_fov(fov: torch.Tensor, num_agents: int, env_positions: torch.Tensor = None) -> torch.Tensor:
    """
    Extract agent positions from field of view observations or use provided positions.
    
    Args:
        fov: Field of view tensor [batch * num_agents, channels, height, width]
        num_agents: Number of agents
        env_positions: Explicit positions from environment [batch, num_agents, 2] (preferred)
        
    Returns:
        positions: Agent positions [batch, num_agents, 2]
    """
    batch_size = fov.size(0) // num_agents
    device = fov.device
    
    if env_positions is not None:
        # Use explicit positions from environment (preferred approach)
        return env_positions.to(device)
    
    # Fallback: Try to extract from center position in FOV
    # The FOV is centered on the agent, so agent position is at center
    fov_reshaped = fov.reshape(batch_size, num_agents, fov.size(1), fov.size(2), fov.size(3))
    fov_height, fov_width = fov.size(2), fov.size(3)
    
    # For now, use a heuristic approach since exact position extraction from FOV
    # would require knowing the global board position encoding
    # Return random positions as fallback (this should be avoided in practice)
    positions = torch.randint(0, 28, (batch_size, num_agents, 2), device=device, dtype=torch.float32)
    
    return positions


def reconstruct_positions_from_trajectories(initial_positions: torch.Tensor,
                                                trajectories: torch.Tensor,
                                                current_time: int,
                                                board_size: int = 28) -> torch.Tensor:
    """
    Reconstruct agent positions at current time step by applying action sequence from initial positions.
    
    Args:
        initial_positions: Starting positions [batch, num_agents, 2]
        trajectories: Action sequence [batch, time_steps, num_agents]
        current_time: Current time step to reconstruct positions for
        board_size: Size of the board for boundary checking
        
    Returns:
        current_positions: Reconstructed positions at current time [batch, num_agents, 2]
    """
    # Action mapping: 0=stay, 1=right, 2=up, 3=left, 4=down
    action_deltas = torch.tensor([
        [0, 0],   # 0: Stay
        [1, 0],   # 1: Right
        [0, 1],   # 2: Up
        [-1, 0],  # 3: Left
        [0, -1]   # 4: Down
    ], device=initial_positions.device, dtype=initial_positions.dtype)
    
    batch_size, num_agents = initial_positions.shape[:2]
    positions = initial_positions.clone()
    
    # Apply actions sequentially up to current time
    for t in range(current_time):
        if t < trajectories.shape[1]:
            actions = trajectories[:, t, :].long()  # [batch, num_agents]
            
            # Get deltas for each agent's action
            deltas = action_deltas[actions]  # [batch, num_agents, 2]
            
            # Update positions
            positions = positions + deltas
            
            # Apply boundary constraints
            positions = torch.clamp(positions, 0, board_size - 1)
    
    return positions


def load_initial_positions_from_dataset(dataset_root: str, case_indices: List[int], 
                                        mode: str = "train") -> torch.Tensor:
    """
    Load initial agent positions from input.yaml files in the dataset.
    
    Args:
        dataset_root: Root directory of the dataset (e.g., 'dataset/5_8_28')
        case_indices: List of case indices to load positions for
        mode: Dataset mode ('train' or 'val')
        
    Returns:
        initial_positions: Initial positions tensor [num_cases, num_agents, 2]
    """
    dataset_path = os.path.join(dataset_root, mode)
    initial_positions_list = []
    
    for case_idx in case_indices:
        case_name = f"case_{case_idx}"
        input_yaml_path = os.path.join(dataset_path, case_name, "input.yaml")
        
        if not os.path.exists(input_yaml_path):
            print(f"WARNING: [FALLBACK TRIGGERED] input.yaml not found for {case_name}, using SAFE dummy positions")
            print(f"         Generating obstacle-free dummy map. Dataset path: {input_yaml_path}")
            # Use safe dummy positions - spread agents with no obstacles
            dummy_positions_tensor = generate_safe_dummy_positions(num_agents=5, position_type="start")
            print(f"         Generated safe spread positions: {dummy_positions_tensor}")
            initial_positions_list.append(dummy_positions_tensor)
            continue
            
        try:
            with open(input_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                
            agents = data.get('agents', [])
            positions = []
            
            for agent in agents:
                start_pos = agent.get('start', [0, 0])
                positions.append([float(start_pos[0]), float(start_pos[1])])
                
            positions_tensor = torch.tensor(positions, dtype=torch.float32)
            initial_positions_list.append(positions_tensor)
            
        except Exception as e:
            print(f"ERROR: [FALLBACK TRIGGERED] Error loading {input_yaml_path}: {e}")
            print(f"       Using SAFE dummy positions - generating obstacle-free map!")
            # Use safe dummy positions instead of random ones
            dummy_positions_tensor = generate_safe_dummy_positions(num_agents=5, position_type="start")
            print(f"       Generated safe spread positions: {dummy_positions_tensor}")
            initial_positions_list.append(dummy_positions_tensor)
    
    # Stack all initial positions
    initial_positions = torch.stack(initial_positions_list, dim=0)
    return initial_positions


def load_goal_positions_from_dataset(dataset_root: str, case_indices: List[int], 
                                    mode: str = "train") -> torch.Tensor:
    """
    Load goal agent positions from input.yaml files in the dataset.
    
    Args:
        dataset_root: Root directory of the dataset (e.g., 'dataset/5_8_28')
        case_indices: List of case indices to load goal positions for
        mode: Dataset mode ('train' or 'val')
        
    Returns:
        goal_positions: Goal positions tensor [num_cases, num_agents, 2]
    """
    dataset_path = os.path.join(dataset_root, mode)
    goal_positions_list = []
    
    for case_idx in case_indices:
        case_name = f"case_{case_idx}"
        input_yaml_path = os.path.join(dataset_path, case_name, "input.yaml")
        
        if not os.path.exists(input_yaml_path):
            print(f"WARNING: [FALLBACK TRIGGERED] input.yaml not found for {case_name}, using SAFE dummy goal positions")
            print(f"         Generating obstacle-free dummy map. Dataset path: {input_yaml_path}")
            # Use safe dummy goal positions - spread goals with no obstacles
            dummy_positions_tensor = generate_safe_dummy_positions(num_agents=5, position_type="goal")
            print(f"         Generated safe spread goal positions: {dummy_positions_tensor}")
            goal_positions_list.append(dummy_positions_tensor)
            continue
            
        try:
            with open(input_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                
            agents = data.get('agents', [])
            positions = []
            
            for agent in agents:
                goal_pos = agent.get('goal', [0, 0])
                positions.append([float(goal_pos[0]), float(goal_pos[1])])
                
            positions_tensor = torch.tensor(positions, dtype=torch.float32)
            goal_positions_list.append(positions_tensor)
            
        except Exception as e:
            print(f"ERROR: [FALLBACK TRIGGERED] Error loading {input_yaml_path}: {e}")
            print(f"       Using SAFE dummy goal positions - generating obstacle-free map!")
            # Use safe dummy goal positions instead of random ones
            dummy_positions_tensor = generate_safe_dummy_positions(num_agents=5, position_type="goal")
            print(f"       Generated safe spread goal positions: {dummy_positions_tensor}")
            goal_positions_list.append(dummy_positions_tensor)
    
    # Stack all goal positions
    goal_positions = torch.stack(goal_positions_list, dim=0)
    return goal_positions


def compute_goal_proximity_reward(current_positions: torch.Tensor, goal_positions: torch.Tensor,
                                 reward_type: str = 'inverse', max_distance: float = 10.0,
                                 success_threshold: float = 1.0, track_progress: bool = True,
                                 previous_positions: torch.Tensor = None, config=None) -> torch.Tensor:
    """
    Compute goal proximity reward based on distance to goal positions with progress tracking.
    Only rewards agents that are moving toward their goals, not away from them.
    
    Args:
        current_positions: Current agent positions [batch_size, num_agents, 2]
        goal_positions: Goal positions [batch_size, num_agents, 2]
        reward_type: Type of reward computation ('inverse', 'exponential', 'linear')
        max_distance: Maximum distance for reward normalization
        success_threshold: Distance threshold to consider agent has reached goal (used when track_progress=True)
        track_progress: If True, only reward agents that haven't reached their goals yet
        previous_positions: Previous agent positions [batch_size, num_agents, 2] for movement direction tracking
        
    Returns:
        proximity_reward: Proximity reward scalar for the batch
    """
    # Compute Euclidean distances to goals
    distances = torch.norm(current_positions - goal_positions, dim=2)  # [batch_size, num_agents]
    
    # Create base mask for eligible agents
    eligible_agents_mask = torch.ones_like(distances, dtype=torch.bool)
    
    # Apply progress tracking if enabled
    if track_progress:
        # Identify agents that have reached their goals
        agents_at_goal_mask = distances <= success_threshold  # [batch_size, num_agents]
        # Create mask for agents that still need to reach goals
        eligible_agents_mask = ~agents_at_goal_mask  # [batch_size, num_agents]
    
    # Apply movement direction filter if previous positions are provided
    if previous_positions is not None:
        # Compute previous distances to goals
        prev_distances = torch.norm(previous_positions - goal_positions, dim=2)  # [batch_size, num_agents]
        
        # Only reward agents that are moving closer to their goals (distance decreased)
        moving_toward_goal_mask = distances < prev_distances  # [batch_size, num_agents]
        
        # Combine with existing eligibility mask
        eligible_agents_mask = eligible_agents_mask & moving_toward_goal_mask
    
    # Get distances for eligible agents only
    if torch.any(eligible_agents_mask):
        active_distances = distances[eligible_agents_mask]
    else:
        # No eligible agents (all at goals or moving away), return zero reward as scalar tensor
        return torch.tensor(0.0, device=current_positions.device, dtype=torch.float32)
    
    if reward_type == 'inverse':
        # Inverse distance reward: closer = higher reward
        # Add small epsilon to avoid division by zero and clamp to reasonable range
        epsilon = 1e-6
        rewards = 1.0 / (active_distances + epsilon)
        # Apply configurable multiplier for stronger goal-seeking behavior
        proximity_scale = config.get('proximity_reward_scale', 25.0) if config else 25.0
        rewards = rewards * proximity_scale
        # Clamp rewards to prevent extreme values (scale cap based on multiplier)
        rewards = torch.clamp(rewards, max=proximity_scale * 10.0)  # Cap at scale * 10
    elif reward_type == 'exponential':
        # Exponential decay reward: exp(-distance/scale)
        scale = max_distance / 3.0  # Scale factor for exponential decay
        rewards = torch.exp(-active_distances / scale)
        # Apply configurable multiplier for stronger goal-seeking behavior
        proximity_scale = config.get('proximity_reward_scale', 25.0) if config else 25.0
        rewards = rewards * proximity_scale
    elif reward_type == 'linear':
        # Linear reward: (max_distance - distance) / max_distance
        rewards = torch.clamp((max_distance - active_distances) / max_distance, min=0.0)
        # Apply configurable multiplier for stronger goal-seeking behavior
        proximity_scale = config.get('proximity_reward_scale', 25.0) if config else 25.0
        rewards = rewards * proximity_scale
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")
    
    # Average reward across selected agents
    proximity_reward = torch.mean(rewards)
    return proximity_reward


def compute_goal_success_bonus(current_positions: torch.Tensor, goal_positions: torch.Tensor,
                             success_threshold: float = 1.0) -> torch.Tensor:
    """
    Compute goal success bonus for agents that have reached their goals.
    
    Args:
        current_positions: Current agent positions [batch_size, num_agents, 2] or [num_agents, 2]
        goal_positions: Goal positions [batch_size, num_agents, 2] or [num_agents, 2]
        success_threshold: Distance threshold to consider agent has reached goal
        
    Returns:
        success_bonus: Success bonus scalar for the batch
    """
    # Handle both 2D and 3D tensor inputs
    if current_positions.dim() == 2:
        # Add batch dimension if missing [num_agents, 2] -> [1, num_agents, 2]
        current_positions = current_positions.unsqueeze(0)
        goal_positions = goal_positions.unsqueeze(0)
    
    # Compute Euclidean distances to goals
    distances = torch.norm(current_positions - goal_positions, dim=2)  # [batch_size, num_agents]
    
    # Count agents that have reached their goals (distance <= threshold)
    success_mask = distances <= success_threshold  # [batch_size, num_agents]
    
    # Calculate success bonus: number of successful agents normalized by total agents
    num_successful = torch.sum(success_mask.float())
    total_agents = torch.numel(success_mask)
    
    success_bonus = num_successful / total_agents if total_agents > 0 else torch.tensor(0.0, device=current_positions.device)
    
    return success_bonus


def compute_goal_success_bonus_with_state_tracking(current_positions: torch.Tensor, 
                                                  goal_positions: torch.Tensor,
                                                  success_threshold: float = 1.0,
                                                  previous_agents_at_goal: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute goal success bonus with state tracking to reward only newly successful agents.
    
    Args:
        current_positions: Current agent positions [batch_size, num_agents, 2] or [num_agents, 2]
        goal_positions: Goal positions [batch_size, num_agents, 2] or [num_agents, 2] 
        success_threshold: Distance threshold to consider agent has reached goal
        previous_agents_at_goal: Previous timestep's goal achievement mask [batch_size, num_agents]
        
    Returns:
        Tuple of (success_bonus, agents_at_goal_mask)
        - success_bonus: Success bonus for newly successful agents only
        - agents_at_goal_mask: Current goal achievement mask [batch_size, num_agents]
    """
    # Handle both 2D and 3D tensor inputs
    if current_positions.dim() == 2:
        current_positions = current_positions.unsqueeze(0)
        goal_positions = goal_positions.unsqueeze(0)
        if previous_agents_at_goal is not None and previous_agents_at_goal.dim() == 1:
            previous_agents_at_goal = previous_agents_at_goal.unsqueeze(0)
    
    # Compute Euclidean distances to goals
    distances = torch.norm(current_positions - goal_positions, dim=2)  # [batch_size, num_agents]
    
    # Current agents that have reached their goals
    current_agents_at_goal = distances <= success_threshold  # [batch_size, num_agents]
    
    if previous_agents_at_goal is not None:
        # Only reward newly successful agents (reached goal this timestep)
        newly_successful = current_agents_at_goal & (~previous_agents_at_goal)
        num_newly_successful = torch.sum(newly_successful.float())
        total_agents = torch.numel(current_agents_at_goal)
        
        success_bonus = num_newly_successful / total_agents if total_agents > 0 else torch.tensor(0.0, device=current_positions.device)
    else:
        # First timestep or no previous state: reward all currently successful agents
        num_successful = torch.sum(current_agents_at_goal.float())
        total_agents = torch.numel(current_agents_at_goal)
        
        success_bonus = num_successful / total_agents if total_agents > 0 else torch.tensor(0.0, device=current_positions.device)
    
    return success_bonus, current_agents_at_goal


def load_obstacles_from_dataset(dataset_root: str, case_indices: List[int], 
                               mode: str = "train") -> Optional[torch.Tensor]:
    """
    Load obstacle positions from input.yaml files in the dataset.
    
    Args:
        dataset_root: Root directory of the dataset (e.g., 'dataset/5_8_28')
        case_indices: List of case indices to load obstacles for
        mode: Dataset mode ('train' or 'val')
        
    Returns:
        obstacles: Obstacle positions tensor [num_cases, num_obstacles, 2] or None if no obstacles
    """
    dataset_path = os.path.join(dataset_root, mode)
    obstacles_list = []
    
    for case_idx in case_indices:
        case_name = f"case_{case_idx}"
        input_yaml_path = os.path.join(dataset_path, case_name, "input.yaml")
        
        if not os.path.exists(input_yaml_path):
            print(f"WARNING: input.yaml not found for {case_name}, assuming no obstacles")
            obstacles_list.append(None)
            continue
            
        try:
            with open(input_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                
            map_data = data.get('map', {})
            obstacles = map_data.get('obstacles', [])
            
            if obstacles:
                # Convert obstacle positions to tensor
                obstacle_positions = []
                for obs in obstacles:
                    obstacle_positions.append([float(obs[0]), float(obs[1])])
                obstacles_tensor = torch.tensor(obstacle_positions, dtype=torch.float32)
                obstacles_list.append(obstacles_tensor)
            else:
                # No obstacles in this case
                obstacles_list.append(None)
                
        except Exception as e:
            print(f"ERROR: Error loading obstacles from {input_yaml_path}: {e}")
            obstacles_list.append(None)
    
    # Check if any cases have obstacles
    valid_obstacles = [obs for obs in obstacles_list if obs is not None]
    
    if not valid_obstacles:
        # No obstacles found in any case
        return None
    
    # For cases with obstacles, return them; for cases without, we'll need to handle separately
    # For now, return the obstacles from cases that have them
    # TODO: This could be improved to handle mixed obstacle/no-obstacle cases better
    return valid_obstacles[0] if len(valid_obstacles) == 1 else torch.stack(valid_obstacles, dim=0)

def compute_best_distance_progress_reward(current_positions: torch.Tensor, goal_positions: torch.Tensor,
                                         best_distances: torch.Tensor,
                                         progress_weight: float = 1.0, success_threshold: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute progress reward based on "best-distance-so-far" system.
    Only rewards agents when they achieve a new best (minimum) distance to their goal.
    Idle behavior (no progress) receives neutral reward (0).
    
    Args:
        current_positions: Current agent positions [batch_size, num_agents, 2] or [num_agents, 2]
        goal_positions: Goal positions [batch_size, num_agents, 2] or [num_agents, 2]
        best_distances: Current best distances achieved so far [batch_size, num_agents] or [num_agents]
        progress_weight: Weight for progress reward scaling
        success_threshold: Distance threshold to consider agent has reached goal
        
    Returns:
        Tuple of (progress_reward, updated_best_distances)
        - progress_reward: Progress reward scalar for the batch
        - updated_best_distances: Updated best distances [batch_size, num_agents] or [num_agents]
    """
    # Handle both 2D and 3D tensor inputs
    if current_positions.dim() == 2:
        # Add batch dimension if missing [num_agents, 2] -> [1, num_agents, 2]
        current_positions = current_positions.unsqueeze(0)
        goal_positions = goal_positions.unsqueeze(0)
        best_distances = best_distances.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Compute current distances to goals
    current_distances = torch.norm(current_positions - goal_positions, dim=2)  # [batch_size, num_agents]
    
    # Handle first timestep: initialize best_distances if they are inf
    is_first_timestep = torch.isinf(best_distances)
    initial_best_distances = torch.where(is_first_timestep, current_distances, best_distances)
    
    # Find agents that achieved new best (lower) distances (but not on first timestep)
    progress_mask = (current_distances < initial_best_distances) & (~is_first_timestep)  # [batch_size, num_agents]
    
    # Exclude agents that have already reached their goals from progress rewards
    agents_at_goal_mask = current_distances <= success_threshold  # [batch_size, num_agents]
    eligible_progress_mask = progress_mask & (~agents_at_goal_mask)  # [batch_size, num_agents]
    
    # Update best distances: use current distance if it's better, or if this is first timestep
    updated_best_distances = torch.where(
        (current_distances < initial_best_distances) | is_first_timestep, 
        current_distances, 
        initial_best_distances
    )
    
    # Calculate progress reward only for eligible agents
    total_reward = torch.tensor(0.0, device=current_positions.device, dtype=torch.float32)
    
    if torch.any(eligible_progress_mask):
        # Compute progress amount for eligible agents
        progress_amounts = initial_best_distances[eligible_progress_mask] - current_distances[eligible_progress_mask]
        
        # Clamp progress amounts to avoid inf values (should not happen but safety check)
        progress_amounts = torch.clamp(progress_amounts, max=100.0)  # Max reasonable progress per step
        
        # Scale progress by progress weight
        scaled_progress = progress_amounts * progress_weight
        
        # Sum progress reward across progressing agents only (not averaged by total agents)
        total_reward = torch.sum(scaled_progress)
    
    # Remove batch dimension if input was 2D
    if squeeze_output:
        updated_best_distances = updated_best_distances.squeeze(0)
    
    return total_reward, updated_best_distances

def compute_obstacle_aware_progress_reward(current_positions: torch.Tensor, goal_positions: torch.Tensor,
                                          best_distances: torch.Tensor, obstacles: torch.Tensor,
                                          progress_weight: float = 1.0, success_threshold: float = 1.0,
                                          danger_zone_radius: float = 2.0, danger_penalty_weight: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute obstacle-aware progress reward that actively discourages moving towards obstacles.
    
    KEY BEHAVIOR:
    - NO progress reward when moving towards obstacles (even if moving towards goal)
    - Active penalty when getting too close to obstacles, scaled by best distance achieved
    - This forces agents to actively avoid danger and learn to reroute
    
    Args:
        current_positions: Current agent positions [batch_size, num_agents, 2] or [num_agents, 2]
        goal_positions: Goal positions [batch_size, num_agents, 2] or [num_agents, 2]
        best_distances: Current best distances achieved so far [batch_size, num_agents] or [num_agents]
        obstacles: Obstacle grid [batch_size, grid_h, grid_w] or list of obstacle positions
        progress_weight: Weight for progress reward scaling
        success_threshold: Distance threshold to consider agent has reached goal
        danger_zone_radius: Radius around obstacles considered dangerous
        danger_penalty_weight: Weight for danger zone penalty
        
    Returns:
        Tuple of (progress_reward, updated_best_distances)
        - progress_reward: Progress reward scalar for the batch (can be negative due to penalties)
        - updated_best_distances: Updated best distances [batch_size, num_agents] or [num_agents]
    """
    # Handle both 2D and 3D tensor inputs
    if current_positions.dim() == 2:
        # Add batch dimension if missing [num_agents, 2] -> [1, num_agents, 2]
        current_positions = current_positions.unsqueeze(0)
        goal_positions = goal_positions.unsqueeze(0)
        best_distances = best_distances.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, num_agents = current_positions.shape[:2]
    device = current_positions.device
    
    # Compute current distances to goals
    current_distances = torch.norm(current_positions - goal_positions, dim=2)  # [batch_size, num_agents]
    
    # Handle first timestep: initialize best_distances if they are inf
    is_first_timestep = torch.isinf(best_distances)
    initial_best_distances = torch.where(is_first_timestep, current_distances, best_distances)
    
    # === OBSTACLE DANGER DETECTION ===
    danger_penalties = torch.zeros_like(current_distances)  # [batch_size, num_agents]
    moving_towards_obstacles = torch.zeros_like(current_distances, dtype=torch.bool)  # [batch_size, num_agents]
    
    for b in range(batch_size):
        # Convert obstacles to list of positions if it's a grid
        if isinstance(obstacles, torch.Tensor) and obstacles.dim() == 3:
            # obstacles is [batch, grid_h, grid_w]
            obstacle_positions = []
            grid_h, grid_w = obstacles.shape[1], obstacles.shape[2]
            for y in range(grid_h):
                for x in range(grid_w):
                    if obstacles[b, y, x] > 0.5:  # Obstacle present
                        obstacle_positions.append([x, y])  # Note: x,y order
            obstacle_positions = torch.tensor(obstacle_positions, device=device, dtype=torch.float32)
        elif isinstance(obstacles, list) and len(obstacles) > 0:
            # obstacles is list of positions
            obstacle_positions = torch.tensor(obstacles, device=device, dtype=torch.float32)
        else:
            # No obstacles, skip danger detection
            continue
            
        if len(obstacle_positions) == 0:
            continue
            
        for a in range(num_agents):
            agent_pos = current_positions[b, a]  # [2]
            goal_pos = goal_positions[b, a]  # [2]
            
            # Find closest obstacle
            if len(obstacle_positions) > 0:
                obstacle_distances = torch.norm(obstacle_positions - agent_pos.unsqueeze(0), dim=1)  # [num_obstacles]
                min_obstacle_dist = torch.min(obstacle_distances)
                closest_obstacle_pos = obstacle_positions[torch.argmin(obstacle_distances)]
                
                # Check if agent is in danger zone - apply penalty proportional to best distance
                # The closer to obstacle and the better the best distance, the higher the penalty
                if min_obstacle_dist <= danger_zone_radius:
                    # Calculate penalty based on how close to obstacle and best distance achieved
                    danger_proximity = (danger_zone_radius - min_obstacle_dist) / danger_zone_radius  # 0 to 1
                    best_dist_factor = initial_best_distances[b, a] + 1.0  # Add 1 to avoid division by zero
                    
                    # Penalty scales with both proximity to danger and progress made (best distance)
                    # This makes agents that have made progress more cautious about dangers
                    danger_penalties[b, a] = danger_penalty_weight * danger_proximity * best_dist_factor
                
                # Check if agent is moving towards closest obstacle
                # Direction from agent to goal
                goal_direction = goal_pos - agent_pos
                goal_direction_norm = goal_direction / (torch.norm(goal_direction) + 1e-8)
                
                # Direction from agent to closest obstacle
                obstacle_direction = closest_obstacle_pos - agent_pos
                obstacle_direction_norm = obstacle_direction / (torch.norm(obstacle_direction) + 1e-8)
                
                # Check if directions are similar (dot product > threshold)
                # Lower threshold means we're more strict about what counts as "moving towards obstacle"
                direction_similarity = torch.dot(goal_direction_norm, obstacle_direction_norm)
                if direction_similarity > 0.2:  # Moving towards obstacle (more strict threshold)
                    moving_towards_obstacles[b, a] = True
    
    # === PROGRESS REWARD COMPUTATION ===
    # Find agents that achieved new best (lower) distances (but not on first timestep)
    progress_mask = (current_distances < initial_best_distances) & (~is_first_timestep)  # [batch_size, num_agents]
    
    # Exclude agents that have already reached their goals from progress rewards
    agents_at_goal_mask = current_distances <= success_threshold  # [batch_size, num_agents]
    
    # STRICT RULE: NO progress reward if moving towards obstacles, even if making progress towards goal
    safe_progress_mask = progress_mask & (~agents_at_goal_mask) & (~moving_towards_obstacles)  # [batch_size, num_agents]
    
    # Update best distances: use current distance if it's better, or if this is first timestep
    # Do NOT include danger penalties in best distance - keep it as pure goal distance
    updated_best_distances = torch.where(
        (current_distances < initial_best_distances) | is_first_timestep, 
        current_distances, 
        initial_best_distances
    )
    
    # Calculate progress reward only for safe-progress agents (those NOT moving towards obstacles)
    total_reward = torch.tensor(0.0, device=device, dtype=torch.float32)
    
    if torch.any(safe_progress_mask):
        # Compute progress amount for safe-progress agents
        progress_amounts = initial_best_distances[safe_progress_mask] - current_distances[safe_progress_mask]
        
        # Clamp progress amounts to avoid inf values
        progress_amounts = torch.clamp(progress_amounts, max=100.0)
        
        # Scale progress by progress weight
        scaled_progress = progress_amounts * progress_weight
        
        # Sum progress reward across safe-progressing agents
        total_reward += torch.sum(scaled_progress)
    
    # Apply danger penalties to total reward (negative contribution)
    # This creates a strong incentive to avoid danger zones
    if torch.any(danger_penalties > 0):
        total_penalty = torch.sum(danger_penalties)
        total_reward -= total_penalty
    
    # Remove batch dimension if input was 2D
    if squeeze_output:
        updated_best_distances = updated_best_distances.squeeze(0)
    
    return total_reward, updated_best_distances

def compute_continuous_collision_penalty(positions: torch.Tensor, 
                                       collision_config: Dict) -> torch.Tensor:
    """
    Compute continuous collision penalty based on inverse Manhattan distance.
    Penalizes agents that are close to each other (within threshold distance).
    Uses smooth, continuous penalty for better numerical stability and gradients.
    
    Args:
        positions: Agent positions [batch, num_agents, 2]
        collision_config: Configuration parameters
        
    Returns:
        collision_penalty: Penalty per agent [batch, num_agents]
    """
    batch_size, num_agents = positions.shape[:2]
    device = positions.device
    
    # Get collision threshold and weights
    threshold = collision_config.get('collision_threshold', 2.0)
    vertex_weight = collision_config.get('vertex_collision_weight', 1.0)
    
    # Initialize penalty tensor
    collision_penalty = torch.zeros(batch_size, num_agents, device=device)
    
    # Compute pairwise distances for all agent pairs
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            # Manhattan distance between agents i and j
            pos_i = positions[:, i, :]  # [batch, 2]
            pos_j = positions[:, j, :]  # [batch, 2]
            
            manhattan_dist = torch.sum(torch.abs(pos_i - pos_j), dim=1)  # [batch]
            
            # Apply penalty for agents within threshold distance
            within_threshold = manhattan_dist < threshold
            
            if within_threshold.any():
                # Continuous inverse distance penalty: penalty = weight * (threshold - dist) / (dist + epsilon)
                # This gives smooth gradients and penalizes closer agents more heavily
                epsilon = 0.01  # Small epsilon for numerical stability
                
                # Smooth penalty function: higher penalty for closer agents
                # penalty = weight * (threshold - dist)^2 / (dist + epsilon)
                # This ensures penalty -> 0 as dist -> threshold, and penalty -> infinity as dist -> 0
                distance_diff = torch.clamp(threshold - manhattan_dist, min=0.0)  # Only positive differences
                inv_dist_penalty = vertex_weight * (distance_diff ** 2) / (manhattan_dist + epsilon)
                
                # Apply penalty only when within threshold
                penalty_value = torch.where(within_threshold, inv_dist_penalty, torch.zeros_like(inv_dist_penalty))
                
                # Add penalty to both agents involved in the collision
                collision_penalty[:, i] += penalty_value
                collision_penalty[:, j] += penalty_value
    
    return collision_penalty
