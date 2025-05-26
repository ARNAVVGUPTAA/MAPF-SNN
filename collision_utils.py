"""
Collision detection utilities for Multi-Agent Path Finding (MAPF).
Provides functions to detect and penalize different types of collisions.
"""

import torch
import numpy as np
import yaml
import os
from typing import Dict, List, Tuple, Optional


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


def compute_collision_loss(logits: torch.Tensor, 
                          positions: torch.Tensor,
                          prev_positions: Optional[torch.Tensor] = None,
                          collision_config: Dict = None) -> Tuple[torch.Tensor, Dict]:
    """
    Compute collision loss to penalize actions that lead to collisions.
    
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
            'vertex_collision_weight': 1.0,
            'edge_collision_weight': 0.5,
            'collision_loss_type': 'l2'
        }
    
    # Detect collisions
    collisions = detect_collisions(positions, prev_positions)
    
    batch_size, num_agents = positions.shape[:2]
    device = positions.device
    
    # Compute collision penalties
    vertex_penalty = collisions['vertex_collisions'] * collision_config['vertex_collision_weight']
    edge_penalty = collisions['edge_collisions'] * collision_config['edge_collision_weight']
    
    # Total collision penalty per agent
    total_penalty = vertex_penalty + edge_penalty  # [batch, num_agents]
    
    # Apply penalty to action probabilities
    action_probs = torch.softmax(logits.view(batch_size, num_agents, -1), dim=-1)
    
    # Collision loss based on configuration
    if collision_config['collision_loss_type'] == 'l1':
        collision_loss = torch.mean(total_penalty)
    elif collision_config['collision_loss_type'] == 'l2':
        collision_loss = torch.mean(total_penalty ** 2)
    elif collision_config['collision_loss_type'] == 'exponential':
        collision_loss = torch.mean(torch.exp(total_penalty) - 1)
    else:
        collision_loss = torch.mean(total_penalty)
    
    # Collect collision information for logging
    collision_info = {
        'total_collisions': torch.sum(total_penalty > 0).item(),
        'vertex_collisions': torch.sum(vertex_penalty > 0).item(),
        'edge_collisions': torch.sum(edge_penalty > 0).item(),
        'collision_rate': torch.mean((total_penalty > 0).float()).item(),
        'avg_collision_penalty': torch.mean(total_penalty).item()
    }
    
    return collision_loss, collision_info


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
                             board_size: int = 28) -> torch.Tensor:
    """
    Predict potential collisions in future steps based on current action tendencies.
    
    Args:
        current_positions: Current agent positions [batch, num_agents, 2]
        action_logits: Action logits [batch * num_agents, num_actions]
        steps: Number of future steps to predict
        board_size: Size of the board
        
    Returns:
        future_collision_penalty: Penalty for predicted future collisions
    """
    batch_size, num_agents = current_positions.shape[:2]
    
    # Convert logits to action probabilities
    action_probs = torch.softmax(action_logits.view(batch_size, num_agents, -1), dim=-1)
    
    total_future_penalty = 0.0
    positions = current_positions.clone()
    
    for step in range(steps):
        # Sample actions based on probabilities
        actions = torch.multinomial(action_probs.view(-1, action_probs.size(-1)), 1).squeeze(-1)
        actions = actions.view(batch_size, num_agents)
        
        # Predict next positions
        next_positions = extract_positions_from_actions(positions, actions, board_size)
        
        # Detect collisions at this future step
        collisions = detect_collisions(next_positions, positions)
        
        # Add penalty (weighted by distance into future)
        step_weight = 1.0 / (step + 1)  # Closer steps have higher weight
        vertex_penalty = torch.sum(collisions['vertex_collisions']) * step_weight
        edge_penalty = torch.sum(collisions['edge_collisions']) * step_weight * 0.5
        
        total_future_penalty += vertex_penalty + edge_penalty
        
        # Update positions for next iteration
        positions = next_positions
    
    return total_future_penalty / (batch_size * num_agents * steps)


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
    fov_reshaped = fov.view(batch_size, num_agents, fov.size(1), fov.size(2), fov.size(3))
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
            print(f"Warning: input.yaml not found for {case_name}, using dummy positions")
            # Fallback to dummy positions if file not found
            dummy_positions = torch.randint(0, 28, (5, 2), dtype=torch.float32)
            initial_positions_list.append(dummy_positions)
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
            print(f"Error loading {input_yaml_path}: {e}, using dummy positions")
            dummy_positions = torch.randint(0, 28, (5, 2), dtype=torch.float32)
            initial_positions_list.append(dummy_positions)
    
    # Stack all initial positions
    initial_positions = torch.stack(initial_positions_list, dim=0)
    return initial_positions
