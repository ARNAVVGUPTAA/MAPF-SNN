"""
Expert trajectory utilities for teacher-forcing in SNN training.
Provides functions to extract and convert A*/CBS expert solutions to action sequences.
"""

import os
import yaml
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from debug_utils import debug_print
def position_to_action(prev_pos: Tuple[int, int], curr_pos: Tuple[int, int]) -> int:
    """
    Convert position change to action index.
    Actions: 0=stay, 1=right, 2=up, 3=left, 4=down
    
    Args:
        prev_pos: Previous position (x, y)
        curr_pos: Current position (x, y)
    
    Returns:
        Action index (0-4)
    """
    if prev_pos == curr_pos:
        return 0  # Stay
    
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    
    if dx == 1 and dy == 0:
        return 1  # Right
    elif dx == 0 and dy == 1:
        return 2  # Up
    elif dx == -1 and dy == 0:
        return 3  # Left
    elif dx == 0 and dy == -1:
        return 4  # Down
    else:
        # Invalid move, default to stay
        return 0


def extract_expert_actions_from_solution(solution_path: str, num_agents: int, max_time: int) -> Optional[torch.Tensor]:
    """
    Extract expert action sequence from solution.yaml file.
    
    Args:
        solution_path: Path to solution.yaml file
        num_agents: Number of agents
        max_time: Maximum time steps to extract
    
    Returns:
        Tensor of shape [time_steps, num_agents] with action indices, or None if failed
    """
    try:
        with open(solution_path, 'r') as f:
            solution = yaml.safe_load(f)
        
        if 'schedule' not in solution:
            return None
        
        schedule = solution['schedule']
        
        # Initialize action tensor
        actions = torch.zeros(max_time, num_agents, dtype=torch.long)
        
        for agent_idx in range(num_agents):
            agent_key = f'agent{agent_idx}'
            
            if agent_key not in schedule:
                continue
            
            agent_schedule = schedule[agent_key]
            
            # Extract positions for each time step
            positions = []
            for step in agent_schedule:
                positions.append((step['x'], step['y']))
            
            # Convert position changes to actions
            for t in range(min(len(positions) - 1, max_time - 1)):
                prev_pos = positions[t]
                curr_pos = positions[t + 1]
                action = position_to_action(prev_pos, curr_pos)
                actions[t, agent_idx] = action
            
            # Last time step: stay action
            if len(positions) <= max_time:
                actions[len(positions) - 1:, agent_idx] = 0  # Stay
        
        return actions
        
    except Exception as e:
        debug_print(f"Error loading expert actions from {solution_path}: {e}")
        return None


def load_expert_demonstrations(dataset_root: str, case_indices: List[int], 
                             mode: str = "train", max_time: int = 50) -> Optional[torch.Tensor]:
    """
    Load expert demonstrations for given case indices.
    
    Args:
        dataset_root: Root directory of dataset
        case_indices: List of case indices to load
        mode: Dataset mode ('train' or 'val')
        max_time: Maximum time steps to load
    
    Returns:
        Expert action tensor [num_cases, time_steps, num_agents] or None
    """
    dataset_path = os.path.join(dataset_root, mode)
    expert_actions_list = []
    
    for case_idx in case_indices:
        case_name = f"case_{case_idx}"
        solution_path = os.path.join(dataset_path, case_name, "solution.yaml")
        
        if not os.path.exists(solution_path):
            debug_print(f"Warning: solution.yaml not found for {case_name}")
            # Generate dummy expert actions (all stay)
            dummy_actions = torch.zeros(max_time, 5, dtype=torch.long)  # Assume 5 agents
            expert_actions_list.append(dummy_actions)
            continue
        
        # Extract expert actions from solution
        expert_actions = extract_expert_actions_from_solution(solution_path, 5, max_time)
        
        if expert_actions is None:
            # Fallback to dummy actions
            dummy_actions = torch.zeros(max_time, 5, dtype=torch.long)
            expert_actions_list.append(dummy_actions)
        else:
            expert_actions_list.append(expert_actions)
    
    if not expert_actions_list:
        return None
    
    # Stack all expert actions
    expert_demonstrations = torch.stack(expert_actions_list, dim=0)  # [num_cases, time_steps, num_agents]
    return expert_demonstrations


def create_mixed_batch(regular_data: Dict, expert_data: torch.Tensor, expert_ratio: float = 0.3) -> Dict:
    """
    Create a mixed batch combining regular training data with expert demonstrations.
    
    Args:
        regular_data: Regular training batch data
        expert_data: Expert demonstration tensor [num_cases, time_steps, num_agents]
        expert_ratio: Ratio of expert samples in the mixed batch
    
    Returns:
        Mixed batch dictionary with 'is_expert' mask
    """
    regular_batch_size = regular_data['states'].shape[0]
    expert_batch_size = int(regular_batch_size * expert_ratio)
    
    if expert_batch_size == 0 or expert_data is None:
        # No expert data, return regular data with expert mask
        regular_data['is_expert'] = torch.zeros(regular_batch_size, dtype=torch.bool)
        return regular_data
    
    # Sample expert indices
    total_expert_cases = expert_data.shape[0]  # expert_data is tensor of actions
    expert_indices = torch.randint(0, total_expert_cases, (expert_batch_size,))
    
    # Extract expert action samples
    expert_actions = expert_data[expert_indices]  # [expert_batch_size, time_steps, num_agents]
    
    # Select regular indices to keep
    regular_indices = torch.randperm(regular_batch_size)[:regular_batch_size - expert_batch_size]
    
    # Get regular actions subset
    regular_actions = regular_data['actions'][regular_indices]
    
    debug_print(f"DEBUG: Regular actions shape: {regular_actions.shape}")
    debug_print(f"DEBUG: Expert actions shape: {expert_actions.shape}")
    
    # Handle dimension mismatches
    # Regular actions: [batch_subset, time, agents]
    # Expert actions: [expert_batch_size, time_steps, num_agents]
    
    # Ensure expert actions are on the same device as regular actions
    if expert_actions.device != regular_actions.device:
        expert_actions = expert_actions.to(regular_actions.device)
        debug_print(f"DEBUG: Moved expert actions to device {regular_actions.device}")
    
    # Fix time dimension mismatch
    if regular_actions.shape[1] != expert_actions.shape[1]:
        target_time = regular_actions.shape[1]
        debug_print(f"DEBUG: Time dimension mismatch - Regular: {regular_actions.shape[1]}, Expert: {expert_actions.shape[1]}")
        
        if expert_actions.shape[1] > target_time:
            expert_actions = expert_actions[:, :target_time, :]
            debug_print(f"DEBUG: Trimmed expert actions to {expert_actions.shape}")
        elif expert_actions.shape[1] < target_time:
            # Pad with stay actions (0)
            padding_size = target_time - expert_actions.shape[1]
            padding = torch.zeros(expert_actions.shape[0], padding_size, expert_actions.shape[2], 
                                dtype=expert_actions.dtype, device=expert_actions.device)
            expert_actions = torch.cat([expert_actions, padding], dim=1)
            debug_print(f"DEBUG: Padded expert actions to {expert_actions.shape}")
    
    # Fix agent dimension mismatch
    if regular_actions.shape[2] != expert_actions.shape[2]:
        target_agents = regular_actions.shape[2]
        debug_print(f"WARNING: Agent count mismatch - Regular: {regular_actions.shape[2]}, Expert: {expert_actions.shape[2]}")
        
        if expert_actions.shape[2] > target_agents:
            expert_actions = expert_actions[:, :, :target_agents]
            debug_print(f"DEBUG: Trimmed expert agents to {expert_actions.shape}")
        elif expert_actions.shape[2] < target_agents:
            # Pad with zeros or duplicate last agent
            agent_padding_size = target_agents - expert_actions.shape[2]
            agent_padding = torch.zeros(expert_actions.shape[0], expert_actions.shape[1], agent_padding_size,
                                       dtype=expert_actions.dtype, device=expert_actions.device)
            expert_actions = torch.cat([expert_actions, agent_padding], dim=2)
            debug_print(f"DEBUG: Padded expert agents to {expert_actions.shape}")
    
    # Now concatenate along batch dimension
    debug_print(f"DEBUG: Final shapes - Regular: {regular_actions.shape}, Expert: {expert_actions.shape}")
    
    # Build mixed batch
    mixed_batch = {}
    
    # Mix states: Use reduced regular states + duplicate some regular states for expert samples
    regular_states_subset = regular_data['states'][regular_indices]
    
    # For expert samples, we need corresponding state observations
    # Since expert demos don't include states, duplicate some regular states
    expert_states_indices = torch.randint(0, len(regular_indices), (expert_batch_size,))
    expert_states = regular_data['states'][regular_indices[expert_states_indices]]
    
    mixed_batch['states'] = torch.cat([regular_states_subset, expert_states], dim=0)
    
    # Mix actions: concat regular and expert actions along batch dimension
    mixed_batch['actions'] = torch.cat([regular_actions, expert_actions], dim=0)
    
    # Handle optional fields (positions, goals, obstacles) only from regular_data
    for key in ['positions', 'goals', 'obstacles']:
        if key in regular_data and regular_data[key] is not None:
            mixed_batch[key] = regular_data[key][regular_indices]
    
    # Create expert mask - mark which samples are expert data
    total_mixed_size = regular_actions.shape[0] + expert_actions.shape[0]
    expert_mask = torch.zeros(total_mixed_size, dtype=torch.bool)
    expert_mask[regular_actions.shape[0]:] = True  # Mark expert samples as True
    mixed_batch['is_expert'] = expert_mask
    
    debug_print(f"DEBUG: Mixed batch - States: {mixed_batch['states'].shape}, Actions: {mixed_batch['actions'].shape}")
    debug_print(f"DEBUG: Expert mask: {expert_mask.sum().item()}/{len(expert_mask)}")
    
    return mixed_batch


def compute_expert_weight_decay(epoch: int, max_expert_epochs: int = 50) -> float:
    """
    Compute decaying weight for expert loss over training epochs.
    
    Args:
        epoch: Current epoch (0-indexed)
        max_expert_epochs: Number of epochs to use expert data
    
    Returns:
        Expert loss weight (0.0 to 1.0)
    """
    if epoch >= max_expert_epochs:
        return 0.0
    
    # Linear decay from 1.0 to 0.0
    return 1.0 - (epoch / max_expert_epochs)


def should_apply_expert_training(epoch: int, expert_interval: int = 10) -> bool:
    """
    Check if expert training should be applied at the current epoch.
    Expert training is applied at intervals (epoch 10, 20, 30, 40...).
    
    Args:
        epoch: Current epoch (0-indexed)
        expert_interval: Interval between expert training epochs
    
    Returns:
        True if expert training should be applied, False otherwise
    """
    # Apply expert training at intervals starting from epoch 10
    # epoch 10, 20, 30, 40, etc.
    return epoch > 0 and (epoch + 1) % expert_interval == 0


def compute_interval_expert_weight(expert_weight: float = 0.6) -> float:
    """
    Compute the weight for expert loss for interval-based expert training.
    Returns a constant weight value when expert training is active.
    
    Args:
        expert_weight: Fixed weight for expert samples
    
    Returns:
        Expert loss weight
    """
    return expert_weight
