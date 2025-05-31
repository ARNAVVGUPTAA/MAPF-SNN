"""
Expert trajectory utilities for teacher-forcing in SNN training.
Provides functions to extract and convert A*/CBS expert solutions to action sequences.
"""

import os
import yaml
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def position_to_action(prev_pos: Tuple[int, int], curr_pos: Tuple[int, int]) -> int:
    """
    Convert position change to action index.
    Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
    
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
    
    if dx == 0 and dy == 1:
        return 1  # Up
    elif dx == 0 and dy == -1:
        return 2  # Down
    elif dx == -1 and dy == 0:
        return 3  # Left
    elif dx == 1 and dy == 0:
        return 4  # Right
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
            for t in range(min(len(positions) - 1, max_time)):
                action = position_to_action(positions[t], positions[t + 1])
                actions[t, agent_idx] = action
            
            # Fill remaining time steps with stay action if needed
            for t in range(len(positions) - 1, max_time):
                actions[t, agent_idx] = 0  # Stay action
        
        return actions
    
    except Exception as e:
        print(f"Error extracting expert actions from {solution_path}: {e}")
        return None


def load_expert_demonstrations(dataset_root: str, case_indices: List[int], 
                              mode: str = "train", num_agents: int = 5, 
                              max_time: int = 25) -> Dict[int, torch.Tensor]:
    """
    Load expert demonstrations for specified cases.
    
    Args:
        dataset_root: Root directory of dataset
        case_indices: List of case indices to load
        mode: 'train' or 'val'
        num_agents: Number of agents
        max_time: Maximum time steps
    
    Returns:
        Dictionary mapping case_index -> expert_actions tensor
    """
    expert_demos = {}
    
    if mode == "train":
        dataset_path = os.path.join(dataset_root, "train")
    else:
        dataset_path = os.path.join(dataset_root, "val")
    
    for case_idx in case_indices:
        case_dir = os.path.join(dataset_path, f"case_{case_idx}")
        solution_path = os.path.join(case_dir, "solution.yaml")
        
        if os.path.exists(solution_path):
            expert_actions = extract_expert_actions_from_solution(
                solution_path, num_agents, max_time
            )
            if expert_actions is not None:
                expert_demos[case_idx] = expert_actions
    
    return expert_demos


def create_mixed_batch(normal_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      expert_demos: Dict[int, torch.Tensor],
                      batch_case_indices: List[int],
                      expert_ratio: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a mixed batch with normal training data and expert demonstrations.
    
    Args:
        normal_batch: (states, trajectories, gso) from normal data loader
        expert_demos: Dictionary of expert demonstrations
        batch_case_indices: Case indices corresponding to the batch
        expert_ratio: Ratio of expert data to include (0.25 = 25%)
    
    Returns:
        (states, trajectories, gso, expert_mask) where expert_mask indicates which samples use expert data
    """
    states, trajectories, gso = normal_batch
    batch_size = states.shape[0]
    
    # Determine which samples will use expert data
    num_expert_samples = int(batch_size * expert_ratio)
    expert_indices = np.random.choice(batch_size, num_expert_samples, replace=False)
    
    # Create expert mask
    expert_mask = torch.zeros(batch_size, dtype=torch.bool)
    expert_mask[expert_indices] = True
    
    # Replace trajectories with expert actions for selected samples
    modified_trajectories = trajectories.clone()
    
    for i, case_idx in enumerate(batch_case_indices):
        if i in expert_indices and case_idx in expert_demos:
            expert_actions = expert_demos[case_idx]
            time_steps = min(expert_actions.shape[0], trajectories.shape[1])
            
            # Replace with expert actions
            modified_trajectories[i, :time_steps, :] = expert_actions[:time_steps, :]
    
    return states, modified_trajectories, gso, expert_mask


def compute_expert_loss_weight(epoch: int, max_expert_epochs: int = 5) -> float:
    """
    Compute the weight for expert loss based on current epoch.
    Expert loss weight decreases from 1.0 to 0.0 over the first few epochs.
    
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
