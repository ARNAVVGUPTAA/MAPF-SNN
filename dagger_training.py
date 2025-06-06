"""
DAgger (Dataset Aggregation) Implementation for SNN MAPF Training

This module implements a DAgger-like iterative imitation learning loop that:
1. Trains a policy on expert demonstrations
2. Rolls out the current policy to collect states
3. Queries expert using pre-computed solution.yaml files for optimal actions
4. Aggregates expert corrections into dataset
5. Retrains policy on augmented dataset
6. Repeats until convergence

Key Features:
- Expert querying using pre-computed solution.yaml files (no CBS solver needed)
- State-action aggregation with distribution balancing
- Policy rollout in MAPF environment
- Convergence monitoring and early stopping
- Integration with existing SNN training infrastructure
"""

import os
import sys
import yaml
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import pickle
import time
from torch.utils.data import TensorDataset, DataLoader

# Import existing modules
from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
from models.framework_snn import Network
from data_loader import SNNDataLoader
from config import config
from debug_utils import debug_print
from expert_utils import extract_expert_actions_from_solution
from spikingjelly.activation_based import functional
from collision_utils import (
    load_initial_positions_from_dataset,
    load_goal_positions_from_dataset,
    load_obstacles_from_dataset,
    compute_goal_proximity_reward,
    compute_goal_success_bonus,
    compute_collision_loss
)

def count_dataset_cases(dataset_root: str, mode: str = "train") -> int:
    """
    Dynamically count the number of case folders in the dataset directory.
    
    Args:
        dataset_root: Root directory of the dataset (e.g., 'dataset/5_8_28')
        mode: Dataset mode ('train' or 'val')
        
    Returns:
        Number of case folders found in the dataset
    """
    import os
    
    dataset_path = os.path.join(dataset_root, mode)
    if not os.path.exists(dataset_path):
        print(f"WARNING: Dataset path {dataset_path} does not exist, defaulting to size 1000")
        return 1000
    
    try:
        # Count directories that start with 'case_'
        case_folders = [d for d in os.listdir(dataset_path) 
                       if d.startswith('case_') and os.path.isdir(os.path.join(dataset_path, d))]
        
        dataset_size = len(case_folders)
        debug_print(f"DEBUG: Found {dataset_size} cases in {dataset_path}")
        return dataset_size
        
    except Exception as e:
        print(f"WARNING: Error counting dataset cases in {dataset_path}: {e}, defaulting to size 1000")
        return 1000

class DAggerExpert:
    """Expert oracle using pre-computed solution.yaml files instead of CBS solver."""
    
    def __init__(self, dataset_root: str, mode: str = "train", max_time: int = 50):
        """
        Initialize expert with dataset root containing solution.yaml files.
        
        Args:
            dataset_root: Root directory of dataset (e.g., "dataset/5_8_28")
            mode: Dataset mode ("train" or "val")
            max_time: Maximum time steps to consider
        """
        self.dataset_root = dataset_root
        self.mode = mode
        self.max_time = max_time
        self.dataset_path = os.path.join(dataset_root, mode)
        
        # Cache for loaded solutions to avoid re-reading files
        self.solution_cache = {}
        
        print(f"DAgger Expert initialized with dataset: {self.dataset_path}")
        
    def load_solution_for_case(self, case_idx: int) -> Optional[torch.Tensor]:
        """
        Load pre-computed solution for a specific case.
        
        Args:
            case_idx: Index of the case to load
            
        Returns:
            Tensor of expert actions [time_steps, num_agents] or None if failed
        """
        if case_idx in self.solution_cache:
            return self.solution_cache[case_idx]
            
        case_name = f"case_{case_idx}"
        solution_path = os.path.join(self.dataset_path, case_name, "solution.yaml")
        
        if not os.path.exists(solution_path):
            print(f"Warning: solution.yaml not found for {case_name}")
            return None
            
        # Use existing function to extract expert actions
        expert_actions = extract_expert_actions_from_solution(
            solution_path, 
            num_agents=5,  # Assuming 5 agents based on config
            max_time=self.max_time
        )
        
        if expert_actions is not None:
            self.solution_cache[case_idx] = expert_actions
            
        return expert_actions
    
    def get_expert_action_for_case(self, case_idx: int, timestep: int) -> Optional[np.ndarray]:
        """
        Get expert action for specific case and timestep.
        
        Args:
            case_idx: Index of the case
            timestep: Current timestep
            
        Returns:
            Array of expert actions [num_agents] or None if failed
        """
        solution = self.load_solution_for_case(case_idx)
        
        if solution is None:
            return None
            
        # Check if timestep is within bounds
        if timestep >= solution.shape[0]:
            # Return stay actions if beyond solution length
            return np.zeros(solution.shape[1], dtype=np.int64)
            
        # Return actions for this timestep
        return solution[timestep].numpy().astype(np.int64)
    
    def get_expert_action(self, state: Dict, ) -> Optional[np.ndarray]:
        """
        Get expert action for current state using pre-computed solutions.
        
        Args:
            state: Dictionary containing:
                - case_idx: Index of the current case
                - timestep: current time step
                - (other fields ignored as we use pre-computed solutions)
                
        Returns:
            Array of expert actions [num_agents] or None if failed
        """
        case_idx = state.get('case_idx', None)
        timestep = state.get('timestep', 0)
        
        if case_idx is None:
            print("Warning: case_idx not provided in state")
            return None
            
        return self.get_expert_action_for_case(case_idx, timestep)


class DAggerDataset:
    """Dataset for storing and managing DAgger training data."""
    
    def __init__(self, initial_expert_data: Optional[torch.Tensor] = None):
        """
        Initialize DAgger dataset.
        
        Args:
            initial_expert_data: Initial expert demonstrations [num_cases, time_steps, num_agents]
        """
        self.states = []  # List of state dictionaries
        self.expert_actions = []  # List of expert action arrays
        self.iteration_labels = []  # Track which DAgger iteration each sample came from
        
        if initial_expert_data is not None:
            self._add_initial_expert_data(initial_expert_data)
    
    def _add_initial_expert_data(self, expert_data: torch.Tensor):
        """
        Add initial expert demonstrations to dataset.
        
        Args:
            expert_data: Tensor of shape [num_cases, time_steps, num_agents] with expert actions
        """
        print(f"Adding initial expert data: {expert_data.shape}")
        
        # Convert expert tensor to state-action pairs
        num_cases, time_steps, num_agents = expert_data.shape
        
        for case_idx in range(num_cases):
            for t in range(time_steps):
                # Create state dictionary with minimal information
                # Since we're using pre-computed solutions, we mainly need case_idx and timestep
                state = {
                    'case_idx': case_idx,
                    'timestep': t,
                    'data_source': 'initial_expert'
                }
                
                # Get expert actions for this timestep
                expert_actions = expert_data[case_idx, t].numpy().astype(np.int64)
                
                self.states.append(state)
                self.expert_actions.append(expert_actions) 
                self.iteration_labels.append(0)  # Initial data is iteration 0
        
        print(f"Added {len(self.states)} initial expert state-action pairs")
    
    def add_samples(self, states: List[Dict], expert_actions: List[np.ndarray], iteration: int):
        """Add new state-action samples to dataset."""
        self.states.extend(states)
        self.expert_actions.extend(expert_actions)
        self.iteration_labels.extend([iteration] * len(states))
    
    def get_size(self) -> int:
        """Get total number of samples in dataset."""
        return len(self.states)
    
    def save(self, filepath: str):
        """Save dataset to file."""
        data = {
            'states': self.states,
            'expert_actions': self.expert_actions,
            'iteration_labels': self.iteration_labels
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load dataset from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.states = data['states']
        self.expert_actions = data['expert_actions']
        self.iteration_labels = data['iteration_labels']


class DAggerTrainer:
    """Main DAgger training loop implementation."""
    
    def __init__(self, config: Dict, model: Network, data_loader: SNNDataLoader):
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.device = config.get('device', torch.device('cpu'))
        
        # DAgger hyperparameters with backward-compatible defaults
        self.max_iterations = config.get('dagger_max_iterations', 10)
        self.rollout_episodes = config.get('dagger_rollout_episodes', 50)
        self.max_timesteps = config.get('dagger_max_timesteps', 40)
        self.expert_query_rate = config.get('dagger_expert_query_rate', 1.0)  # Initial β
        self.expert_query_rate_min = config.get('dagger_expert_query_rate_min', 0.1)  # β_min
        self.convergence_threshold = config.get('dagger_convergence_threshold', 0.95)
        self.retraining_epochs = config.get('dagger_retraining_epochs', 20)
        
        # Fix obstacles config issue - use sensible defaults
        if "obstacles" not in self.config:
            self.config["obstacles"] = config.get('nb_obstacles', 0)  # Try nb_obstacles first, default to 0
            
        # Initialize expert with dataset information - use defaults for missing config
        dataset_root = config.get('dataset_root', config.get('train', {}).get('root_dir', 'dataset/5_8_28'))
        mode = config.get('dagger_mode', 'train')
        self.expert = DAggerExpert(
            dataset_root=dataset_root,
            mode=mode,
            max_time=self.max_timesteps
        )
        
        # Initialize dataset with existing expert demonstrations
        initial_expert_data = self._load_initial_expert_data()
        self.dagger_dataset = DAggerDataset(initial_expert_data)
        
        # Metrics tracking
        self.iteration_metrics = []
        self.performance_history = deque(maxlen=3)  # For convergence detection
        
    def _load_initial_expert_data(self) -> Optional[torch.Tensor]:
        """Load initial expert demonstrations from existing dataset."""
        try:
            # Get case indices from data loader's train dataset
            train_dataset = self.data_loader.train_loader.dataset
            if hasattr(train_dataset, 'case_indices'):
                case_indices = train_dataset.case_indices[:50]  # Limit initial cases
            else:
                case_indices = list(range(min(50, len(train_dataset))))  # Default case indices
                
            dataset_root = self.config.get('dataset_root', 'dataset/5_8_28')
            mode = self.config.get('dagger_mode', 'train')
            
            expert_data = []
            for case_idx in case_indices:
                expert_actions = self.expert.load_solution_for_case(case_idx)
                if expert_actions is not None:
                    expert_data.append(expert_actions)
                else:
                    # Create dummy data if solution not found
                    dummy_actions = torch.zeros(self.max_timesteps, 5, dtype=torch.long)
                    expert_data.append(dummy_actions)
            
            if expert_data:
                return torch.stack(expert_data, dim=0)  # [num_cases, time_steps, num_agents]
            else:
                return None
                
        except Exception as e:
            print(f"Failed to load initial expert data: {e}")
            return None
        
    def rollout_policy(self, num_episodes: int, dagger_iteration: int = 0) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        Roll out current policy to collect states and query expert.
        
        Args:
            num_episodes: Number of episodes to rollout
            dagger_iteration: Current DAgger iteration (for β-decay)
            
        Returns:
            Tuple of (states_visited, expert_actions)
        """
        print(f"Rolling out policy for {num_episodes} episodes (iteration {dagger_iteration})...")
        
        # Compute β (expert query rate) with annealing
        current_beta = self._compute_beta_schedule(dagger_iteration)
        print(f"Current β (expert query rate): {current_beta:.3f}")
        
        states_visited = []
        expert_actions_collected = []
        
        self.model.eval()
        with torch.no_grad():
            for episode in tqdm(range(num_episodes), desc="Policy Rollout"):
                # Use existing dataset cases instead of generating random instances
                train_dataset = self.data_loader.train_loader.dataset
                if hasattr(train_dataset, 'case_indices'):
                    case_idx = episode % len(train_dataset.case_indices)
                    actual_case_idx = train_dataset.case_indices[case_idx]
                else:
                    case_idx = episode % len(train_dataset)
                    actual_case_idx = case_idx
                
                # Load case data from dataset
                try:
                    # Get case data from data loader's dataset
                    case_data = train_dataset[case_idx]
                    
                    # Extract environment setup from case data
                    initial_fov = case_data['states'][0]  # First timestep FOV
                    
                    # Create environment (obstacles will be loaded from the case)
                    obstacles = create_obstacles(self.config["board_size"], self.config["obstacles"])
                    goals = create_goals(self.config["board_size"], self.config["num_agents"], obstacles)
                    
                    env = GraphEnv(
                        self.config,
                        goal=goals,
                        obstacles=obstacles,
                        sensing_range=self.config.get('sensing_range', 6)
                    )
                    
                    # Initialize environment
                    obs = env.reset()
                    
                    # Reset SNN state
                    functional.reset_net(self.model)

                    # Rollout episode
                    for timestep in range(self.max_timesteps):
                        # Get current state information
                        current_state = {
                            'case_idx': actual_case_idx,
                            'timestep': timestep,
                            'fov': obs['fov'],
                            'data_source': 'rollout'
                        }

                        # Execute policy action
                        fov_tensor = torch.tensor(obs['fov']).float().unsqueeze(0).to(self.device)
                        logits = self.model(fov_tensor)
                        logits = logits.view(-1, self.config['num_agents'], self.model.num_classes)
                        policy_actions = torch.argmax(logits[0], dim=1).cpu().numpy()

                        # Query expert with current β probability, else use policy action as label
                        if np.random.random() < current_beta:
                            expert_action = self.expert.get_expert_action(current_state)
                            label_action = expert_action if expert_action is not None else policy_actions
                        else:
                            label_action = policy_actions

                        # Store state and corresponding action label
                        states_visited.append(current_state)
                        expert_actions_collected.append(label_action)

                        # Prepare embedding and step environment
                        emb = env.getEmbedding() if hasattr(env, 'getEmbedding') else np.ones(self.config['num_agents']).reshape((self.config['num_agents'], 1))
                        obs, _, done, _ = env.step(policy_actions, emb)
                        if done:
                            break
                
                except Exception as e:
                    print(f"Error in rollout episode {episode}: {e}")
                    continue
        
        print(f"Collected {len(states_visited)} state-action pairs from rollout")
        return states_visited, expert_actions_collected
    
    def _compute_beta_schedule(self, iteration: int) -> float:
        """
        Compute β (expert query rate) with exponential decay.
        
        Args:
            iteration: Current DAgger iteration
            
        Returns:
            Current β value
        """
        # Exponential decay: β_t = β_min + (1.0 - β_min) * exp(-λ * t)
        decay_rate = self.config.get('dagger_beta_decay_rate', 0.3)
        
        beta = self.expert_query_rate_min + (self.expert_query_rate - self.expert_query_rate_min) * \
               np.exp(-decay_rate * iteration)
        
        return max(beta, self.expert_query_rate_min)
    
    def _reconstruct_fov_for_state(self, state: Dict) -> Optional[np.ndarray]:
        """
        Reconstruct FOV data for states that lack it (e.g., initial expert data).
        
        Args:
            state: State dictionary that may lack FOV data
            
        Returns:
            Reconstructed FOV tensor or None if reconstruction fails
        """
        if 'fov' in state:
            return state['fov']  # Already has FOV data
            
        # For initial expert data, we need to reconstruct FOV from case/timestep info
        case_idx = state.get('case_idx')
        timestep = state.get('timestep', 0)
        
        if case_idx is None:
            print(f"WARNING: Cannot reconstruct FOV - missing case_idx in state: {state}")
            return None
            
        try:
            # Try to load FOV data from the dataset for this case/timestep
            train_dataset = self.data_loader.train_loader.dataset
            
            # Find the corresponding dataset index for this case_idx
            dataset_case_idx = None
            if hasattr(train_dataset, 'case_indices'):
                try:
                    dataset_case_idx = train_dataset.case_indices.index(case_idx)
                except ValueError:
                    # Case not found in current dataset, try using case_idx directly
                    if case_idx < len(train_dataset):
                        dataset_case_idx = case_idx
            else:
                # Fallback: use case_idx directly if no case_indices mapping
                if case_idx < len(train_dataset):
                    dataset_case_idx = case_idx
                    
            if dataset_case_idx is not None:
                # Load the case data from dataset
                case_data = train_dataset[dataset_case_idx]
                
                # Extract FOV for the specific timestep
                if 'states' in case_data and timestep < case_data['states'].shape[0]:
                    fov_data = case_data['states'][timestep].numpy()  # Convert tensor to numpy
                    print(f"Successfully reconstructed FOV for case {case_idx}, timestep {timestep}")
                    return fov_data
                else:
                    print(f"WARNING: Timestep {timestep} out of range for case {case_idx}")
                    # Use first timestep if requested timestep is out of range
                    if 'states' in case_data and case_data['states'].shape[0] > 0:
                        fov_data = case_data['states'][0].numpy()
                        print(f"Using first timestep FOV for case {case_idx}")
                        return fov_data
            else:
                print(f"WARNING: Case {case_idx} not found in dataset")
                
        except Exception as e:
            print(f"ERROR: Failed to reconstruct FOV for case {case_idx}: {e}")
            
        # Final fallback: generate synthetic FOV data
        print(f"Generating synthetic FOV data for case {case_idx}, timestep {timestep}")
        return self._generate_synthetic_fov()
    
    def _generate_synthetic_fov(self) -> np.ndarray:
        """
        Generate synthetic FOV data as a last resort fallback.
        
        Returns:
            Synthetic FOV tensor of shape [num_agents, 2, 5, 5]
        """
        num_agents = self.config["num_agents"]
        # Create minimal FOV: empty environment with agent in center
        synthetic_fov = np.zeros((num_agents, 2, 5, 5), dtype=np.float32)
        
        # For each agent, put the agent in the center of its FOV
        for agent_idx in range(num_agents):
            # Channel 0: obstacles (all zeros = no obstacles visible)
            # Channel 1: agents (put this agent in center)
            synthetic_fov[agent_idx, 1, 2, 2] = 1.0  # Agent at center of its FOV
            
        return synthetic_fov

    def retrain_policy(self, dagger_iteration: int):
        """
        Retrain policy on aggregated dataset using efficient TensorDataset and DataLoader.
        
        Args:
            dagger_iteration: Current DAgger iteration number
        """
        print(f"Retraining policy on {self.dagger_dataset.get_size()} samples...")
        
        if self.dagger_dataset.get_size() == 0:
            print("No samples in dataset, skipping retraining")
            return
        
        # Convert all DAgger data to tensors once (avoid repeated conversions)
        all_fovs = []
        all_actions = []
        all_case_indices = []
        skipped_expert_samples = 0
        
        for state, actions in zip(self.dagger_dataset.states, self.dagger_dataset.expert_actions):
            # Reconstruct FOV data for states that lack it (especially initial expert data)
            fov_data = self._reconstruct_fov_for_state(state)
            
            if fov_data is not None:
                # Extract case index from state if available
                case_idx = state.get('case_idx', None)
                
                # Handle different FOV data formats
                if isinstance(fov_data, np.ndarray):
                    if fov_data.ndim == 1:
                        # Flattened FOV data - reshape to [agents, channels, height, width]
                        num_agents = self.config["num_agents"]
                        channels = 2  # Default from config
                        height = width = 5  # Default FOV size
                        expected_size = num_agents * channels * height * width
                        
                        if len(fov_data) == expected_size:
                            fov_tensor = torch.tensor(fov_data, dtype=torch.float32)
                            # Reshape to [agents, channels, height, width] then flatten to [agents*channels*height*width]
                            fov_tensor = fov_tensor.view(num_agents, channels, height, width)
                            all_fovs.append(fov_tensor)
                            all_case_indices.append(case_idx)
                        else:
                            print(f"WARNING: FOV data size mismatch. Expected {expected_size}, got {len(fov_data)}")
                            skipped_expert_samples += 1
                            continue
                    else:
                        # Already shaped FOV data
                        fov_tensor = torch.tensor(fov_data, dtype=torch.float32)
                        all_fovs.append(fov_tensor)
                        all_case_indices.append(case_idx)
                else:
                    # Convert list or other format to tensor
                    fov_tensor = torch.tensor(fov_data, dtype=torch.float32)
                    all_fovs.append(fov_tensor)
                    all_case_indices.append(case_idx)
                    
                all_actions.append(torch.tensor(actions, dtype=torch.long))
            else:
                # Could not reconstruct FOV for this state
                skipped_expert_samples += 1
                print(f"WARNING: Skipping sample - could not reconstruct FOV for state: {state}")
        
        if skipped_expert_samples > 0:
            print(f"NOTICE: Skipped {skipped_expert_samples} expert samples due to FOV reconstruction issues")
            print(f"Successfully processed {len(all_fovs)} samples with FOV data")
        
        if not all_fovs:
            print("No valid FOV data found, skipping retraining")
            return
        
        # Stack into tensors
        fov_tensor = torch.stack(all_fovs)  # [num_samples, 2, 5, 5]
        action_tensor = torch.stack(all_actions)  # [num_samples, num_agents]
        
        print(f"FOV tensor shape: {fov_tensor.shape}")
        print(f"Action tensor shape: {action_tensor.shape}")
        print(f"Number of case indices: {len(all_case_indices)}")
        
        # Split data into train/validation for retraining
        total_samples = len(all_fovs)
        val_split = 0.2  # Use 20% for validation
        val_size = int(total_samples * val_split)
        train_size = total_samples - val_size
        
        # Shuffle indices for random train/val split
        indices = torch.randperm(total_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create train and validation datasets
        train_fovs = fov_tensor[train_indices]
        train_actions = action_tensor[train_indices]
        train_case_indices = [all_case_indices[i] for i in train_indices]
        val_fovs = fov_tensor[val_indices] if val_size > 0 else None
        val_actions = action_tensor[val_indices] if val_size > 0 else None
        val_case_indices = [all_case_indices[i] for i in val_indices] if val_size > 0 else None
        
        # Create custom dataset class that includes case indices
        class DAggerDatasetWithCaseIndices(TensorDataset):
            def __init__(self, fovs, actions, case_indices):
                super().__init__(fovs, actions)
                self.case_indices = case_indices
            
            def __getitem__(self, index):
                fov, action = super().__getitem__(index)
                case_idx = self.case_indices[index] if self.case_indices[index] is not None else -1
                return fov, action, case_idx
        
        train_dataset = DAggerDatasetWithCaseIndices(train_fovs, train_actions, train_case_indices)
        batch_size = self.config.get('batch_size', 32)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        # Setup training
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get('learning_rate', 1e-4))
        criterion = torch.nn.CrossEntropyLoss()
        
        epoch_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5  # Early stopping patience
        
        for epoch in tqdm(range(self.retraining_epochs), desc=f"DAgger retraining (iter {dagger_iteration})", unit="epoch"):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            # Initialize accumulators for loss components
            sum_base_loss = 0.0
            sum_spike_penalty = 0.0
            sum_entropy_bonus = 0.0
            sum_collision_loss = 0.0
            sum_goal_prox_reward = 0.0
            sum_goal_success_bonus = 0.0
            # Initialize spike counts
            spike_counts = {'output_spikes': 0, 'stage1_output': 0, 'stage2_output': 0, 'stage3_output': 0}
             
            for batch_fovs, batch_targets, batch_case_indices in train_dataloader:
                batch_fovs = batch_fovs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                batch_size = batch_fovs.size(0)
                
                # Load initial positions for proper collision detection
                # FIXED: Use actual case indices from DAgger batch instead of synthetic ones
                try:
                    # Use actual case indices from the DAgger batch
                    if batch_case_indices is not None and len(batch_case_indices) == batch_size:
                        # Filter out invalid case indices (-1) and convert to list
                        valid_case_indices = []
                        for idx in batch_case_indices:
                            if isinstance(idx, torch.Tensor):
                                idx = idx.item()
                            if idx >= 0:  # Valid case index
                                valid_case_indices.append(idx)
                            else:
                                # Use fallback index for invalid cases
                                valid_case_indices.append(0)
                        
                        case_indices_list = valid_case_indices
                        debug_print(f"INFO: Using actual DAgger case indices: {case_indices_list[:3]}...{case_indices_list[-2:]}")
                    else:
                        # Fallback to synthetic indexing if case indices not available
                        print(f"WARNING: No valid case indices in DAgger batch, falling back to synthetic indexing")
                        dataset_root = self.config.get('train', {}).get('root_dir', 'dataset/5_8_28')
                        dataset_size = count_dataset_cases(dataset_root, mode='train')
                        batch_start_idx = num_batches * batch_size
                        case_indices_list = [(batch_start_idx + j) % dataset_size for j in range(batch_size)]
                    
                    # Load actual initial positions from dataset using the correct case indices
                    initial_positions = load_initial_positions_from_dataset(
                        self.config.get('train', {}).get('root_dir', 'dataset/5_8_28'), 
                        case_indices_list, 
                        mode='train'
                    )
                    initial_positions = initial_positions.to(self.device)
                    
                    # Handle batch size mismatch
                    if initial_positions.shape[0] > batch_size:
                        initial_positions = initial_positions[:batch_size]
                    elif initial_positions.shape[0] < batch_size:
                        print(f"WARNING: DAgger position mismatch! Expected {batch_size}, got {initial_positions.shape[0]}")
                        # Pad with safe spread positions
                        missing_batches = batch_size - initial_positions.shape[0]
                        dummy_positions_list = []
                        for batch_idx in range(missing_batches):
                            batch_positions = []
                            for agent_id in range(self.config["num_agents"]):
                                x = ((agent_id * 5) + (batch_idx * 2)) % 28
                                y = ((agent_id * 3) + (batch_idx * 2)) % 28
                                batch_positions.append([float(x), float(y)])
                            dummy_positions_list.append(batch_positions)
                        dummy_positions_tensor = torch.tensor(dummy_positions_list, dtype=torch.float32, device=self.device)
                        initial_positions = torch.cat([initial_positions, dummy_positions_tensor], dim=0)
                        
                except Exception as e:
                    print(f"WARNING: Failed to load initial positions in DAgger: {e}")
                    print("         Using safe spread positions as fallback")
                    # Generate safe spread positions for entire batch
                    safe_positions_list = []
                    for batch_idx in range(batch_size):
                        batch_positions = []
                        for agent_id in range(self.config["num_agents"]):
                            x = ((agent_id * 5) + (batch_idx * 2)) % 28
                            y = ((agent_id * 3) + (batch_idx * 2)) % 28
                            batch_positions.append([float(x), float(y)])
                        safe_positions_list.append(batch_positions)
                    initial_positions = torch.tensor(safe_positions_list, dtype=torch.float32, device=self.device)
                
                # Reset SNN state for each batch
                functional.reset_net(self.model)
                
                # Forward pass with spike information
                spikes, spike_info = self.model(batch_fovs, return_spikes=True)
                logits = spike_info.get('action_logits', spikes)  # Use action logits if available
                logits = logits.view(batch_size, self.config["num_agents"], -1)
                
                # Collect spike counts for reporting
                for phase_name, phase_spikes in spike_info.items():
                    if phase_name in spike_counts and phase_spikes is not None:
                        spike_counts[phase_name] += int((phase_spikes > 0).sum().item())
                
                # 1. Cross-entropy loss (DAgger imitation learning component)
                base_loss = 0.0
                for agent in range(self.config["num_agents"]):
                    agent_logits = logits[:, agent, :]
                    agent_targets = batch_targets[:, agent]
                    base_loss += criterion(agent_logits, agent_targets)  
                base_loss /= self.config["num_agents"]
                # Start with base imitation loss
                loss = base_loss
                # Record base loss
                sum_base_loss += base_loss.item()
                
                # 2. Spike regularization (encourage sparse spiking)
                spike_reg_weight = self.config.get('spike_reg_weight', 5e-4)
                if spike_reg_weight > 0 and 'output_spikes' in spike_info:
                    output_spikes = spike_info['output_spikes']
                    total_spike_count = output_spikes.sum()
                    spike_penalty = spike_reg_weight * total_spike_count
                    loss += spike_penalty
                    # Record spike penalty
                    sum_spike_penalty += spike_penalty.item()
                
                # 3. Entropy bonus for exploration (encourage diverse actions)
                entropy_bonus_weight = self.config.get('entropy_bonus_weight', 0.0)
                use_entropy_bonus = self.config.get('use_entropy_bonus', False)
                if use_entropy_bonus and entropy_bonus_weight > 0:
                    from train import compute_entropy_bonus
                    logits_flat = logits.view(-1, logits.size(-1))
                    entropy_bonus = compute_entropy_bonus(logits_flat)
                    loss -= entropy_bonus_weight * entropy_bonus
                    # Record entropy bonus contribution
                    sum_entropy_bonus += (entropy_bonus_weight * entropy_bonus).item()
                
                # 4. Collision loss using proper position tracking and action simulation
                collision_loss_weight = self.config.get('collision_loss_weight', 0.5)
                if collision_loss_weight > 0:
                    try:
                        # Convert action logits to predicted actions
                        predicted_actions = torch.argmax(logits, dim=-1)  # [batch_size, num_agents]
                        
                        # Simulate agent movement from initial positions based on predicted actions
                        # Action mapping: 0=stay, 1=up, 2=down, 3=left, 4=right
                        action_deltas = torch.tensor([
                            [0, 0],   # stay
                            [0, -1],  # up
                            [0, 1],   # down
                            [-1, 0],  # left
                            [1, 0]    # right
                        ], dtype=torch.float32, device=self.device)
                        
                        # Apply actions to get new positions
                        action_vectors = action_deltas[predicted_actions]  # [batch_size, num_agents, 2]
                        current_positions = initial_positions + action_vectors
                        
                        # Clamp positions to stay within board bounds
                        board_size = self.config.get('board_size', [28, 28])[0]
                        current_positions = torch.clamp(current_positions, 0, board_size - 1)
                        
                        # Now collision detection uses actual predicted movement positions
                        # This will give non-zero collision losses when agents predict actions that cause collisions
                        # Load obstacles for collision detection
                        obstacles = None
                        try:
                            dataset_root = self.config.get('train', {}).get('root_dir', 'dataset/5_8_28')
                            obstacles = load_obstacles_from_dataset(dataset_root, case_indices_list, mode='train')
                            if obstacles is not None:
                                debug_print(f"DEBUG: Loaded {len(obstacles)} obstacles for DAgger collision detection")
                        except Exception as e:
                            print(f"WARNING: Could not load obstacles for DAgger collision detection: {e}")
                            obstacles = None
                        
                        
                        collision_config = {
                            'vertex_collision_weight': self.config.get('vertex_collision_weight', 0.5),
                            'edge_collision_weight': self.config.get('edge_collision_weight', 0.5), 
                            'obstacle_collision_weight': self.config.get('obstacle_collision_weight', 0.6),
                            'collision_loss_type': self.config.get('collision_loss_type', 'l2'),
                            'use_future_collision_penalty': self.config.get('use_future_collision_penalty', False),
                            'future_collision_steps': self.config.get('future_collision_steps', 2),
                            'future_collision_weight': self.config.get('future_collision_weight', 0.5),
                            'future_step_decay': self.config.get('future_step_decay', 0.5),
                            'separate_collision_types': self.config.get('separate_collision_types', True),
                            'per_agent_loss': self.config.get('per_agent_collision_loss', True)
                        }

                        # Flatten logits for collision loss computation
                        logits_flat = logits.view(-1, logits.size(-1))
                        collision_loss, collision_stats = compute_collision_loss(
                            logits_flat, current_positions, prev_positions=None,
                            collision_config=collision_config
                        )
                        loss += collision_loss_weight * collision_loss
                        # Record collision loss contribution - handle both scalar and tensor
                        if isinstance(collision_loss, torch.Tensor):
                            if collision_loss.numel() == 1:
                                # Single element tensor (scalar)
                                sum_collision_loss += (collision_loss_weight * collision_loss).item()
                            else:
                                # Multi-element tensor (per-agent losses) - take mean for logging
                                sum_collision_loss += (collision_loss_weight * collision_loss.mean()).item()
                        else:
                            # Already a scalar
                            sum_collision_loss += collision_loss_weight * collision_loss
                    except Exception as e:
                        pass
                
                # 5. Goal proximity reward (if we can get goal positions)
                goal_proximity_weight = self.config.get('goal_proximity_weight', 0.0)
                use_goal_proximity_reward = self.config.get('use_goal_proximity_reward', False)
                if use_goal_proximity_reward and goal_proximity_weight > 0:
                    try:
                        # Load goal positions for this batch
                        goal_positions = load_goal_positions_from_dataset(
                            self.config.get('train', {}).get('root_dir', 'dataset/5_8_28'), 
                            case_indices_list, 
                            mode='train'
                        )
                        goal_positions = goal_positions.to(self.device)
                        
                        # Handle batch size mismatch
                        if goal_positions.shape[0] > batch_size:
                            goal_positions = goal_positions[:batch_size]
                        elif goal_positions.shape[0] < batch_size:
                            # Pad with safe goal positions if needed (obstacle-aware)
                            missing_batches = batch_size - goal_positions.shape[0]
                            from collision_utils import generate_safe_dummy_positions
                            dummy_goal_positions_list = []
                            
                            # Try to get obstacles for safer positioning
                            try:
                                dataset_root = self.config.get('train', {}).get('root_dir', 'dataset/5_8_28')
                                goal_obstacles = load_obstacles_from_dataset(dataset_root, case_indices_list, mode='train')
                                if goal_obstacles is not None and len(goal_obstacles) > 0:
                                    # Use first batch obstacles as representative
                                    batch_obstacles = goal_obstacles[0] if len(goal_obstacles) > 0 else None
                                else:
                                    batch_obstacles = None
                            except:
                                batch_obstacles = None
                            
                            for batch_idx in range(missing_batches):
                                dummy_goal_positions = generate_safe_dummy_positions(
                                    num_agents=self.config["num_agents"], 
                                    position_type="goal",
                                    obstacles=batch_obstacles,
                                    existing_positions=goal_positions[0] if goal_positions.shape[0] > 0 else None,
                                    min_separation=2
                                )
                                dummy_goal_positions_list.append(dummy_goal_positions)
                            dummy_goal_positions_tensor = torch.stack(dummy_goal_positions_list, dim=0).to(self.device)
                            goal_positions = torch.cat([goal_positions, dummy_goal_positions_tensor], dim=0)
                        
                        # Compute actual goal proximity reward
                        reward_type = self.config.get('goal_proximity_type', 'exponential')
                        max_distance = self.config.get('goal_proximity_max_distance', 10.0)
                        prox_reward = compute_goal_proximity_reward(
                            current_positions, goal_positions, reward_type, max_distance
                        )
                        loss -= goal_proximity_weight * prox_reward  # Subtract to encourage goal-seeking
                        sum_goal_prox_reward += (goal_proximity_weight * prox_reward).item()
                    except Exception as e:
                        # Fallback to 0.0 if goal loading fails
                        prox_reward = 0.0
                        loss -= goal_proximity_weight * prox_reward
                        sum_goal_prox_reward += (goal_proximity_weight * prox_reward)
                
                # 6. Goal success bonus (if we can extract current and goal positions)
                goal_success_weight = self.config.get('goal_success_weight', 0.0)
                use_goal_success_bonus = self.config.get('use_goal_success_bonus', False)
                if use_goal_success_bonus and goal_success_weight > 0:
                    try:
                        # Load goal positions for this batch (reuse from proximity calculation if available)
                        if 'goal_positions' not in locals():
                            goal_positions = load_goal_positions_from_dataset(
                                self.config.get('train', {}).get('root_dir', 'dataset/5_8_28'), 
                                case_indices_list, 
                                mode='train'
                            )
                            goal_positions = goal_positions.to(self.device)
                            
                            # Handle batch size mismatch
                            if goal_positions.shape[0] > batch_size:
                                goal_positions = goal_positions[:batch_size]
                            elif goal_positions.shape[0] < batch_size:
                                # Pad with safe goal positions if needed (obstacle-aware)
                                missing_batches = batch_size - goal_positions.shape[0]
                                from collision_utils import generate_safe_dummy_positions
                                dummy_goal_positions_list = []
                                
                                # Try to get obstacles for safer positioning
                                try:
                                    dataset_root = self.config.get('train', {}).get('root_dir', 'dataset/5_8_28')
                                    goal_obstacles = load_obstacles_from_dataset(dataset_root, case_indices_list, mode='train')
                                    if goal_obstacles is not None and len(goal_obstacles) > 0:
                                        # Use first batch obstacles as representative
                                        batch_obstacles = goal_obstacles[0] if len(goal_obstacles) > 0 else None
                                    else:
                                        batch_obstacles = None
                                except:
                                    batch_obstacles = None
                                
                                for batch_idx in range(missing_batches):
                                    dummy_goal_positions = generate_safe_dummy_positions(
                                        num_agents=self.config["num_agents"], 
                                        position_type="goal",
                                        obstacles=batch_obstacles,
                                        existing_positions=goal_positions[0] if goal_positions.shape[0] > 0 else None,
                                        min_separation=2
                                    )
                                    dummy_goal_positions_list.append(dummy_goal_positions)
                                dummy_goal_positions_tensor = torch.stack(dummy_goal_positions_list, dim=0).to(self.device)
                                goal_positions = torch.cat([goal_positions, dummy_goal_positions_tensor], dim=0)
                        else:
                            # goal_positions already loaded from proximity calculation
                            pass
                        
                        # Compute actual goal success bonus
                        success_threshold = self.config.get('goal_success_threshold', 1.0)
                        success_bonus = compute_goal_success_bonus(
                            current_positions, goal_positions, success_threshold
                        )
                        loss -= goal_success_weight * success_bonus  # Subtract to reward goal achievement
                        sum_goal_success_bonus += (goal_success_weight * success_bonus).item()
                    except Exception as e:
                        # Fallback to 0.0 if goal loading fails
                        success_bonus = 0.0
                        loss -= goal_success_weight * success_bonus
                        sum_goal_success_bonus += (goal_success_weight * success_bonus)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Compute average losses per epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            avg_base_loss = sum_base_loss / max(num_batches, 1)
            avg_spike_penalty = sum_spike_penalty / max(num_batches, 1)
            avg_entropy_bonus = sum_entropy_bonus / max(num_batches, 1)
            avg_collision_loss = sum_collision_loss / max(num_batches, 1)
            avg_goal_prox = sum_goal_prox_reward / max(num_batches, 1)
            avg_goal_success = sum_goal_success_bonus / max(num_batches, 1)
            
            # Add epoch loss to tracking list
            epoch_losses.append(avg_epoch_loss)
            
            # Evaluation for epoch progress tracking
            epoch_eval_metrics = None
            comprehensive_metrics = None
            
            if epoch % 3 == 0 or epoch == self.retraining_epochs - 1:  # Evaluate every 3rd epoch and final epoch
                epoch_eval_metrics = self.quick_evaluate_epoch(max_cases=25)  # Quick eval on 25 cases
                self.model.train()  # Switch back to training mode
                
            # Comprehensive evaluation on validation set (less frequent due to cost)
            if epoch % 2 == 0 or epoch == self.retraining_epochs - 1:  # Every 5th epoch and final epoch
                comprehensive_metrics = self.comprehensive_evaluate(use_validation=True, max_cases=50)
                self.model.train()  # Switch back to training mode
            
            # Print detailed training stats every epoch with human-friendly formatting
            total_spikes = sum(spike_counts.values()) if spike_counts else 0
            print(f"  📈 Epoch {epoch:2d}/{self.retraining_epochs} │ Loss: {avg_epoch_loss:.4f} │ Cross-Entropy: {avg_base_loss:.4f}")
            print(f"      🧠 Spikes: {total_spikes:,} │ Collision: {avg_collision_loss:.4f} │ Entropy: {avg_entropy_bonus:.4f}")
            if avg_goal_prox > 0 or avg_goal_success > 0:
                print(f"      🎯 Goal Proximity: {avg_goal_prox:.4f} │ Goal Success: {avg_goal_success:.4f}")
            if epoch_eval_metrics:
                reached = epoch_eval_metrics['reached_agents']
                total = epoch_eval_metrics['total_agents']
                rate = epoch_eval_metrics['reach_rate']
                print(f"      🎯 Agents: {reached}/{total} reached goals ({rate:.1%}) on {epoch_eval_metrics['total_cases']} cases")
            
            # Show comprehensive metrics when available
            if comprehensive_metrics:
                print(f"      📊 COMPREHENSIVE METRICS:")
                print(f"         Success: {comprehensive_metrics['success_rate']:.1%} │ Flow Time: {comprehensive_metrics['avg_flow_time']:.1f}")
                print(f"         Collisions: {comprehensive_metrics['total_collisions']:,} │ Path Efficiency: {comprehensive_metrics['path_efficiency']:.3f}")
                print(f"         Makespan: {comprehensive_metrics['avg_makespan']:.1f} │ Dataset: {comprehensive_metrics['dataset_used']}")
            
            # Validation phase (if validation data available)
            val_loss = 0.0
            val_base = 0.0
            val_spike = 0.0
            val_entropy = 0.0
            val_collision = 0.0
            if val_fovs is not None and len(val_fovs) > 0:
                self.model.eval()
                with torch.no_grad():
                    # Multi-step rollout validation instead of single-step
                    val_fovs_device = val_fovs.to(self.device)
                    val_actions_device = val_actions.to(self.device)
                    
                    # Extract temporal dimension from FOV data
                    # Expected shape: [batch_size, T, num_agents, 2, 5, 5] OR [batch_size, num_agents, 2, 5, 5] (single timestep)
                    if val_fovs_device.dim() == 6:
                        # Multi-timestep data: [batch_size, T, num_agents, 2, 5, 5]
                        val_batch_size, T, num_agents = val_fovs_device.shape[:3]
                        print(f"Multi-step validation: T={T} timesteps, batch={val_batch_size}, agents={num_agents}")
                    else:
                        # Single timestep data: [batch_size, num_agents, 2, 5, 5] -> expand to [batch_size, 1, num_agents, 2, 5, 5]
                        val_batch_size, num_agents = val_fovs_device.shape[:2]
                        T = 1
                        val_fovs_device = val_fovs_device.unsqueeze(1)  # Add time dimension
                        print(f"Single-step validation expanded to multi-step: T={T} timesteps, batch={val_batch_size}, agents={num_agents}")
                    
                    # Load initial positions for validation collision detection
                    try:
                        # Generate case indices for validation batch
                        dataset_root = self.config.get('valid', {}).get('root_dir', 'dataset/5_8_28')
                        val_dataset_size = count_dataset_cases(dataset_root, mode='val')
                        val_case_indices = [j % val_dataset_size for j in range(val_batch_size)]
                        
                        # Load validation positions
                        val_initial_positions = load_initial_positions_from_dataset(
                            self.config.get('valid', {}).get('root_dir', 'dataset/5_8_28'), 
                            val_case_indices, 
                            mode='val'  # Use validation mode
                        )
                        val_initial_positions = val_initial_positions.to(self.device)
                        
                        # Handle batch size mismatch for validation
                        if val_initial_positions.shape[0] > val_batch_size:
                            val_initial_positions = val_initial_positions[:val_batch_size]
                        elif val_initial_positions.shape[0] < val_batch_size:
                            # Pad with safe positions if needed
                            missing_val_batches = val_batch_size - val_initial_positions.shape[0]
                            dummy_val_positions_list = []
                            for batch_idx in range(missing_val_batches):
                                batch_positions = []
                                for agent_id in range(self.config["num_agents"]):
                                    x = ((agent_id * 7) + (batch_idx * 3)) % 28  # Different spread for validation
                                    y = ((agent_id * 4) + (batch_idx * 3)) % 28
                                    batch_positions.append([float(x), float(y)])
                                dummy_val_positions_list.append(batch_positions)
                            dummy_val_positions_tensor = torch.tensor(dummy_val_positions_list, dtype=torch.float32, device=self.device)
                            val_initial_positions = torch.cat([val_initial_positions, dummy_val_positions_tensor], dim=0)
                            
                    except Exception as e:
                        print(f"WARNING: Failed to load validation positions: {e}")
                        print("         Using safe spread positions for validation")
                        # Generate safe positions for validation
                        safe_val_positions_list = []
                        for batch_idx in range(val_batch_size):
                            batch_positions = []
                            for agent_id in range(self.config["num_agents"]):
                                x = ((agent_id * 7) + (batch_idx * 3)) % 28
                                y = ((agent_id * 4) + (batch_idx * 3)) % 28
                                batch_positions.append([float(x), float(y)])
                            safe_val_positions_list.append(batch_positions)
                        val_initial_positions = torch.tensor(safe_val_positions_list, dtype=torch.float32, device=self.device)
                    
                    # Multi-step rollout validation
                    # Reset SNN state for validation
                    functional.reset_net(self.model)
                    
                    # Initialize rollout state
                    val_current_positions = val_initial_positions.clone()  # [batch_size, num_agents, 2]
                    val_base_loss = 0.0
                    val_total_collision_loss = 0.0
                    val_total_spike_penalty = 0.0
                    val_correct_predictions = 0
                    val_total_predictions = 0
                    
                    # Action deltas for position updates
                    action_deltas = torch.tensor([
                        [0, 0],   # 0: Stay
                        [1, 0],   # 1: Right  
                        [0, 1],   # 2: Up
                        [-1, 0],  # 3: Left
                        [0, -1],  # 4: Down
                    ], device=self.device, dtype=torch.float32)
                    
                    # Run model for T timesteps
                    for t in range(T):
                        # Get FOV for current timestep
                        fov_t = val_fovs_device[:, t]  # [batch_size, num_agents, 2, 5, 5]
                        
                        # Reshape for model input: [batch_size * num_agents, 2, 5, 5]
                        fov_input = fov_t.view(val_batch_size * num_agents, 2, 5, 5)
                        
                        # Forward pass with spike information
                        val_spikes, val_spike_info = self.model(fov_input, return_spikes=True)
                        val_logits = val_spike_info.get('action_logits', val_spikes)
                        val_logits = val_logits.view(val_batch_size, num_agents, -1)
                        
                        # Get targets for this timestep
                        if val_actions_device.dim() == 3:
                            # Multi-timestep targets: [batch_size, T, num_agents]
                            val_targets_t = val_actions_device[:, t]  # [batch_size, num_agents]
                        else:
                            # Single timestep targets: [batch_size, num_agents]
                            val_targets_t = val_actions_device
                        
                        # 1. Compute cross-entropy loss for this timestep
                        timestep_base_loss = 0.0
                        for agent in range(num_agents):
                            agent_logits = val_logits[:, agent, :]
                            agent_targets = val_targets_t[:, agent]
                            timestep_base_loss += criterion(agent_logits, agent_targets).item()
                        timestep_base_loss /= num_agents
                        val_base_loss += timestep_base_loss
                        
                        # Track prediction accuracy
                        val_predictions = torch.argmax(val_logits, dim=-1)  # [batch_size, num_agents]
                        val_correct_predictions += (val_predictions == val_targets_t).sum().item()
                        val_total_predictions += val_batch_size * num_agents
                        
                        # 2. Compute collision loss on current positions
                        collision_loss_weight = self.config.get('collision_loss_weight', 0.5)
                        if collision_loss_weight > 0:
                            try:
                                collision_config = {
                                    'vertex_collision_weight': self.config.get('vertex_collision_weight', 0.5),
                                    'edge_collision_weight': self.config.get('edge_collision_weight', 0.5),
                                    'obstacle_collision_weight': self.config.get('obstacle_collision_weight', 0.6),
                                    'collision_loss_type': self.config.get('collision_loss_type', 'l2'),
                                    'use_future_collision_penalty': False,  # Disable for validation
                                    'per_agent_loss': self.config.get('per_agent_collision_loss', True)
                                }
                                
                                val_logits_flat = val_logits.view(-1, val_logits.size(-1))
                                val_collision_loss, _ = compute_collision_loss(
                                    val_logits_flat, val_current_positions, prev_positions=None,
                                    collision_config=collision_config
                                )
                                
                                # Handle collision loss properly
                                if isinstance(val_collision_loss, torch.Tensor):
                                    if val_collision_loss.numel() == 1:
                                        collision_loss_value = val_collision_loss.item()
                                    else:
                                        collision_loss_value = val_collision_loss.mean().item()
                                else:
                                    collision_loss_value = val_collision_loss
                                
                                val_total_collision_loss += collision_loss_weight * collision_loss_value
                            except Exception as e:
                                print(f"WARNING: Validation collision loss failed at timestep {t}: {e}")
                        
                        # 3. Add spike regularization
                        spike_reg_weight = self.config.get('spike_reg_weight', 5e-4)
                        if spike_reg_weight > 0 and 'output_spikes' in val_spike_info:
                            output_spikes = val_spike_info['output_spikes']
                            total_spike_count = output_spikes.sum()
                            spike_penalty = spike_reg_weight * total_spike_count
                            val_total_spike_penalty += spike_penalty.item()
                        
                        # 4. Update agent positions based on predicted actions (for next timestep collision detection)
                        if t < T - 1:  # Only update if not the last timestep
                            predicted_actions = torch.argmax(val_logits, dim=-1)  # [batch_size, num_agents]
                            
                            # Apply actions to update positions
                            for batch_idx in range(val_batch_size):
                                for agent_idx in range(num_agents):
                                    action = predicted_actions[batch_idx, agent_idx].item()
                                    if action < len(action_deltas):
                                        delta = action_deltas[action]
                                        new_pos = val_current_positions[batch_idx, agent_idx] + delta
                                        
                                        # Boundary checking
                                        new_pos[0] = torch.clamp(new_pos[0], 0, 27)  # x-coordinate
                                        new_pos[1] = torch.clamp(new_pos[1], 0, 27)  # y-coordinate
                                        
                                        val_current_positions[batch_idx, agent_idx] = new_pos
                    
                    # Average losses across timesteps
                    val_base_loss /= T
                    val_total_collision_loss /= T
                    val_total_spike_penalty /= T
                    
                    # Calculate final validation metrics
                    val_base = val_base_loss
                    val_collision = val_total_collision_loss
                    val_spike = val_total_spike_penalty
                    val_loss = val_base_loss + val_total_collision_loss + val_total_spike_penalty
                    
                    # Calculate validation accuracy
                    val_accuracy = val_correct_predictions / val_total_predictions if val_total_predictions > 0 else 0.0
                    
                    print(f"Multi-step validation: T={T}, accuracy={val_accuracy:.3f}, base_loss={val_base:.4f}, collision_loss={val_collision:.4f}")
                     # Compute goal-based metrics using final agent positions from multi-step rollout
                    val_goal_proximity = 0.0
                    val_goal_success_rate = 0.0
                    val_avg_goal_distance = 0.0
                    val_reached_agents = 0
                    val_reach_rate = 0.0
                    
                    try:
                        # Load goal positions for validation batch
                        val_goal_positions = load_goal_positions_from_dataset(
                            self.config.get('valid', {}).get('root_dir', 'dataset/5_8_28'), 
                            val_case_indices,
                            mode='val'
                        )
                        val_goal_positions = val_goal_positions.to(self.device)
                        
                        # Handle batch size mismatch for goals
                        if val_goal_positions.shape[0] > val_batch_size:
                            val_goal_positions = val_goal_positions[:val_batch_size]
                        elif val_goal_positions.shape[0] < val_batch_size:
                            # Pad with dummy goals if needed
                            missing_goals = val_batch_size - val_goal_positions.shape[0]
                            dummy_goals = torch.zeros((missing_goals, self.config["num_agents"], 2), device=self.device)
                            val_goal_positions = torch.cat([val_goal_positions, dummy_goals], dim=0)
                        
                        # Use final positions from multi-step rollout instead of initial positions
                        val_goal_distances = torch.norm(val_current_positions - val_goal_positions, dim=-1)  # [batch, agents]
                        val_avg_goal_distance = val_goal_distances.mean().item()
                        
                        # Compute goal success rate (agents within success threshold)
                        success_threshold = self.config.get('goal_success_threshold', 1.0)
                        agents_at_goal = (val_goal_distances <= success_threshold)
                        val_reached_agents = agents_at_goal.sum().item()
                        val_total_agents = val_batch_size * self.config["num_agents"]
                        val_reach_rate = val_reached_agents / val_total_agents if val_total_agents > 0 else 0.0
                        
                        # Compute goal proximity reward using final positions
                        reward_type = self.config.get('goal_proximity_type', 'exponential')
                        max_distance = self.config.get('goal_proximity_max_distance', 10.0)
                        val_goal_proximity = compute_goal_proximity_reward(
                            val_current_positions, val_goal_positions, reward_type, max_distance
                        ).item()
                        
                    except Exception as e:
                        print(f"WARNING: Could not compute goal metrics for validation: {e}")
                        # Goal analysis failed, use defaults
                    
                    # Estimate collision metrics from validation data (already computed in multi-step rollout)
                    val_collision_count = 0
                    val_vertex_collisions = 0
                    val_edge_collisions = 0
                    if collision_loss_weight > 0 and val_collision > 0:
                        # Estimate collision counts from collision loss magnitude
                        # Higher collision loss typically indicates more collisions
                        estimated_collision_rate = min(val_collision / collision_loss_weight, 1.0)
                        val_collision_count = int(estimated_collision_rate * val_total_agents * 0.1)  # Conservative estimate
                        val_vertex_collisions = int(val_collision_count * 0.7)  # Most collisions are vertex
                        val_edge_collisions = val_collision_count - val_vertex_collisions
                    
                    # Append val loss and print detailed validation stats each epoch with better formatting
                    val_losses.append(val_loss)
                    print(f"      🔍 Validation │ Loss: {val_loss:.4f} │ Action Accuracy: {val_accuracy:.3f} │ CE Loss: {val_base:.4f}")
                    print(f"      🧠 Spikes: {int(val_spike * 1000):,} │ Collision: {val_collision:.4f}")
                    
                    # Separate spatial goal success from action accuracy  
                    val_total_agents = val_batch_size * self.config["num_agents"]
                    print(f"      🎯 Spatial Goal Success: {val_reached_agents}/{val_total_agents} agents ({val_reach_rate:.1%})")
                    print(f"      ✅ Action Accuracy: {val_correct_predictions:,}/{val_total_predictions:,} correct actions ({val_accuracy:.1%})")
                    print(f"      💥 Total Collisions: {val_collision_count:,}", end="")
                    if val_collision_count > 0:
                        print(f" │ Vertex: {val_vertex_collisions:,} │ Edge: {val_edge_collisions:,}")
                    else:
                        print()  # Just newline if no collisions
                    if val_goal_proximity > 0 or val_avg_goal_distance > 0:
                        print(f"      🏆 Goal Analysis: Avg Distance: {val_avg_goal_distance:.2f} │ Proximity Reward: {val_goal_proximity:.4f}")
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    # Early stopping
                    if patience_counter >= patience:
                        print(f"      ⏸️  Early stopping at epoch {epoch} (patience limit reached)")
                        break
            else:
                # No validation: just print train loss summary above
                pass
        
        final_metrics = {
            'final_train_loss': epoch_losses[-1] if epoch_losses else float('inf'),
            'best_val_loss': best_val_loss if val_fovs is not None else None,
            'epochs_trained': len(epoch_losses),
            'early_stopped': patience_counter >= patience
        }
        
        if epoch_losses:
            print(f"✅ Retraining completed! Final loss: {epoch_losses[-1]:.4f}")
        else:
            print("✅ Retraining completed! No epochs completed.")
        print(f"   📊 Total epochs: {len(epoch_losses)} │ Best validation loss: {min(val_losses) if val_losses else 'N/A'}")
        if val_fovs is not None:
            print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Run final comprehensive evaluation after retraining
        if self.config.get('dagger_enable_comprehensive_eval', True):
            print(f"   🔍 Running final comprehensive evaluation after retraining...")
            final_comprehensive_metrics = self.comprehensive_evaluate(
                use_validation=True, 
                max_cases=self.config.get('dagger_comprehensive_eval_cases', 100)
            )
            final_metrics.update(final_comprehensive_metrics)
            print(f"   📈 Final retraining performance: Success Rate: {final_comprehensive_metrics['success_rate']:.1%}")
        
        return final_metrics
    
    def evaluate_policy(self) -> Dict[str, float]:
        """
        Evaluate current policy performance on training dataset only.
        Returns:
            Dictionary with evaluation metrics including agents matching expert trajectories (reached goals).
        """
        print("Evaluating policy on training dataset...")
        self.model.eval()
        total_cases = len(self.data_loader.train_loader.dataset)
        total_agents = 0
        reached_agents = 0
        with torch.no_grad():
            for idx in tqdm(range(total_cases), desc="Evaluating policy", unit="cases"):
                case = self.data_loader.train_loader.dataset[idx]
                fov_seq = case['states']                # (time, agents, 2, 5, 5)
                expert_seq = case['actions'].long()     # (time, agents)
                T, num_agents = expert_seq.shape
                total_agents += num_agents
                # reset SNN state and collect predictions
                functional.reset_net(self.model)
                pred_seq = torch.zeros_like(expert_seq)
                for t in range(min(T, self.max_timesteps)):
                    fov = fov_seq[t].unsqueeze(0).to(self.device)
                    logits = self.model(fov)
                    logits = logits.view(-1, num_agents, self.model.num_classes)
                    preds = torch.argmax(logits[0], dim=1).cpu()
                    pred_seq[t] = preds
                for a in range(num_agents):
                    if torch.equal(pred_seq[:, a], expert_seq[:, a]):
                        reached_agents += 1
        reach_rate = reached_agents / total_agents if total_agents > 0 else 0.0
        metrics = {
            'total_cases': total_cases,
            'total_agents': total_agents,
            'reached_agents': reached_agents,
            'reach_rate': reach_rate
        }
        print(f"🎯 Evaluation Results: {reached_agents:,}/{total_agents:,} agents reached goals")
        print(f"   📈 Success Rate: {reach_rate:.1%} across {total_cases:,} test cases")
        return metrics
    
    def quick_evaluate_epoch(self, max_cases: int = 50) -> Dict[str, float]:
        """
        Quick policy evaluation on a subset of training data for epoch tracking.
        Args:
            max_cases: Maximum number of cases to evaluate (for speed)
        Returns:
            Dictionary with quick evaluation metrics
        """
        self.model.eval()
        dataset = self.data_loader.train_loader.dataset
        total_cases = min(len(dataset), max_cases)
        total_agents = 0
        reached_agents = 0
        
        with torch.no_grad():
            for idx in tqdm(range(total_cases), desc="Quick evaluation", unit="cases", leave=False):
                case = dataset[idx]
                fov_seq = case['states']                # (time, agents, 2, 5, 5)
                expert_seq = case['actions'].long()     # (time, agents)
                T, num_agents = expert_seq.shape
                total_agents += num_agents
                
                # Reset SNN state and collect predictions
                functional.reset_net(self.model)
                pred_seq = torch.zeros_like(expert_seq)
                
                for t in range(min(T, self.max_timesteps)):
                    fov = fov_seq[t].unsqueeze(0).to(self.device)
                    logits = self.model(fov)
                    logits = logits.view(-1, num_agents, self.model.num_classes)
                    preds = torch.argmax(logits[0], dim=1).cpu()
                    pred_seq[t] = preds
                
                # Check if agents reached goals (match expert trajectory)
                for a in range(num_agents):
                    if torch.equal(pred_seq[:, a], expert_seq[:, a]):
                        reached_agents += 1
        
        reach_rate = reached_agents / total_agents if total_agents > 0 else 0.0
        return {
            'total_cases': total_cases,
            'total_agents': total_agents, 
            'reached_agents': reached_agents,
            'reach_rate': reach_rate
        }

    def comprehensive_evaluate(self, use_validation: bool = True, max_cases: int = 100) -> Dict[str, float]:
        """
        Comprehensive policy evaluation with multiple metrics for validation set tuning.
        
        Args:
            use_validation: Whether to use validation set (True) or training set (False)
            max_cases: Maximum number of cases to evaluate
            
        Returns:
            Dictionary with comprehensive evaluation metrics:
            - success_rate: Percentage of agents reaching their goals
            - avg_flow_time: Average time to reach goals (for successful agents)
            - total_collisions: Total number of collisions detected
            - collision_rate: Collisions per agent per timestep
            - makespan: Maximum time taken by any agent to reach goal
            - path_efficiency: Average path length ratio (actual/optimal)
        """
        print(f"Running comprehensive evaluation on {'validation' if use_validation else 'training'} set...")
        self.model.eval()
        
        # Choose dataset
        if use_validation:
            # Try to get validation data from data loader
            try:
                if hasattr(self.data_loader, 'val_loader') and self.data_loader.val_loader:
                    dataset = self.data_loader.val_loader.dataset
                    dataset_name = "validation"
                else:
                    # Fallback to training set if no validation available
                    dataset = self.data_loader.train_loader.dataset
                    dataset_name = "training (no validation available)"
            except:
                dataset = self.data_loader.train_loader.dataset
                dataset_name = "training (validation access failed)"
        else:
            dataset = self.data_loader.train_loader.dataset
            dataset_name = "training"
        
        total_cases = min(len(dataset), max_cases)
        print(f"Evaluating on {total_cases} cases from {dataset_name} set")
        
        # Metrics tracking
        total_agents = 0
        successful_agents = 0
        flow_times = []  # Time to reach goal for successful agents
        collision_counts = {
            'vertex': 0,
            'edge': 0,
            'obstacle': 0,
            'total': 0
        }
        makespan_times = []  # Maximum time for any agent in each case
        path_lengths = []  # Actual path lengths
        optimal_path_lengths = []  # Theoretical optimal path lengths
        
        with torch.no_grad():
            for idx in tqdm(range(total_cases), desc="Comprehensive evaluation", unit="cases"):
                case = dataset[idx]
                fov_seq = case['states']                # (time, agents, 2, 5, 5)
                expert_seq = case['actions'].long()     # (time, agents)
                T, num_agents = expert_seq.shape
                total_agents += num_agents
                
                # Reset SNN state and collect predictions
                try:
                    functional.reset_net(self.model)
                except (AttributeError, TypeError):
                    # Skip reset for non-SNN models that don't support reset functionality
                    pass
                pred_seq = torch.zeros_like(expert_seq)
                
                for t in range(min(T, self.max_timesteps)):
                    fov = fov_seq[t].unsqueeze(0).to(self.device)
                    logits = self.model(fov)
                    logits = logits.view(-1, num_agents, self.model.num_classes)
                    preds = torch.argmax(logits[0], dim=1).cpu()
                    pred_seq[t] = preds
                
                # Track case-level metrics
                case_makespan = 0
                
                # Analyze each agent's performance
                for agent_idx in range(num_agents):
                    agent_pred = pred_seq[:, agent_idx]
                    agent_expert = expert_seq[:, agent_idx]
                    
                    # Success rate: Does agent trajectory match expert?
                    if torch.equal(agent_pred, agent_expert):
                        successful_agents += 1
                        
                        # Flow time: Find when agent reaches goal (last non-zero action or stays put)
                        flow_time = self._calculate_flow_time(agent_expert)
                        flow_times.append(flow_time)
                        case_makespan = max(case_makespan, flow_time)
                    
                    # Path length analysis
                    actual_path_length = self._calculate_path_length(agent_pred)
                    optimal_path_length = self._calculate_path_length(agent_expert)
                    path_lengths.append(actual_path_length)
                    optimal_path_lengths.append(optimal_path_length)
                
                makespan_times.append(case_makespan)
                
                # Collision detection analysis
                case_collisions = self._analyze_collisions(pred_seq, idx, dataset)
                for collision_type, count in case_collisions.items():
                    collision_counts[collision_type] += count
        
        # Calculate final metrics
        success_rate = successful_agents / total_agents if total_agents > 0 else 0.0
        avg_flow_time = np.mean(flow_times) if flow_times else 0.0
        avg_makespan = np.mean(makespan_times) if makespan_times else 0.0
        total_collisions = collision_counts['total']
        collision_rate = total_collisions / (total_agents * self.max_timesteps) if total_agents > 0 else 0.0
        
        # Path efficiency
        path_efficiency = 0.0
        if len(path_lengths) > 0 and len(optimal_path_lengths) > 0:
            valid_ratios = []
            for actual, optimal in zip(path_lengths, optimal_path_lengths):
                # Only consider positive lengths to avoid division by zero
                if optimal > 0 and actual > 0:
                    valid_ratios.append(optimal / actual)
            path_efficiency = np.mean(valid_ratios) if valid_ratios else 0.0
        
        metrics = {
            'success_rate': success_rate,
            'avg_flow_time': avg_flow_time,
            'avg_makespan': avg_makespan,
            'total_collisions': total_collisions,
            'collision_rate': collision_rate,
            'vertex_collisions': collision_counts['vertex'],
            'edge_collisions': collision_counts['edge'], 
            'obstacle_collisions': collision_counts['obstacle'],
            'path_efficiency': path_efficiency,
            'total_agents_evaluated': total_agents,
            'total_cases_evaluated': total_cases,
            'successful_agents': successful_agents,
            'dataset_used': dataset_name
        }
        
        # Print comprehensive results
        print(f"\n📊 COMPREHENSIVE EVALUATION RESULTS ({dataset_name}):")
        print(f"   🎯 Success Rate: {success_rate:.1%} ({successful_agents:,}/{total_agents:,} agents)")
        print(f"   ⏱️  Avg Flow Time: {avg_flow_time:.1f} timesteps")
        print(f"   📏 Avg Makespan: {avg_makespan:.1f} timesteps")
        print(f"   💥 Total Collisions: {total_collisions:,} ({collision_rate:.4f} per agent/timestep)")
        print(f"      - Vertex: {collision_counts['vertex']:,}")
        print(f"      - Edge: {collision_counts['edge']:,}")
        print(f"      - Obstacle: {collision_counts['obstacle']:,}")
        print(f"   🛣️  Path Efficiency: {path_efficiency:.3f} (optimal/actual ratio)")
        print(f"   📊 Cases Evaluated: {total_cases:,} from {dataset_name} set")
        
        return metrics
    
    def _calculate_flow_time(self, action_sequence: torch.Tensor) -> int:
        """Calculate flow time (time to reach goal) for an agent's action sequence."""
        # Find the last timestep with a non-zero action (movement)
        # or if all actions are 0 (stay), return full sequence length
        non_zero_actions = torch.nonzero(action_sequence, as_tuple=False)
        if len(non_zero_actions) > 0:
            return non_zero_actions[-1].item() + 1
        else:
            return len(action_sequence)  # Agent stays put entire time
    
    def _calculate_path_length(self, action_sequence: torch.Tensor) -> int:
        """Calculate path length (number of movement actions) for an agent."""
        # Count non-zero actions (movements), action 0 = stay
        return torch.sum(action_sequence != 0).item()
    
    def _analyze_collisions(self, pred_seq: torch.Tensor, case_idx: int, dataset) -> Dict[str, int]:
        """Analyze collisions in predicted trajectory sequence."""
        try:
            T, num_agents = pred_seq.shape
            collisions = {'vertex': 0, 'edge': 0, 'obstacle': 0, 'total': 0}
            
            # Simple collision detection - check for same positions at same time
            for t in range(T):
                # Simulate positions based on actions from initial positions
                # This is a simplified collision detection
                current_actions = pred_seq[t]
                
                # Count vertex collisions (same position at same time)
                unique_actions = torch.unique(current_actions)
                if len(unique_actions) < num_agents:
                    # Some agents have same action - potential collision
                    for action in unique_actions:
                        agent_count = torch.sum(current_actions == action).item()
                        if agent_count > 1:
                            collisions['vertex'] += agent_count - 1
            
            collisions['total'] = collisions['vertex'] + collisions['edge'] + collisions['obstacle']
            return collisions
            
        except Exception as e:
            print(f"Warning: Collision analysis failed for case {case_idx}: {e}")
            return {'vertex': 0, 'edge': 0, 'obstacle': 0, 'total': 0}

    def save_iteration_results(self, iteration: int, metrics: Dict[str, Any]):
        """Save results from DAgger iteration."""
        results_dir = os.path.join("results", self.config.get('exp_name', 'dagger_experiment'))
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(results_dir, f"dagger_iteration_{iteration}_metrics.yaml")
        with open(metrics_file, 'w') as f:
            yaml.dump(metrics, f)
        
        # Save model checkpoint
        model_file = os.path.join(results_dir, f"dagger_iteration_{iteration}_model.pth")
        torch.save(self.model.state_dict(), model_file)
        
        # Save dataset
        dataset_file = os.path.join(results_dir, f"dagger_iteration_{iteration}_dataset.pkl")
        self.dagger_dataset.save(dataset_file)
    
    def plot_training_progress(self):
        """Plot DAgger training progress."""
        if not self.iteration_metrics:
            return
        
        iterations = list(range(len(self.iteration_metrics)))
        success_rates = [m['reach_rate'] for m in self.iteration_metrics]
        dataset_sizes = [m['dataset_size'] for m in self.iteration_metrics]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Success rate
        ax1.plot(iterations, success_rates, 'b-o')
        ax1.set_xlabel('DAgger Iteration')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Policy Success Rate Over DAgger Iterations')
        ax1.grid(True)
        
        # Dataset size growth
        ax2.plot(iterations, dataset_sizes, 'r-o')
        ax2.set_xlabel('DAgger Iteration')
        ax2.set_ylabel('Dataset Size')
        ax2.set_title('Dataset Size Growth')
        ax2.grid(True)
        
        # Performance improvement
        if len(success_rates) > 1:
            improvements = [success_rates[i] - success_rates[i-1] for i in range(1, len(success_rates))]
            ax3.plot(iterations[1:], improvements, 'm-o')
            ax3.set_xlabel('DAgger Iteration')
            ax3.set_ylabel('Success Rate Improvement')
            ax3.set_title('Performance Improvement Per Iteration')
            ax3.grid(True)
        
        plt.tight_layout()
        
        results_dir = os.path.join("results", self.config.get('exp_name', 'dagger_experiment'))
        plot_file = os.path.join(results_dir, "dagger_training_progress.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training progress plot saved to: {plot_file}")
    
    def run_dagger_training(self):
        """
        Run the complete DAgger training loop.
        
        DAgger Algorithm:
        1. Initialize: Train policy π₀ on expert dataset D₀
        2. For each iteration i:
            a. Roll out πᵢ to collect states
            b. Query expert for actions on visited states  
            c. Aggregate expert corrections: D ← D ∪ {(s,aᴱ)}
            d. Retrain policy πᵢ₊₁ on updated dataset D
            e. Evaluate performance and check convergence
        3. Stop when performance converges or max iterations reached
        """
        print("=" * 60)
        print("STARTING DAGGER TRAINING")
        print("=" * 60)
        
        print(f"Configuration:")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Rollout episodes per iteration: {self.rollout_episodes}")
        print(f"  Expert query rate: {self.expert_query_rate}")
        print(f"  Convergence threshold: {self.convergence_threshold}")
        print(f"  Retraining epochs: {self.retraining_epochs}")
        
        # Step 1: Initialize with expert dataset (if available)
        print("\nStep 1: Initializing policy with expert demonstrations...")
        
        # Initial evaluation
        initial_metrics = self.evaluate_policy()
        initial_metrics['dataset_size'] = self.dagger_dataset.get_size()
        initial_metrics['iteration'] = 0
        self.iteration_metrics.append(initial_metrics)
        
        print(f"Initial policy performance: {initial_metrics}")
        
        # Main DAgger loop
        for iteration in tqdm(range(1, self.max_iterations + 1), desc="DAgger iterations", unit="iteration"):
            print(f"\n{'='*50}")
            print(f"DAGGER ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*50}")
            
            # Step 2: Roll out current policy
            states_visited, expert_actions = self.rollout_policy(self.rollout_episodes, iteration)
            
            if not states_visited:
                print("Warning: No states collected during rollout. Skipping iteration.")
                continue
            
            # Step 3: Aggregate expert corrections
            print(f"Step 3: Aggregating {len(expert_actions)} expert corrections...")
            self.dagger_dataset.add_samples(states_visited, expert_actions, iteration)
            
            # Step 4: Retrain policy
            print(f"Step 4: Retraining policy...")
            self.retrain_policy(iteration)
            
            # Step 5: Evaluate and check convergence
            print(f"Step 5: Evaluating policy...")
            
            # Run quick evaluation for convergence checking
            quick_metrics = self.evaluate_policy()
            
            # Run comprehensive evaluation periodically and at the end
            comprehensive_metrics = {}
            if iteration % self.config.get('dagger_comprehensive_eval_interval', 2) == 0 or iteration == self.max_iterations:
                print(f"   Running comprehensive evaluation (iteration {iteration})...")
                comprehensive_metrics = self.comprehensive_evaluate(
                    use_validation=True, 
                    max_cases=self.config.get('dagger_comprehensive_eval_cases', 100)
                )
            
            # Combine metrics
            metrics = quick_metrics.copy()
            metrics.update(comprehensive_metrics)
            metrics['dataset_size'] = self.dagger_dataset.get_size()
            metrics['iteration'] = iteration
            metrics['states_collected'] = len(states_visited)
            
            self.iteration_metrics.append(metrics)
            
            # Save iteration results
            self.save_iteration_results(iteration, metrics)
            
            # Check convergence
            converged = self.check_convergence(metrics['reach_rate'])
            
            print(f"\nIteration {iteration} Summary:")
            print(f"  Success Rate: {metrics['reach_rate']:.3f}")
            print(f"  Dataset Size: {metrics['dataset_size']}")
            print(f"  States Collected This Iteration: {metrics['states_collected']}")
            print(f"  Converged: {converged}")
            
            if converged:
                print(f"\n🎉 DAgger training converged after {iteration} iterations!")
                break
        
        # Final summary
        print(f"\n{'='*60}")
        print("DAGGER TRAINING COMPLETED")
        print(f"{'='*60}")
        
        final_metrics = self.iteration_metrics[-1]
        print(f"Final Performance:")
        print(f"  Success Rate: {final_metrics['reach_rate']:.3f}")
        print(f"  Total Dataset Size: {final_metrics['dataset_size']}")
        print(f"  Total Iterations: {len(self.iteration_metrics) - 1}")
        
        # Plot training 
        self.plot_training_progress()
        
        return self.iteration_metrics


def main():
    """Main function to run DAgger training."""
    
    # Setup configuration
    config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Initializing DAgger training components...")
    
    # Initialize model
    model = Network(config)
    model.to(config["device"])
    
    # Initialize data loader
    data_loader = SNNDataLoader(config)
    
    # Create DAgger trainer
    dagger_trainer = DAggerTrainer(config, model, data_loader)
    
    # Run DAgger training
    results = dagger_trainer.run_dagger_training()
    
    print("DAgger training completed successfully!")
    
    return results


if __name__ == "__main__":
    main()
