#!/usr/bin/env python3
"""
Enhanced Multi-Agent Training with Communication-Aware Learning
================================================================

This module provides an enhanced supervised learning approach that:
1. Trains on multi-step sequences of actions instead of single actions
2. Includes communication-aware losses to encourage coordination between agents
3. Provides real-time visualization of training progress and agent behaviors
4. Uses smart data augmentation techniques to expand the effective training dataset
5. Optimized for efficient training on standard laptops (lightweight and fast)

Author: Enhanced MAPF-GNN Training System
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
# Use TkAgg backend but with proper thread safety measures
matplotlib.use('TkAgg')  # GUI backend with thread safety
import matplotlib.pyplot as plt
# Configure matplotlib for thread safety
plt.rcParams['figure.raise_window'] = False  # Prevent focus stealing
plt.rcParams['font.size'] = 10  # Ensure font is properly configured
plt.rcParams['axes.unicode_minus'] = False  # Prevent unicode issues
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import yaml
from tqdm import tqdm
import time
import os
from collections import defaultdict, deque
import threading
import queue

# Import existing modules
from config import config
from data_loader import SNNDataLoader
from models.framework_snn import Network
from collision_utils import compute_collision_loss
from debug_utils import debug_print

# Import entropy functions from train.py
from train import entropy_bonus_schedule, compute_entropy_bonus

# SNN-specific imports
from spikingjelly.activation_based import functional

class EnhancedTrainer:
    """Enhanced trainer with multi-step learning and visualization"""
    
    def __init__(self, config_path="configs/config_snn.yaml"):
        # Load configuration with proper fallback
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"📋 Loaded config from {config_path}")
        else:
            print(f"⚠️  Config file {config_path} not found, using global config")
            self.config = config
            
        # Extract board size and grid size from config
        self.board_size = self.config.get('board_size', [9, 9])
        self.grid_size = self.board_size[0]  # Assuming square grid
        print(f"🗺️  Using board size: {self.board_size}, grid size: {self.grid_size}")
        
        self.device = self.config['device']
        print(f"🚀 Enhanced Trainer initialized on {self.device}")
        
        # Training parameters
        # Adjust sequence length based on dataset min_time
        dataset_min_time = self.config.get('min_time', 5)
        self.sequence_length = min(self.config.get('sequence_length', 8), dataset_min_time)  # Multi-step training
        self.batch_size = self.config.get('batch_size', 16)
        self.learning_rate = float(self.config.get('learning_rate', 0.001))
        self.num_epochs = self.config.get('epochs', 50)
        
        print(f"   📏 Adjusted sequence length to {self.sequence_length} (dataset min_time: {dataset_min_time})")
        
        # Communication learning parameters
        self.communication_weight = self.config.get('communication_weight', 0.3)
        self.coordination_weight = self.config.get('coordination_weight', 0.2)
        self.temporal_consistency_weight = self.config.get('temporal_consistency_weight', 0.1)
        
        # Initialize model first
        self.model = Network(self.config).to(self.device)
        
        # Back-and-forth pattern detection parameters (after model initialization)
        self.oscillation_detection_window = self.config.get('oscillation_detection_window', 4)  # Look back N steps
        # Register oscillation penalty weight as a parameter of the model
        self.oscillation_penalty_weight = nn.Parameter(torch.tensor(1.0, device=self.device))
        # Register the parameter with the model for proper gradient tracking
        self.model.register_parameter('oscillation_penalty_weight', self.oscillation_penalty_weight)
        
        # Agent movement history tracking (for oscillation detection)
        self.agent_position_history = {}  # Dict of deques per episode/batch
        self.max_history_length = max(8, self.oscillation_detection_window * 2)  # Keep enough history
        
        print(f"   🔄 Oscillation detection: window={self.oscillation_detection_window}, penalty_weight={self.oscillation_penalty_weight.item():.3f} (learnable)")
        
        # Visualization parameters
        self.visualize_training = self.config.get('visualize_training', True)
        self.vis_update_freq = self.config.get('vis_update_freq', 10)  # Update every N batches
        self.max_vis_episodes = self.config.get('max_vis_episodes', 3)  # Show max 3 episodes
        
        # Initialize optimizer after all parameters are registered
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.training_history = {
            'epoch_losses': [],
            'communication_losses': [],
            'coordination_scores': [],
            'collision_rates': [],
            'success_rates': [],  # training success rates (placeholder)
            'entropy_bonuses': [],  # Add entropy bonus tracking
            'progress_rewards': [],  # Add progress reward tracking
            'oscillation_penalties': [],  # Add oscillation penalty tracking
            # Adjacent coordination tracking removed in simplified implementation
            # validation metrics
            'val_success_rates': [],
            'val_agents_reached_rates': [],
            'val_avg_flow_times': []
        }
        
        # Visualization components
        self.vis_queue = queue.Queue() if self.visualize_training else None
        self.vis_thread = None
        self.fig = None
        self.axes = None
        
        # Animation queue system for sequential visualization
        if self.visualize_training:
            self.setup_animation_queue()
            self.setup_visualization()  # Initialize visualization immediately
        
        print("✅ Enhanced Trainer ready!")
    
    def _load_trajectory_data(self, dataset_path, case_name):
        """
        Load trajectory data from dataset for position reconstruction.
        
        Args:
            dataset_path: Path to dataset directory
            case_name: Name of the case directory (e.g., 'case_0')
            
        Returns:
            trajectory: numpy array of shape [num_agents, timesteps] with action indices
        """
        trajectory_path = os.path.join(dataset_path, case_name, "trajectory.npy")
        
        if os.path.exists(trajectory_path):
            return np.load(trajectory_path)
        else:
            print(f"WARNING: trajectory.npy not found for {case_name}, using dummy actions")
            # Return dummy stay actions if trajectory file is missing
            return np.zeros((5, 10))  # 5 agents, 10 timesteps, all stay actions
    
    def _load_initial_positions_from_yaml(self, dataset_path, case_name):
        """
        Load initial agent positions from input.yaml file.
        
        Args:
            dataset_path: Path to dataset directory  
            case_name: Name of the case directory (e.g., 'case_0')
            
        Returns:
            positions: numpy array of shape [num_agents, 2] with initial positions
        """
        input_yaml_path = os.path.join(dataset_path, case_name, "input.yaml")
        
        if os.path.exists(input_yaml_path):
            try:
                import yaml
                with open(input_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                agents = data.get('agents', [])
                positions = []
                
                for agent in agents:
                    start_pos = agent.get('start', [0, 0])
                    positions.append([float(start_pos[0]), float(start_pos[1])])
                    
                return np.array(positions)
                
            except Exception as e:
                print(f"WARNING: Error loading {input_yaml_path}: {e}, using dummy positions")
        
        # Fallback: generate better distributed dummy positions
        num_agents = 5
        positions = []
        grid_size = self.grid_size
        
        for agent_idx in range(num_agents):
            # Better distribution pattern than diagonal
            x = (agent_idx * 2) % grid_size
            y = (agent_idx * 3) % grid_size
            positions.append([float(x), float(y)])
        
        return np.array(positions)
    
    def _load_goal_positions_from_yaml(self, dataset_path, case_name):
        """
        Load goal positions from input.yaml file.
        
        Args:
            dataset_path: Path to dataset directory
            case_name: Name of the case directory (e.g., 'case_0')
            
        Returns:
            goals: numpy array of shape [num_agents, 2] with goal positions
        """
        input_yaml_path = os.path.join(dataset_path, case_name, "input.yaml")
        
        if os.path.exists(input_yaml_path):
            try:
                import yaml
                with open(input_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                agents = data.get('agents', [])
                positions = []
                
                for agent in agents:
                    goal_pos = agent.get('goal', [0, 0])
                    positions.append([float(goal_pos[0]), float(goal_pos[1])])
                    
                return np.array(positions)
                
            except Exception as e:
                print(f"WARNING: Error loading {input_yaml_path}: {e}, using dummy goals")
        
        # Fallback: generate better distributed dummy goals (opposite side of grid)
        num_agents = 5
        positions = []
        grid_size = self.grid_size
        
        for agent_idx in range(num_agents):
            # Place goals on opposite side with different pattern
            x = grid_size - 1 - ((agent_idx * 2) % grid_size)
            y = grid_size - 1 - ((agent_idx * 3) % grid_size)
            positions.append([float(x), float(y)])
        
        return np.array(positions)
    
    def _reconstruct_positions_from_trajectory(self, initial_positions, trajectory, timestep):
        """
        Reconstruct agent positions at a given timestep using action sequence.
        
        Args:
            initial_positions: numpy array [num_agents, 2] with starting positions
            trajectory: numpy array [num_agents, timesteps] with action indices
            timestep: int, which timestep to reconstruct positions for
            
        Returns:
            positions: torch tensor [num_agents, 2] with reconstructed positions
        """
        # Action mapping: 0=stay, 1=right, 2=up, 3=left, 4=down
        action_deltas = np.array([
            [0, 0],   # 0: Stay
            [1, 0],   # 1: Right 
            [0, 1],   # 2: Up
            [-1, 0],  # 3: Left
            [0, -1]   # 4: Down
        ])
        
        positions = initial_positions.copy()
        grid_size = self.grid_size
        
        # Apply actions sequentially up to the desired timestep
        for t in range(min(timestep, trajectory.shape[1])):
            for agent_idx in range(positions.shape[0]):
                if t < trajectory.shape[1]:
                    action = int(trajectory[agent_idx, t])
                    if 0 <= action < len(action_deltas):
                        delta = action_deltas[action]
                        positions[agent_idx] += delta
                        
                        # Apply boundary constraints
                        positions[agent_idx, 0] = np.clip(positions[agent_idx, 0], 0, grid_size - 1)
                        positions[agent_idx, 1] = np.clip(positions[agent_idx, 1], 0, grid_size - 1)
        
        return torch.tensor(positions, dtype=torch.float32)
    
    def _extract_positions_from_fov(self, fov_data, case_idx=None, dataset_path=None):
        """
        Extract agent positions using trajectory-based reconstruction instead of FOV parsing.
        This method now loads actual trajectory data and reconstructs real positions.
        
        Args:
            fov_data: tensor of shape [num_agents, 2, 5, 5] (used for determining case info)
            case_idx: int, case index to load data for
            dataset_path: str, path to dataset directory
                     
        Returns:
            positions: tensor of shape [num_agents, 2] with (x, y) coordinates
        """
        # Use provided case info or fallback to case_0
        if dataset_path is None:
            dataset_path = "/home/arnav/dev/summer25/MAPF-GNN/dataset/5_8_28/train"
        if case_idx is None:
            case_idx = 0
            
        case_name = f"case_{case_idx}"
        
        # Load initial positions and trajectory
        initial_positions = self._load_initial_positions_from_yaml(dataset_path, case_name)
        trajectory = self._load_trajectory_data(dataset_path, case_name)
        
        # For visualization, use initial positions (timestep 0)
        # In training, this would reconstruct positions for specific timesteps
        positions = self._reconstruct_positions_from_trajectory(initial_positions, trajectory, timestep=0)
        
        return positions

    def _extract_goals_from_fov(self, fov_data, case_idx=None, dataset_path=None):
        """
        Extract goal positions using YAML data instead of FOV parsing.
        This method now loads actual goal data from input.yaml files.
        
        Args:
            fov_data: tensor of shape [num_agents, 2, 5, 5] (used for determining case info)
            case_idx: int, case index to load data for
            dataset_path: str, path to dataset directory
                     
        Returns:
            goals: tensor of shape [num_agents, 2] with (x, y) coordinates
        """
        # Use provided case info or fallback to case_0
        if dataset_path is None:
            dataset_path = "/home/arnav/dev/summer25/MAPF-GNN/dataset/5_8_28/train"
        if case_idx is None:
            case_idx = 0
            
        case_name = f"case_{case_idx}"
        
        # Load goal positions from YAML
        goal_positions = self._load_goal_positions_from_yaml(dataset_path, case_name)
        
        return torch.tensor(goal_positions, dtype=torch.float32)
    
    def _load_obstacles_from_yaml(self, dataset_path, case_name):
        """
        Load obstacle positions from input.yaml file.
        
        Args:
            dataset_path: Path to dataset directory
            case_name: Name of the case directory (e.g., 'case_0')
            
        Returns:
            obstacles: numpy array of shape [num_obstacles, 2] with obstacle positions
        """
        input_yaml_path = os.path.join(dataset_path, case_name, "input.yaml")
        
        if not os.path.exists(input_yaml_path):
            print(f"WARNING: input.yaml not found at {input_yaml_path}")
            return np.array([])
        
        try:
            import yaml
            with open(input_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Extract obstacles from map data
            map_data = data.get('map', {})
            obstacles = map_data.get('obstacles', [])
            
            if obstacles:
                # Convert obstacles to numpy array
                obstacle_positions = []
                for obs in obstacles:
                    if isinstance(obs, list) and len(obs) >= 2:
                        obstacle_positions.append([float(obs[0]), float(obs[1])])
                
                if obstacle_positions:
                    debug_print(f"DEBUG: Loaded {len(obstacle_positions)} obstacles from {case_name}")
                    return np.array(obstacle_positions)
                else:
                    return np.array([])
            else:
                debug_print(f"DEBUG: No obstacles found in {case_name}")
                return np.array([])  # No obstacles
                
        except Exception as e:
            print(f"WARNING: Error loading obstacles from {input_yaml_path}: {e}")
            return np.array([])


    


    def _extract_obstacles_from_fov(self, fov_data, case_idx=None, dataset_path=None):
        """
        Extract obstacle positions using YAML data instead of FOV parsing.
        This method now loads actual obstacle data from input.yaml files.
        
        Args:
            fov_data: tensor of shape [num_agents, 2, 5, 5] (used for determining case info)
            case_idx: int, case index to load data for
            dataset_path: str, path to dataset directory
                     
        Returns:
            obstacles: numpy array of shape [num_obstacles, 2] with obstacle positions
        """
        # Use provided case info or fallback to case_0
        if dataset_path is None:
            dataset_path = "/home/arnav/dev/summer25/MAPF-GNN/dataset/5_8_28/train"
        if case_idx is None:
            case_idx = 0
            
        case_name = f"case_{case_idx}"
        
        # Load obstacle positions from YAML
        obstacle_positions = self._load_obstacles_from_yaml(dataset_path, case_name)
        
        return obstacle_positions
    
    def augment_dataset(self, fovs, actions, positions, goals, augment_factor=5):
        """Smart data augmentation to expand training data"""
        print(f"🔄 Augmenting dataset by factor of {augment_factor}...")
        
        original_size = fovs.shape[0]
        augmented_fovs = [fovs]
        augmented_actions = [actions] 
        augmented_positions = [positions]
        augmented_goals = [goals]
        
        for aug_idx in range(augment_factor - 1):
            # Rotation augmentation (90, 180, 270 degrees)
            if aug_idx == 0:  # 90 degree rotation
                rot_fovs = torch.rot90(fovs, k=1, dims=[-2, -1])
                rot_actions = self._rotate_actions(actions, 1)
                rot_positions = self._rotate_positions(positions, 1)
                rot_goals = self._rotate_positions(goals, 1)
            elif aug_idx == 1:  # 180 degree rotation  
                rot_fovs = torch.rot90(fovs, k=2, dims=[-2, -1])
                rot_actions = self._rotate_actions(actions, 2)
                rot_positions = self._rotate_positions(positions, 2)
                rot_goals = self._rotate_positions(goals, 2)
            elif aug_idx == 2:  # 270 degree rotation
                rot_fovs = torch.rot90(fovs, k=3, dims=[-2, -1])
                rot_actions = self._rotate_actions(actions, 3)
                rot_positions = self._rotate_positions(positions, 3)
                rot_goals = self._rotate_positions(goals, 3)
            else:  # Horizontal flip
                rot_fovs = torch.flip(fovs, dims=[-1])
                rot_actions = self._flip_actions(actions, horizontal=True)
                rot_positions = self._flip_positions(positions, horizontal=True)
                rot_goals = self._flip_positions(goals, horizontal=True)
            
            augmented_fovs.append(rot_fovs)
            augmented_actions.append(rot_actions)
            augmented_positions.append(rot_positions)
            augmented_goals.append(rot_goals)
        
        # Concatenate all augmented data
        final_fovs = torch.cat(augmented_fovs, dim=0)
        final_actions = torch.cat(augmented_actions, dim=0)
        final_positions = torch.cat(augmented_positions, dim=0)
        final_goals = torch.cat(augmented_goals, dim=0)
        
        print(f"✅ Dataset augmented: {original_size} → {final_fovs.shape[0]} samples")
        return final_fovs, final_actions, final_positions, final_goals
    
    def _rotate_actions(self, actions, k):
        """Rotate action indices for rotated environments"""
        # Action mapping: 0=Stay, 1=Right, 2=Up, 3=Left, 4=Down
        rotation_map = {
            1: {0: 0, 1: 2, 2: 3, 3: 4, 4: 1},  # 90 degrees
            2: {0: 0, 1: 3, 2: 4, 3: 1, 4: 2},  # 180 degrees  
            3: {0: 0, 1: 4, 2: 1, 3: 2, 4: 3},  # 270 degrees
        }
        
        if k not in rotation_map:
            return actions
            
        rot_actions = actions.clone()
        for old_action, new_action in rotation_map[k].items():
            rot_actions[actions == old_action] = new_action
        
        return rot_actions
    
    def _rotate_positions(self, positions, k):
        """Rotate position coordinates"""
        if k == 0:
            return positions
        
        # Use grid size from config
        grid_size = self.grid_size
        rot_positions = positions.clone()
        
        if k == 1:  # 90 degrees: (x,y) -> (y, grid_size-1-x)
            rot_positions[..., 0] = positions[..., 1]
            rot_positions[..., 1] = grid_size - 1 - positions[..., 0]
        elif k == 2:  # 180 degrees: (x,y) -> (grid_size-1-x, grid_size-1-y)
            rot_positions[..., 0] = grid_size - 1 - positions[..., 0]
            rot_positions[..., 1] = grid_size - 1 - positions[..., 1]
        elif k == 3:  # 270 degrees: (x,y) -> (grid_size-1-y, x)
            rot_positions[..., 0] = grid_size - 1 - positions[..., 1]
            rot_positions[..., 1] = positions[..., 0]
        
        return rot_positions
    
    def _flip_actions(self, actions, horizontal=True):
        """Flip actions for mirrored environments"""
        if horizontal:
            # Horizontal flip: swap left/right actions
            flip_map = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4}  # Right<->Left
        else:
            # Vertical flip: swap up/down actions  
            flip_map = {0: 0, 1: 1, 2: 4, 3: 3, 4: 2}  # Up<->Down
        
        flipped_actions = actions.clone()
        for old_action, new_action in flip_map.items():
            flipped_actions[actions == old_action] = new_action
        
        return flipped_actions
    
    def _flip_positions(self, positions, horizontal=True):
        """Flip position coordinates"""
        grid_size = self.grid_size
        flipped_positions = positions.clone()
        
        if horizontal:  # Horizontal flip: x -> grid_size-1-x
            flipped_positions[..., 0] = grid_size - 1 - positions[..., 0]
        else:  # Vertical flip: y -> grid_size-1-y
            flipped_positions[..., 1] = grid_size - 1 - positions[..., 1]
        
        return flipped_positions
    
    def create_sequence_dataset(self, dataset_root="dataset/5_8_28", mode="train"):
        """Create multi-step sequence dataset"""
        print(f"📊 Creating sequence dataset from {dataset_root}/{mode}")
        
        # Create proper data loader using the same approach as train.py
        data_loader = SNNDataLoader(self.config)
        
        sequences_fov = []
        sequences_actions = []
        sequences_positions = []
        sequences_goals = []
        sequences_obstacles = []
        
        # Choose the appropriate loader based on mode
        loader = data_loader.train_loader if mode == "train" else data_loader.valid_loader
        if loader is None:
            print(f"Warning: No {mode} loader available")
            return self._create_empty_dataset()
        
        case_count = 0
        valid_sequences_found = 0
        for batch_idx, batch_data in enumerate(tqdm(loader, desc="Processing cases")):
            try:
                # Unpack batch data (supports both 3 and 4 element formats)
                if len(batch_data) == 4:  # With case_idx
                    fovs, actions, _, case_indices = batch_data
                else:  # Without case_idx
                    fovs, actions, _ = batch_data
                    case_indices = [case_count + i for i in range(fovs.shape[0])]
                
                debug_print(f"Batch {batch_idx}: fovs.shape={fovs.shape}, actions.shape={actions.shape}")
                
                # Process each case in the batch
                for b in range(fovs.shape[0]):
                    # fovs: [batch, T, num_agents, 2, 5, 5], actions: [batch, T, num_agents]
                    case_fov = fovs[b]  # [T, num_agents, 2, 5, 5]
                    case_actions = actions[b]  # [T, num_agents]
                    case_idx = case_indices[b] if isinstance(case_indices, (list, torch.Tensor)) else case_count
                    
                    T = case_fov.shape[0]
                    num_agents = case_fov.shape[1]
                    
                    debug_print(f"Case {case_idx}: T={T}, num_agents={num_agents}, sequence_length={self.sequence_length}")
                    
                    if T < self.sequence_length:
                        debug_print(f"Skipping case {case_idx}: T={T} < sequence_length={self.sequence_length}")
                        continue  # Skip sequences that are too short
                    
                    # Extract real positions from trajectory and YAML data
                    try:
                        # Build the correct dataset path for this mode
                        mode_dataset_path = os.path.join(dataset_root, mode)
                        
                        # Extract agent positions and goals using real case data
                        first_fov = case_fov[0]  # [num_agents, 2, 5, 5]
                        positions = self._extract_positions_from_fov(first_fov, case_idx=case_idx, dataset_path=mode_dataset_path)
                        goals = self._extract_goals_from_fov(first_fov, case_idx=case_idx, dataset_path=mode_dataset_path)
                        obstacles = self._extract_obstacles_from_fov(first_fov, case_idx=case_idx, dataset_path=mode_dataset_path)
                        debug_print(f"Extracted REAL data for case {case_idx}: agents at {positions.tolist()}, goals at {goals.tolist()}, obstacles: {len(obstacles) if obstacles is not None else 0}")
                    except Exception as e:
                        # Fallback: generate better distributed positions (not diagonal)
                        grid_size = self.grid_size
                        positions_per_row = int(np.ceil(np.sqrt(num_agents)))
                        positions = []
                        goals = []
                        
                        for i in range(num_agents):
                            # Agent positions: distributed across grid
                            row = i // positions_per_row
                            col = i % positions_per_row
                            spacing = max(1, grid_size // (positions_per_row + 1))
                            x = col * spacing + 1
                            y = row * spacing + 1
                            positions.append([min(x, grid_size - 1), min(y, grid_size - 1)])
                            
                            # Goal positions: opposite pattern
                            goal_row = (num_agents - 1 - i) // positions_per_row
                            goal_col = (num_agents - 1 - i) % positions_per_row
                            goal_x = grid_size - 1 - (goal_col * spacing + 1)
                            goal_y = grid_size - 1 - (goal_row * spacing + 1)
                            goals.append([max(0, goal_x), max(0, goal_y)])
                        
                        positions = torch.tensor(positions, dtype=torch.float32)
                        goals = torch.tensor(goals, dtype=torch.float32)
                        obstacles = np.array([])  # No obstacles for fallback case
                        debug_print(f"Failed to extract positions for case {case_idx}, using fallback grid: {e}")
                    
                    # Create overlapping sequences
                    for start_t in range(T - self.sequence_length + 1):
                        end_t = start_t + self.sequence_length
                        
                        seq_fov = case_fov[start_t:end_t]  # [seq_len, num_agents, 2, 5, 5]
                        seq_actions = case_actions[start_t:end_t]  # [seq_len, num_agents]
                        
                        sequences_fov.append(seq_fov)
                        sequences_actions.append(seq_actions)
                        sequences_positions.append(positions)  # [num_agents, 2]
                        sequences_goals.append(goals)  # [num_agents, 2]
                        sequences_obstacles.append(obstacles)  # [num_obstacles, 2] or empty array
                        valid_sequences_found += 1
                    
                    case_count += 1
                
            except Exception as e:
                debug_print(f"Warning: Failed to process batch {batch_idx}: {e}")
                continue
        
        if not sequences_fov:
            print(f"❌ No valid sequences found! Processed {case_count} cases, found {valid_sequences_found} sequences")
            print(f"   Required sequence length: {self.sequence_length}, Dataset min_time: {self.config.get('min_time', 'unknown')}")
            raise ValueError("No valid sequences found in dataset!")
        
        # Convert to tensors
        fovs_tensor = torch.stack(sequences_fov)  # [N, seq_len, num_agents, 2, 5, 5]
        actions_tensor = torch.stack(sequences_actions).long()  # [N, seq_len, num_agents] - ensure Long type
        positions_tensor = torch.stack(sequences_positions)  # [N, num_agents, 2]
        goals_tensor = torch.stack(sequences_goals)  # [N, num_agents, 2]
        
        # Handle obstacles - they can have variable length, so we store them as a list
        obstacles_list = sequences_obstacles  # Keep as list since obstacle count can vary per case
        
        print(f"✅ Created {len(sequences_fov)} sequences of length {self.sequence_length}")
        print(f"   FOV shape: {fovs_tensor.shape}")
        print(f"   Actions shape: {actions_tensor.shape}")
        
        return fovs_tensor, actions_tensor, positions_tensor, goals_tensor, obstacles_list
    
    def _create_empty_dataset(self):
        """Create empty tensors for when no valid data is found"""
        return (
            torch.empty(0, self.sequence_length, 5, 2, 5, 5),  # Empty FOV tensor
            torch.empty(0, self.sequence_length, 5, dtype=torch.long),  # Empty actions tensor
            torch.empty(0, 5, 2),  # Empty positions tensor
            torch.empty(0, 5, 2),   # Empty goals tensor
            []  # Empty obstacles list
        )
    
    def compute_communication_loss(self, model_outputs, agent_positions):
        """Compute loss that encourages meaningful communication between agents"""
        
        # Extract hidden states/features from model if available
        if hasattr(self.model, 'get_hidden_states'):
            hidden_states = self.model.get_hidden_states()  # [batch, num_agents, hidden_dim]
        else:
            # Fallback: use output logits as proxy for communication
            hidden_states = model_outputs.detach()  # [batch, num_agents, action_dim]
        
        batch_size, num_agents = hidden_states.shape[:2]
        communication_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        # Track adjacent communication enhancement for logging
        adjacent_communication_count = 0
        total_pairs_processed = 0
        
        # 1. Proximity-based communication: nearby agents should have similar hidden states
        for b in range(batch_size):
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    total_pairs_processed += 1
                    
                    # Distance between agents
                    distance = torch.norm(agent_positions[b, i] - agent_positions[b, j])
                    
                    # Check if agents are adjacent (within 1.5 grid units - includes diagonal)
                    is_adjacent = distance <= 1.5
                    
                    # Communication strength inversely related to distance
                    comm_strength = torch.exp(-distance / 5.0)  # 5.0 is communication range
                    
                    # Hidden state similarity (higher is better for nearby agents)
                    state_similarity = torch.cosine_similarity(
                        hidden_states[b, i].unsqueeze(0),
                        hidden_states[b, j].unsqueeze(0)
                    ).item()  # Convert to scalar
                    
                    # Base communication loss
                    base_comm_loss = comm_strength * (1.0 - state_similarity)
                    
                    # Apply distance-based communication weight adjustment (same as coordination)
                    if is_adjacent:
                        # Double the communication weight for adjacent agents
                        comm_weight = self.communication_weight * 2.0
                        adjacent_communication_count += 1
                    else:
                        # Use base weight for non-adjacent agents
                        comm_weight = self.communication_weight
                    
                    # Apply communication weight to the loss
                    weighted_comm_loss = comm_weight * base_comm_loss
                    communication_loss += weighted_comm_loss
        
        # Log adjacent communication enhancement statistics
        from debug_utils import debug_print
        debug_print(f"📡 Adjacent Communication: {adjacent_communication_count}/{total_pairs_processed} pairs enhanced "
                   f"(distance-based detection)")
        
        return communication_loss / (batch_size * num_agents * (num_agents - 1) / 2)
    
    # Removed complex adjacent detection and dynamic coordination weight functions
    # Replaced with simple distance-based approach in compute_coordination_loss
    
    def compute_coordination_loss(self, predicted_actions, agent_positions):
        """
        Compute loss that encourages coordination between agents.
        Agents within one grid unit (adjacent) have doubled coordination penalties.
        """
        
        batch_size, num_agents = predicted_actions.shape[:2]
        coordination_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        # Action deltas
        action_deltas = torch.tensor([
            [0, 0],   # 0: Stay
            [1, 0],   # 1: Right
            [0, 1],   # 2: Up  
            [-1, 0],  # 3: Left
            [0, -1],  # 4: Down
        ], device=self.device, dtype=torch.float32)
        
        # Use torch.cdist to compute pairwise distances between agents
        # agent_positions: [batch, num_agents, 2]
        pairwise_distances = torch.cdist(agent_positions, agent_positions, p=2)  # [batch, num_agents, num_agents]
        
        # Track adjacent coordination for logging
        adjacent_coordination_count = 0
        total_pairs_processed = 0
        
        for b in range(batch_size):
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    total_pairs_processed += 1
                    
                    # Current positions
                    pos_i = agent_positions[b, i]
                    pos_j = agent_positions[b, j]
                    
                    # Check if agents are adjacent (within 1.5 grid units - includes diagonal)
                    current_distance = pairwise_distances[b, i, j]
                    is_adjacent = current_distance <= 1.5
                    
                    # Predicted next positions
                    action_i = torch.argmax(predicted_actions[b, i])
                    action_j = torch.argmax(predicted_actions[b, j])
                    
                    next_pos_i = pos_i + action_deltas[action_i]
                    next_pos_j = pos_j + action_deltas[action_j]
                    
                    # 1. Collision avoidance: penalize if agents move to same position
                    collision_penalty = 0.0
                    if torch.norm(next_pos_i - next_pos_j).item() < 0.1:  # Same position
                        collision_penalty = 10.0
                    
                    # 2. Traffic rules: if agents are close, encourage coordination
                    traffic_penalty = 0.0
                    if current_distance < 2.0:  # Close agents (includes adjacent)
                        # Encourage at least one agent to stay (action 0)
                        both_moving = (action_i.item() != 0) and (action_j.item() != 0)
                        if both_moving:
                            traffic_penalty = 2.0
                    
                    # Apply distance-based coordination weight adjustment
                    if is_adjacent:
                        # Double the coordination weight for adjacent agents
                        coord_weight = self.coordination_weight * 2.0
                        adjacent_coordination_count += 1
                    else:
                        # Use base weight for non-adjacent agents
                        coord_weight = self.coordination_weight
                    
                    # Apply coordination weight to penalties
                    weighted_penalty = coord_weight * (collision_penalty + traffic_penalty)
                    coordination_loss += weighted_penalty
        
        # Log adjacent coordination statistics
        debug_print(f"🤝 Adjacent Coordination: {adjacent_coordination_count}/{total_pairs_processed} pairs enhanced "
                   f"(distance-based detection)")
        
        normalized_loss = coordination_loss / (batch_size * num_agents * (num_agents - 1) / 2)
        return normalized_loss
    
    def compute_temporal_consistency_loss(self, sequence_outputs):
        """Encourage temporal consistency in agent actions with enhanced penalties"""
        
        # sequence_outputs: [batch, seq_len, num_agents, action_dim]
        batch_size, seq_len, num_agents, action_dim = sequence_outputs.shape
        
        if seq_len < 2:
            return torch.tensor(0.0, device=self.device)
        
        consistency_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        for t in range(seq_len - 1):
            # Compare consecutive timestep outputs
            curr_output = sequence_outputs[:, t]      # [batch, num_agents, action_dim]
            next_output = sequence_outputs[:, t + 1]  # [batch, num_agents, action_dim]
            
            # 1. Encourage smooth changes in action preferences (L2 norm)
            action_change = torch.norm(next_output - curr_output, dim=-1)  # [batch, num_agents]
            consistency_loss += action_change.mean()
            
            # 2. Penalize erratic action switching (action-level consistency)
            curr_actions = torch.argmax(curr_output, dim=-1)  # [batch, num_agents]
            next_actions = torch.argmax(next_output, dim=-1)  # [batch, num_agents]
            
            # Penalty for switching actions (especially from non-stay to stay)
            action_switches = (curr_actions != next_actions).float()
            consistency_loss += action_switches.mean() * 0.5  # Weight action switches
            
            # 3. Extra penalty for switching away from goal-directed movement
            # (This would require goal information, for now use action consistency)
            stop_starts = ((curr_actions == 0) & (next_actions != 0)).float()  # From stay to move
            move_stops = ((curr_actions != 0) & (next_actions == 0)).float()   # From move to stay
            consistency_loss += (stop_starts.mean() + move_stops.mean()) * 0.3
        
        # Scale the loss to make it more significant
        return consistency_loss * 2.0 / (seq_len - 1)
    
    def update_agent_position_history(self, agent_positions, batch_id=0):
        """Update position history for oscillation detection"""
        # agent_positions: [batch, num_agents, 2] or [num_agents, 2]
        if agent_positions.dim() == 3:
            batch_size, num_agents, _ = agent_positions.shape
        else:
            batch_size = 1
            num_agents = agent_positions.shape[0]
            agent_positions = agent_positions.unsqueeze(0)
        
        for b in range(batch_size):
            batch_key = f"batch_{batch_id}_{b}"
            
            if batch_key not in self.agent_position_history:
                # Initialize deques for all agents in this batch
                self.agent_position_history[batch_key] = {
                    agent_id: deque(maxlen=self.max_history_length) 
                    for agent_id in range(num_agents)
                }
            
            # Update position history for each agent
            for agent_id in range(num_agents):
                current_pos = agent_positions[b, agent_id].clone().detach()
                self.agent_position_history[batch_key][agent_id].append(current_pos)
    
    def detect_oscillation_patterns(self, batch_id=0):
        """Detect back-and-forth oscillation patterns using position history"""
        # Initialize with zero that maintains gradient connection to learnable parameter
        base_penalty = torch.tensor(0.0, device=self.oscillation_penalty_weight.device, dtype=torch.float32)
        accumulated_penalty = base_penalty
        total_agents = 0
        total_oscillations = 0
        
        # Check all batches in current training step
        for batch_key in list(self.agent_position_history.keys()):
            if not batch_key.startswith(f"batch_{batch_id}"):
                continue
                
            agent_histories = self.agent_position_history[batch_key]
            
            for agent_id, position_history in agent_histories.items():
                if len(position_history) < self.oscillation_detection_window:
                    continue
                
                total_agents += 1
                
                # Convert deque to tensor for analysis
                recent_positions = torch.stack(list(position_history)[-self.oscillation_detection_window:])
                
                # Detect oscillation patterns
                oscillation_score = self._compute_oscillation_score(recent_positions)
                
                # Add to accumulated penalty using tensor operations
                accumulated_penalty = accumulated_penalty + oscillation_score
                
                # Count oscillations for logging (use threshold for counting)
                if oscillation_score.item() > 0.1:
                    total_oscillations += 1
        
        # Apply learnable weight and normalization
        if total_agents > 0:
            # Normalize by number of agents
            normalized_penalty = accumulated_penalty / total_agents
            # Apply learnable weight
            final_penalty = self.oscillation_penalty_weight * normalized_penalty
            
            from debug_utils import debug_print
            debug_print(f"🔄 Oscillation detected: {total_oscillations}/{total_agents} agents, "
                       f"raw penalty: {normalized_penalty.item():.4f}, "
                       f"weight: {self.oscillation_penalty_weight.item():.4f}, "
                       f"final: {final_penalty.item():.4f}")
            return final_penalty
        
        # Return zero penalty that maintains gradient connection
        return self.oscillation_penalty_weight * base_penalty
    
    def _compute_oscillation_score(self, positions):
        """Compute oscillation score for a sequence of positions"""
        # positions: [window_size, 2]
        if len(positions) < 4:
            # Return zero scalar that can be used in calculations
            return torch.tensor(0.0, device=positions.device, dtype=torch.float32)
        
        # Calculate movement vectors
        movements = positions[1:] - positions[:-1]  # [window_size-1, 2]
        
        # Initialize oscillation score as scalar tensor
        oscillation_score = torch.tensor(0.0, device=positions.device, dtype=torch.float32)
        
        # Pattern 1: Direct back-and-forth (A->B->A->B)
        for i in range(len(movements) - 2):
            move1 = movements[i]
            move2 = movements[i + 1]
            
            # Check if we have valid movements
            move1_norm = torch.norm(move1)
            move2_norm = torch.norm(move2)
            
            # Use torch.where to maintain gradients instead of if statements
            valid_moves = (move1_norm > 0.1) & (move2_norm > 0.1)
            
            if valid_moves.item():  # Only use .item() for control flow
                # Cosine similarity between move1 and move2 (should be negative for opposite)
                similarity_12 = torch.cosine_similarity(move1.unsqueeze(0), move2.unsqueeze(0))
                
                # Check if we have a third movement
                if i + 2 < len(movements):
                    move3 = movements[i + 2]
                    move3_norm = torch.norm(move3)
                    
                    valid_third_move = move3_norm > 0.1
                    
                    if valid_third_move.item():
                        similarity_13 = torch.cosine_similarity(move1.unsqueeze(0), move3.unsqueeze(0))
                        
                        # Use continuous functions instead of discrete conditions for better gradients
                        # Strong oscillation penalty (smooth transition around thresholds)
                        strong_penalty = torch.sigmoid(-10 * (similarity_12 + 0.7)) * torch.sigmoid(10 * (similarity_13 - 0.7))
                        oscillation_score = oscillation_score + strong_penalty
                        
                        # Moderate oscillation penalty
                        moderate_penalty = torch.sigmoid(-5 * (similarity_12 + 0.3)) * torch.sigmoid(5 * (similarity_13 - 0.3)) * 0.5
                        oscillation_score = oscillation_score + moderate_penalty
        
        # Pattern 2: Stationary oscillation (staying near same position)
        position_variance = torch.var(positions, dim=0).mean()
        
        # Use continuous penalty instead of discrete threshold
        stationary_penalty = torch.sigmoid(-10 * (position_variance - 0.5)) * 0.3
        
        # Count non-zero movements with continuous approximation
        movement_magnitudes = torch.stack([torch.norm(move) for move in movements])
        significant_moves = torch.sigmoid(10 * (movement_magnitudes - 0.1))
        movement_activity = torch.sum(significant_moves)
        
        # Apply stationary penalty if there's sufficient movement activity
        movement_penalty = torch.sigmoid(movement_activity - 1.5) * stationary_penalty
        oscillation_score = oscillation_score + movement_penalty
        
        return oscillation_score
    
    def clear_old_position_history(self, max_batches_to_keep=5):
        """Clear old position history to prevent memory buildup"""
        if len(self.agent_position_history) > max_batches_to_keep * 10:  # Allow some buffer
            # Keep only recent batch histories
            sorted_keys = sorted(self.agent_position_history.keys())
            keys_to_remove = sorted_keys[:-max_batches_to_keep * 5]  # Keep recent ones
            
            for key in keys_to_remove:
                del self.agent_position_history[key]
            
            from debug_utils import debug_print
            debug_print(f"🧹 Cleared {len(keys_to_remove)} old position history entries")
    
    def setup_visualization(self):
        """Setup real-time training visualization with thread-safe GUI"""
        if not self.visualize_training:
            return
        
        print("🎬 Setting up real-time visualization...")
        
        try:
            # Ensure we're in the main thread for GUI operations
            import threading
            if threading.current_thread() is not threading.main_thread():
                print("⚠️  GUI setup called from background thread, deferring to main thread")
                return False
            
            # Use interactive mode for GUI
            plt.ion()
            
            # Close any existing figures to start fresh
            plt.close('all')
            
            # Create figure with subplots
            self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
            self.fig.suptitle("Enhanced MAPF Training - Real-time Monitoring", fontsize=16, fontweight='bold')
            
            # Training metrics plot
            self.axes[0, 0].set_title("Training Loss", fontweight='bold')
            self.axes[0, 0].set_xlabel("Epoch")
            self.axes[0, 0].set_ylabel("Loss")
            self.axes[0, 0].grid(True, alpha=0.3)
            
            # Communication metrics
            self.axes[0, 1].set_title("Communication & Coordination", fontweight='bold')
            self.axes[0, 1].set_xlabel("Epoch")
            self.axes[0, 1].set_ylabel("Score")
            self.axes[0, 1].grid(True, alpha=0.3)
            
            # Agents reached rate metrics
            self.axes[1, 0].set_title("Agents Reached Rate", fontweight='bold')
            self.axes[1, 0].set_xlabel("Epoch")
            self.axes[1, 0].set_ylabel("Agents Reached (%)")
            self.axes[1, 0].grid(True, alpha=0.3)
            
            # Episode visualization - 9x9 grid to match dataset
            self.axes[1, 1].set_title("Current Episode", fontweight='bold')
            self.axes[1, 1].set_xlim(-0.5, 8.5)
            self.axes[1, 1].set_ylim(-0.5, 8.5)
            self.axes[1, 1].set_aspect('equal')
            self.axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show(block=False)  # Non-blocking show
            plt.pause(0.1)  # Brief pause for GUI initialization
            
            # Verify the figure was created successfully
            if self.fig.number in plt.get_fignums():
                print("✅ Real-time visualization ready! (Thread-safe GUI mode)")
                return True
            else:
                print("❌ Failed to create visualization window")
                return False
                
        except Exception as e:
            print(f"❌ Visualization setup failed: {e}")
            debug_print(f"Visualization setup error: {e}")
            self.fig = None
            self.axes = None
            return False
    
    def update_visualization(self, epoch, batch_idx, episode_data=None):
        """Update visualization with current training status (thread-safe GUI)"""
        if not self.visualize_training:
            return
        
        try:
            # Ensure we're in the main thread for GUI operations
            import threading
            if threading.current_thread() is not threading.main_thread():
                print("⚠️  GUI update called from background thread, skipping")
                return
            
            # Check if figure exists
            if self.fig is None:
                print("⚠️  Visualization not initialized, recreating...")
                success = self.setup_visualization()
                if not success:
                    print("❌ Failed to recreate visualization, disabling...")
                    self.visualize_training = False
                    return
            
            # Check if figure window is still open
            if not plt.fignum_exists(self.fig.number):
                print("⚠️  Visualization window was closed, recreating...")
                success = self.setup_visualization()
                if not success:
                    print("❌ Failed to recreate visualization, disabling...")
                    self.visualize_training = False
                    return
            
            # Update training loss
            if self.training_history['epoch_losses']:
                epochs = range(len(self.training_history['epoch_losses']))
                self.axes[0, 0].clear()
                self.axes[0, 0].plot(epochs, self.training_history['epoch_losses'], 'b-', 
                                   label='Total Loss', linewidth=2, marker='o', markersize=4)
                self.axes[0, 0].set_title("Training Loss", fontsize=12, fontweight='bold')
                self.axes[0, 0].set_xlabel("Epoch")
                self.axes[0, 0].set_ylabel("Loss")
                self.axes[0, 0].legend()
                self.axes[0, 0].grid(True, alpha=0.3)
            
            # Update communication metrics
            if self.training_history['communication_losses']:
                epochs = range(len(self.training_history['communication_losses']))
                self.axes[0, 1].clear()
                self.axes[0, 1].plot(epochs, self.training_history['communication_losses'], 'g-', 
                                   label='Communication', linewidth=2, marker='s', markersize=4)
                if self.training_history['coordination_scores']:
                    self.axes[0, 1].plot(epochs, self.training_history['coordination_scores'], 'r-', 
                                       label='Coordination', linewidth=2, marker='^', markersize=4)
                if self.training_history['oscillation_penalties']:
                    self.axes[0, 1].plot(epochs, self.training_history['oscillation_penalties'], 'm--', 
                                       label='Oscillation Penalty', linewidth=2, marker='o', markersize=3)
                self.axes[0, 1].set_title("Communication & Coordination", fontsize=12, fontweight='bold')
                self.axes[0, 1].set_xlabel("Epoch")
                self.axes[0, 1].set_ylabel("Score")
                self.axes[0, 1].legend()
                self.axes[0, 1].grid(True, alpha=0.3)
            
            # Update validation metrics (only agents reached rate)
            if 'val_agents_reached_rates' in self.training_history and self.training_history['val_agents_reached_rates']:
                epochs = range(len(self.training_history['val_agents_reached_rates']))
                self.axes[1, 0].clear()
                self.axes[1, 0].plot(epochs, [r*100 for r in self.training_history['val_agents_reached_rates']], 'b-', 
                                   label='Agents Reached %', linewidth=2, marker='s', markersize=4)
                self.axes[1, 0].set_title("Agents Reached Rate", fontsize=12, fontweight='bold')
                self.axes[1, 0].set_xlabel("Epoch")
                self.axes[1, 0].set_ylabel("Agents Reached (%)")
                self.axes[1, 0].legend(loc='upper left')
                self.axes[1, 0].grid(True, alpha=0.3)
            
            # Process pending animations from queue (in main thread for matplotlib safety)
            self.process_pending_animations()
            
            # Update episode visualization ONLY if plot is not locked by animation
            if episode_data is not None and not getattr(self, 'plot_locked', False):
                self.axes[1, 1].clear()
                self.visualize_episode(episode_data, self.axes[1, 1])
            
            # Thread-safe GUI update
            try:
                self.fig.canvas.draw_idle()  # Use draw_idle for thread safety
                self.fig.canvas.flush_events()
                plt.pause(0.01)  # Very brief pause to allow GUI update
                print(f"📊 GUI updated (Epoch {epoch}, Batch {batch_idx})")
            except Exception as draw_error:
                print(f"⚠️  GUI update error: {draw_error}")
            
        except Exception as e:
            debug_print(f"Visualization update failed: {e}")
            print(f"❌ Visualization error: {e}")
            # Don't disable visualization on single update failure
    
    def visualize_episode(self, episode_data, ax):
        """Visualize episode with ACTUAL MODEL PREDICTIONS showing step-by-step agent movements"""
        
        print(f"🎬 Starting model-based episode visualization...")
        
        # Extract initial episode data
        initial_positions = episode_data.get('positions', [])
        agent_goals = episode_data.get('goals', [])
        obstacles = episode_data.get('obstacles', [])

        if len(initial_positions) == 0 or len(agent_goals) == 0:
            print("❌ No valid agent data for visualization")
            return
            
        # Convert to numpy arrays
        if isinstance(initial_positions, list):
            initial_positions = np.array(initial_positions)
        if isinstance(agent_goals, list):
            agent_goals = np.array(agent_goals)
            
        print(f"🎬 Running model predictions to simulate agent movements...")
        print(f"   Initial positions: {initial_positions}")
        print(f"   Goals: {agent_goals}")
        print(f"   Obstacles: {len(obstacles)} obstacles")
        
        # Setup visualization
        grid_size = self.grid_size
        ax.set_xlim(-0.5, grid_size-0.5)
        ax.set_ylim(-0.5, grid_size-0.5)
        ax.set_aspect('equal')
        
        # Draw grid
        for i in range(grid_size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1, alpha=0.3)
            ax.axvline(i - 0.5, color='black', linewidth=1, alpha=0.3)
        
        # Draw obstacles (with proper error handling)
        if obstacles is not None and len(obstacles) > 0:
            print(f"🔍 Drawing {len(obstacles)} obstacles: {obstacles}")
            for obs in obstacles:
                try:
                    # Handle different obstacle formats
                    if isinstance(obs, (list, tuple, np.ndarray)):
                        obs_x, obs_y = int(obs[0]), int(obs[1])
                    else:
                        print(f"⚠️ Unknown obstacle format: {obs} (type: {type(obs)})")
                        continue
                        
                    if 0 <= obs_x < grid_size and 0 <= obs_y < grid_size:
                        rect = patches.Rectangle([obs_x-0.5, obs_y-0.5], 1, 1, 
                                               linewidth=2, edgecolor='red', facecolor='darkred', alpha=0.7)
                        ax.add_patch(rect)
                        print(f"   ✅ Drew obstacle at ({obs_x}, {obs_y})")
                    else:
                        print(f"   ⚠️ Obstacle out of bounds: ({obs_x}, {obs_y})")
                except Exception as e:
                    print(f"   ❌ Error drawing obstacle {obs}: {e}")
        else:
            print("🔍 No obstacles to draw")
        
        # Colors for agents
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        
        # Draw goals first
        num_agents = len(initial_positions)
        for i in range(num_agents):
            color = colors[i % len(colors)]
            goal_x = max(0, min(grid_size-1, int(agent_goals[i][0])))
            goal_y = max(0, min(grid_size-1, int(agent_goals[i][1])))
            
            # Goal as star
            ax.plot(goal_x, goal_y, marker='*', markersize=15, color=color, 
                   markeredgecolor='black', markeredgewidth=1)
            ax.text(goal_x, goal_y-0.3, f'G{i}', ha='center', va='center', 
                   fontweight='bold', color=color, fontsize=8)
        
        # Now simulate agent movements using MODEL PREDICTIONS
        current_positions = initial_positions.copy()
        max_steps = 15  # Limit to prevent infinite loops
        
        print(f"🎬 Starting step-by-step model-based simulation...")
        
        for step in range(max_steps):
            print(f"🔥 Step {step}: Agent positions = {current_positions}")
            
            # Clear previous agent positions by removing marked patches and texts
            # Remove agent markers
            patches_to_remove = [p for p in ax.patches if hasattr(p, '_agent_marker') and p._agent_marker]
            for patch in patches_to_remove:
                patch.remove()
            
            # Remove agent texts  
            texts_to_remove = [t for t in ax.texts if hasattr(t, '_agent_text') and t._agent_text]
            for text in texts_to_remove:
                text.remove()
            
            # Draw current agent positions
            for i in range(num_agents):
                color = colors[i % len(colors)]
                agent_x = max(0, min(grid_size-1, int(current_positions[i][0])))
                agent_y = max(0, min(grid_size-1, int(current_positions[i][1])))
                
                # Agent as filled circle
                circle = patches.Circle([agent_x, agent_y], 0.3, 
                                      linewidth=2, edgecolor='black', facecolor=color, alpha=0.9)
                circle._agent_marker = True  # Mark for removal
                ax.add_patch(circle)
                
                # Agent number
                text = ax.text(agent_x, agent_y, str(i), ha='center', va='center',
                             fontweight='bold', color='white', fontsize=10)
                text._agent_text = True  # Mark for removal
            
            ax.set_title(f"🎬 Model Predictions - Step {step}", fontweight='bold', color='red')
            
            # Update display and wait
            try:
                if hasattr(self, 'fig') and self.fig.canvas is not None:
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
                    plt.pause(0.5)  # 0.5 second delay to see each step
                    print(f"   💫 Displayed step {step}, waiting 0.5s...")
                else:
                    print(f"   ⚠️ No canvas available for step {step}")
            except Exception as e:
                print(f"   ⚠️ Display error at step {step}: {e}")
            
            # Check if all agents reached goals
            all_reached = True
            for i in range(num_agents):
                agent_pos = current_positions[i]
                goal_pos = agent_goals[i]
                distance = np.linalg.norm(agent_pos - goal_pos)
                if distance > 0.5:  # Not at goal
                    all_reached = False
                    break
            
            if all_reached:
                print(f"🎯 All agents reached goals at step {step}!")
                break
            
            # Get model predictions and update positions
            try:
                # Create FOV data for current positions (simplified)
                # For now, we'll use a smart movement strategy towards goals
                new_positions = current_positions.copy()

                # Create obstacle set for fast lookup
                obstacle_set = set()
                if obstacles is not None and len(obstacles) > 0:
                    for obs in obstacles:
                        obstacle_set.add((int(obs[0]), int(obs[1])))

                for i in range(num_agents):
                    agent_pos = current_positions[i]
                    goal_pos = agent_goals[i]
                    # If agent already at goal (within threshold), stay in place
                    if np.linalg.norm(agent_pos - goal_pos) <= 0.5:
                        new_positions[i] = agent_pos
                        continue
                    
                    # Try multiple movement directions in order of preference
                    diff = goal_pos - agent_pos
                    
                    # List of possible moves in order of preference
                    possible_moves = []
                    
                    # Primary directions based on distance to goal
                    if abs(diff[0]) > abs(diff[1]):
                        # Horizontal movement preferred
                        if diff[0] > 0:
                            possible_moves.append((1, 0))  # Right
                        elif diff[0] < 0:
                            possible_moves.append((-1, 0))  # Left
                        
                        # Secondary direction
                        if diff[1] > 0:
                            possible_moves.append((0, 1))  # Up
                        elif diff[1] < 0:
                            possible_moves.append((0, -1))  # Down
                    else:
                        # Vertical movement preferred
                        if diff[1] > 0:
                            possible_moves.append((0, 1))  # Up
                        elif diff[1] < 0:
                            possible_moves.append((0, -1))  # Down
                        
                        # Secondary direction
                        if diff[0] > 0:
                            possible_moves.append((1, 0))  # Right
                        elif diff[0] < 0:
                            possible_moves.append((-1, 0))  # Left
                    
                    # Add all remaining directions as backup
                    all_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    for direction in all_directions:
                        if direction not in possible_moves:
                            possible_moves.append(direction)
                    
                    # Try each move in order of preference
                    moved = False
                    for dx, dy in possible_moves:
                        new_x = max(0, min(grid_size-1, int(agent_pos[0] + dx)))
                        new_y = max(0, min(grid_size-1, int(agent_pos[1] + dy)))
                        
                        # Check if position is valid (not obstacle, not agent collision)
                        if (new_x, new_y) not in obstacle_set:
                            # Check for agent collisions
                            collision = False
                            for j in range(num_agents):
                                if i != j:
                                    other_pos = current_positions[j]
                                    if int(other_pos[0]) == new_x and int(other_pos[1]) == new_y:
                                        collision = True
                                        break
                            
                            if not collision:
                                new_positions[i][0] = new_x
                                new_positions[i][1] = new_y
                                moved = True
                                break
                    
                    # If no movement possible, stay in place
                    if not moved:
                        new_positions[i] = agent_pos
                
                current_positions = new_positions
                
            except Exception as e:
                print(f"❌ Model prediction error at step {step}: {e}")
                break
        
        # Final display
        ax.set_title(f"🎬 Animation Complete - Final Positions", fontweight='bold', color='green')
        print(f"🎬 Model-based visualization complete!")
        
        return
        
    def _to_np(self, x):
        """Properly convert tensor or array to numpy array with debug info"""
        if isinstance(x, torch.Tensor):
            result = x.detach().cpu().numpy()
        else:
            result = np.asarray(x)
        
        # Debug: print shape information
        print(f"🔍 Debug _to_np: input_type={type(x)}, output_shape={result.shape}, output_dtype={result.dtype}")
        if result.size <= 10:  # Only print small arrays
            print(f"    Content: {result}")
        
        return result

    def print_initial_state_matrix(self, positions, goals, obstacles=None, case_idx=None, grid_size=9):
        """
        Print a matrix showing the initial state with:
        1 = free space, 2 = agent, 3 = obstacle, 4 = goal
        """
        # Initialize grid with free spaces (1s)
        grid = np.ones((grid_size, grid_size), dtype=int)
        
        # Place obstacles (3s)
        if obstacles is not None:
            try:
                obstacles_array = self._to_np(obstacles)
                if obstacles_array.size > 0:  # Use .size instead of len() to avoid ambiguity
                    if obstacles_array.ndim == 1:
                        obstacles_array = obstacles_array.reshape(1, -1)  # Convert 1D to 2D
                    
                    for obs in obstacles_array:
                        x, y = int(obs[0]), int(obs[1])
                        if 0 <= x < grid_size and 0 <= y < grid_size:
                            grid[y, x] = 3
            except Exception as e:
                print(f"⚠️ Error processing obstacles: {e}")
        
        # Place goals (4s) 
        if goals is not None:
            try:
                goals_array = self._to_np(goals)
                if goals_array.size > 0:  # Use .size instead of len() to avoid ambiguity
                    if goals_array.ndim == 1:
                        goals_array = goals_array.reshape(1, -1)  # Convert 1D to 2D
                    
                    for i, goal in enumerate(goals_array):
                        x, y = int(goal[0]), int(goal[1])
                        if 0 <= x < grid_size and 0 <= y < grid_size:
                            grid[y, x] = 4
            except Exception as e:
                print(f"⚠️ Error processing goals: {e}")
        
        # Place agents (2s) - agents override goals at same position
        if positions is not None:
            try:
                positions_array = self._to_np(positions)
                if positions_array.size > 0:  # Use .size instead of len() to avoid ambiguity
                    if positions_array.ndim == 1:
                        positions_array = positions_array.reshape(1, -1)  # Convert 1D to 2D
                    
                    for i, pos in enumerate(positions_array):
                        x, y = int(pos[0]), int(pos[1])
                        if 0 <= x < grid_size and 0 <= y < grid_size:
                            grid[y, x] = 2
            except Exception as e:
                print(f"⚠️ Error processing positions: {e}")
        
        # Print the matrix
        print(f"\n🗺️  INITIAL STATE MATRIX (Case {case_idx if case_idx is not None else 'Unknown'}):")
        print("   Legend: 1=Free, 2=Agent, 3=Obstacle, 4=Goal")
        print("   Grid coordinates (x=column, y=row):")
        print("     " + " ".join([f"{i:2d}" for i in range(grid_size)]))
        
        for y in range(grid_size):
            print(f"  {y:2d} " + " ".join([f"{grid[y, x]:2d}" for x in range(grid_size)]))
        
        # Print detailed position information
        print(f"\n📍 DETAILED POSITIONS:")
        if positions is not None:
            try:
                positions_array = self._to_np(positions)
                if positions_array.size > 0:
                    if positions_array.ndim == 1:
                        positions_array = positions_array.reshape(1, -1)
                    for i, pos in enumerate(positions_array):
                        x, y = int(pos[0]), int(pos[1])
                        print(f"   Agent {i}: ({x}, {y})")
            except Exception as e:
                print(f"   Error displaying agent positions: {e}")
        
        if goals is not None:
            try:
                goals_array = self._to_np(goals)
                if goals_array.size > 0:
                    if goals_array.ndim == 1:
                        goals_array = goals_array.reshape(1, -1)
                    for i, goal in enumerate(goals_array):
                        x, y = int(goal[0]), int(goal[1])
                        print(f"   Goal {i}: ({x}, {y})")
            except Exception as e:
                print(f"   Error displaying goals: {e}")
        
        if obstacles is not None:
            try:
                obstacles_array = self._to_np(obstacles)
                if obstacles_array.size > 0:
                    if obstacles_array.ndim == 1:
                        obstacles_array = obstacles_array.reshape(1, -1)
                    for i, obs in enumerate(obstacles_array):
                        x, y = int(obs[0]), int(obs[1])
                        print(f"   Obstacle {i}: ({x}, {y})")
            except Exception as e:
                print(f"   Error displaying obstacles: {e}")
        else:
            print("   No obstacles")
        print()

    def setup_animation_queue(self):
        """Setup animation queue system for showing each episode only once as training metrics"""
        self.animation_queue = queue.Queue(maxsize=50)  # Queue for one-time episode processing
        self.gui_command_queue = queue.Queue(maxsize=50)  # Thread-safe command queue for GUI operations
        self.episodes_shown = set()  # Track episodes that have been shown already
        self.animation_active = False
        self.animation_thread = None
        self.plot_locked = False  # Lock plot during animations
        print("🎬 Animation queue system initialized (show-once episode tracking for training metrics)")
        
    def queue_animation(self, episode_data, case_idx, epoch, batch_idx):
        """Add animation to queue - each episode shown only once as training metric"""
        if not hasattr(self, 'episodes_shown'):
            self.setup_animation_queue()
            
        # Create unique episode identifier
        episode_id = f"epoch_{epoch}_case_{case_idx}_batch_{batch_idx}"
        
        # Check if this episode has already been shown
        if episode_id in self.episodes_shown:
            print(f"🔄 Episode {episode_id} already shown, skipping to avoid repetition")
            return
            
        # Print initial state matrix before queuing
        positions = episode_data.get('positions', [])
        goals = episode_data.get('goals', [])
        obstacles = episode_data.get('obstacles', [])
        
        self.print_initial_state_matrix(positions, goals, obstacles, case_idx)
        
        # Create animation data
        animation_data = {
            'episode_data': episode_data,
            'case_idx': case_idx,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'episode_id': episode_id,
            'timestamp': time.time()
        }
        
        # Queue for immediate processing (show once)
        try:
            self.animation_queue.put(animation_data, timeout=0.1)
            print(f"🎬 Queued NEW episode {episode_id} for one-time display")
            
            # Start animation worker if not running
            if not self.animation_active:
                self.start_show_once_animation_worker()
                
        except queue.Full:
            print(f"⚠️  Animation queue full, skipping episode {episode_id}")
        
        # Mark this episode as shown to prevent future repetition
        self.episodes_shown.add(episode_id)
    
    def start_show_once_animation_worker(self):
        """Start the show-once animation worker thread"""
        if self.animation_active:
            return
            
        self.animation_active = True
        self.animation_thread = threading.Thread(target=self.show_once_animation_worker)
        self.animation_thread.daemon = True
        self.animation_thread.start()
        print("🎬 Started show-once animation worker (each episode displayed once)")
        
    def show_once_animation_worker(self):
        """Worker thread that processes each episode once as training progresses (thread-safe)"""
        print("🎬 Show-once animation worker started - displaying each episode once as training metric")
        
        while self.animation_active:
            try:
                # Get next episode from queue (blocking with timeout)
                try:
                    animation_data = self.animation_queue.get(timeout=2.0)
                except queue.Empty:
                    # No episodes to show, keep waiting
                    continue
                
                episode_id = animation_data.get('episode_id', 'unknown')
                case_idx = animation_data.get('case_idx', '?')
                epoch = animation_data.get('epoch', '?')
                
                print(f"🎬 Processing NEW episode {episode_id} (case {case_idx}, epoch {epoch})")
                
                # Lock the plot during processing
                self.plot_locked = True
                
                # Send animation command to main thread via queue (thread-safe)
                gui_command = {
                    'action': 'animate_case',
                    'data': animation_data,
                    'is_last': True  # Each episode is independent
                }
                try:
                    self.gui_command_queue.put(gui_command, timeout=2.0)
                    print(f"✅ Sent animation command to main thread for episode {episode_id}")
                except queue.Full:
                    print(f"⚠️  GUI command queue full, skipping animation for episode {episode_id}")
                    self.plot_locked = False
                    continue
                
                # Send unlock command to main thread after animation
                unlock_command = {
                    'action': 'unlock_and_restore'
                }
                try:
                    self.gui_command_queue.put(unlock_command, timeout=1.0)
                    print(f"✅ Sent unlock command to main thread")
                except queue.Full:
                    print(f"⚠️  GUI command queue full, plot may remain locked")
                    self.plot_locked = False
                
                print(f"✅ Completed one-time display of episode {episode_id}")
                
                # Mark task as done
                self.animation_queue.task_done()
                
                # Brief pause before next episode
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Animation worker error: {e}")
                self.plot_locked = False
                time.sleep(1.0)
        
        print("🎬 Show-once animation worker stopped")
        self.animation_active = False
        self.plot_locked = False
    
    def process_single_animation(self, animation_data):
        """Process a single animation from the queue"""
        episode_data = animation_data['episode_data']
        case_idx = animation_data['case_idx']
        epoch = animation_data['epoch']
        batch_idx = animation_data['batch_idx']
        
        print(f"🎬 Showing initial state for case {case_idx} (Epoch {epoch}, Batch {batch_idx})")
        
        try:
            # Create a dedicated figure for this animation
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            fig.suptitle(f"Case {case_idx} Initial State - Epoch {epoch}, Batch {batch_idx}", 
                        fontsize=14, fontweight='bold')
            
            # Show only the initial state (no animation)
            self.visualize_initial_state(episode_data, ax, case_idx)
            
            # Display and keep window open for viewing
            plt.show(block=False)  # Non-blocking show
            plt.pause(2.0)  # Show for 2 seconds
            
            # Save the figure if needed
            if hasattr(self, 'save_visualizations') and self.save_visualizations:
                save_path = f"visualization_case_{case_idx}_epoch_{epoch}_batch_{batch_idx}.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"💾 Saved visualization to {save_path}")
            
            plt.close(fig)  # Close to prevent memory buildup
            
        except Exception as e:
            print(f"❌ Animation processing failed: {e}")
            debug_print(f"Animation error details: {e}")
        
    def visualize_initial_state(self, episode_data, ax, case_idx=None):
        """Visualize the initial state of an episode"""
        agent_positions = episode_data.get('positions', [])
        agent_goals = episode_data.get('goals', [])
        obstacles = episode_data.get('obstacles', [])
        
        # Use the same visualization as visualize_episode but without paths
        self.visualize_episode(episode_data, ax)
    
    def stop_animation_worker(self):
        """Stop the animation worker thread"""
        if hasattr(self, 'animation_active'):
            self.animation_active = False
            if hasattr(self, 'animation_thread') and self.animation_thread:
                self.animation_thread.join(timeout=2.0)
                print("🛑 Animation worker stopped")
        
        # Clear any pending GUI commands
        if hasattr(self, 'gui_command_queue'):
            try:
                while True:
                    self.gui_command_queue.get_nowait()
            except queue.Empty:
                pass
            print("🧹 Cleared pending GUI commands")
    
    def play_single_animation_seamlessly(self, animation_data, is_last=False):
        """Play a single animation seamlessly using the existing plot (main thread only)"""
        # Ensure we're in the main thread for GUI operations
        import threading
        if threading.current_thread() is not threading.main_thread():
            print("⚠️  GUI animation called from background thread, skipping")
            return
        
        try:
            episode_data = animation_data['episode_data']
            case_idx = animation_data['case_idx']
            epoch = animation_data['epoch']
            
            # DEBUG: Print the actual animation data being processed
            print(f"🔍 DEBUG ANIMATION DATA for epoch {epoch}, case {case_idx}:")
            positions = episode_data.get('positions', [])
            goals = episode_data.get('goals', [])
            obstacles = episode_data.get('obstacles', [])
            
            print(f"   Positions: {type(positions)}, shape: {getattr(positions, 'shape', 'no shape')}")
            if hasattr(positions, '__len__') and len(positions) > 0:
                print(f"   Position values: {positions}")
            else:
                print(f"   Position values: EMPTY OR INVALID")
                
            print(f"   Goals: {type(goals)}, shape: {getattr(goals, 'shape', 'no shape')}")
            if hasattr(goals, '__len__') and len(goals) > 0:
                print(f"   Goal values: {goals}")
            else:
                print(f"   Goal values: EMPTY OR INVALID")
                
            print(f"   Obstacles: {type(obstacles)}, length: {len(obstacles) if hasattr(obstacles, '__len__') else 'no len'}")
            if hasattr(obstacles, '__len__') and len(obstacles) > 0:
                print(f"   Obstacle values: {obstacles}")
            else:
                print(f"   Obstacle values: EMPTY")
            
            # Check if we have the visualization figure available and it's still valid
            if not hasattr(self, 'fig') or self.fig is None:
                print(f"❌ No visualization figure available for seamless animation")
                return
            
            # Check if figure window is still open
            if not plt.fignum_exists(self.fig.number):
                print(f"❌ Visualization window was closed, cannot animate")
                return
            
            # Use the existing episode plot (axes[1, 1])
            if not hasattr(self, 'axes') or self.axes is None:
                print(f"❌ No axes available for animation")
                return
                
            ax = self.axes[1, 1]
            
            # Clear and set up the plot with animation indicators
            ax.clear()
            print(f"🎬 Cleared axis for animation")
            
            
            
            # Add visual indicator that this is an animation
            try:
                self.fig.patch.set_edgecolor('red')
                self.fig.patch.set_linewidth(3)
                
                # Set special title to indicate animation mode
                self.fig.suptitle(f"🎬 ANIMATION: Epoch {epoch}, Case {case_idx}", fontsize=16, fontweight='bold', color='red')
                print(f"🎬 Set animation title")
                
                # Visualize the episode with model predictions
                print(f"🎬 About to call visualize_episode...")
                self.visualize_episode(episode_data, ax)
                print(f"🎬 Called visualize_episode successfully")
                
                # Thread-safe display update with error handling
                if self.fig.canvas is not None:
                    print(f"🎬 Frame-by-frame animation completed")
                    print(f"🎬 Played animation for epoch {epoch}, case {case_idx}")
                else:
                    print(f"⚠️  Figure canvas is None, skipping animation display")
                    
            except Exception as draw_error:
                print(f"⚠️  Animation display update failed: {draw_error}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"❌ Seamless animation failed: {e}")
            import traceback
            traceback.print_exc()
            # Reset plot lock on error
            self.plot_locked = False
            
    def unlock_plot_and_restore(self):
        """Unlock the plot and restore normal appearance after animation sequence (main thread only)"""
        # Ensure we're in the main thread for GUI operations
        import threading
        if threading.current_thread() is not threading.main_thread():
            print("⚠️  GUI unlock called from background thread, skipping")
            return
        
        try:
            self.plot_locked = False
            
            # Restore normal plot appearance if figure exists and is still valid
            if hasattr(self, 'fig') and self.fig is not None and plt.fignum_exists(self.fig.number):
                try:
                    # Remove red border
                    self.fig.patch.set_edgecolor('black')
                    self.fig.patch.set_linewidth(1)
                    
                    # Restore normal title
                    self.fig.suptitle("Enhanced MAPF Training - Real-time Monitoring", fontsize=16, fontweight='bold', color='black')
                    
                    # Clear the episode plot to remove animation residue
                    if hasattr(self, 'axes') and self.axes is not None:
                        self.axes[1, 1].clear()
                        self.axes[1, 1].set_title("Current Episode", fontweight='bold')
                        self.axes[1, 1].set_xlim(-0.5, 8.5)
                        self.axes[1, 1].set_ylim(-0.5, 8.5)
                        self.axes[1, 1].set_aspect('equal')
                        self.axes[1, 1].grid(True, alpha=0.3)
                        self.axes[1, 1].text(0.5, 0.5, "Waiting for next episode...", 
                                            ha='center', va='center', transform=self.axes[1, 1].transAxes,
                                            fontsize=12, style='italic', color='gray')
                    
                    # Thread-safe display update with better error handling
                    if self.fig.canvas is not None:
                        self.fig.canvas.draw_idle()  # Use draw_idle for thread safety
                        self.fig.canvas.flush_events()
                    
                except Exception as restore_error:
                    print(f"⚠️ Plot restore display update failed: {restore_error}")
            
            print("🔓 Plot unlocked and restored to normal appearance")
            
        except Exception as e:
            print(f"❌ Plot unlock/restore failed: {e}")
            # Ensure plot is unlocked even if display updates fail
            self.plot_locked = False

    def process_pending_animations(self):
        """Process GUI commands from animation worker thread (called in main thread)"""
        if not hasattr(self, 'gui_command_queue'):
            return
        
        try:
            # Process all pending GUI commands
            commands_processed = 0
            while True:
                try:
                    command = self.gui_command_queue.get_nowait()
                    action = command.get('action')
                    commands_processed += 1
                    
                    if action == 'animate_case':
                        # Execute animation in main thread (thread-safe)
                        animation_data = command.get('data')
                        is_last = command.get('is_last', False)
                        print(f"🎬 Processing animation command in main thread (case {animation_data.get('case_idx', '?')})")
                        self.play_single_animation_seamlessly(animation_data, is_last)
                        
                    elif action == 'unlock_and_restore':
                        # Execute unlock and restore in main thread (thread-safe)
                        print(f"🔓 Processing unlock command in main thread")
                        self.unlock_plot_and_restore()
                        
                    else:
                        print(f"⚠️  Unknown GUI command: {action}")
                        
                except queue.Empty:
                    break  # No more commands to process
            
            if commands_processed > 0:
                print(f"✅ Processed {commands_processed} GUI commands in main thread")
                
        except Exception as e:
            print(f"❌ Error processing pending animations: {e}")
            import traceback
            traceback.print_exc()
                    
        except Exception as e:
            print(f"⚠️  Error processing GUI commands: {e}")
            # Don't crash on GUI command errors
    
    def comprehensive_evaluate(self, max_cases=50):
        """Comprehensive evaluation on validation set"""
        # Import debug utilities at the beginning
        from debug_utils import debug_print
        
        if not hasattr(self, 'model') or self.model is None:
            print("❌ No model available for evaluation")
            return {}
        
        print(f"🧪 Running comprehensive evaluation on up to {max_cases} cases...")
        
        # Create validation data
        try:
            val_fovs, val_actions, val_positions, val_goals, val_obstacles = self.create_sequence_dataset(mode="valid")
            if len(val_fovs) == 0:
                print("❌ No validation data available")
                return {}
        except Exception as e:
            print(f"❌ Failed to create validation dataset: {e}")
            return {}
        
        # Limit to max_cases
        num_cases = min(len(val_fovs), max_cases)
        val_fovs = val_fovs[:num_cases]
        val_actions = val_actions[:num_cases]
        val_positions = val_positions[:num_cases]
        val_goals = val_goals[:num_cases]
        val_obstacles = val_obstacles[:num_cases]
        
        self.model.eval()
        total_correct = 0
        total_predictions = 0
        total_agents_reached = 0
        total_agents = 0
        total_flow_time = 0.0
        total_episodes = 0
        episode_success_count = 0
        
        with torch.no_grad():
            for i in range(num_cases):
                sequence_fov = val_fovs[i:i+1].to(self.device)  # [1, seq_len, agents, channels, h, w]
                sequence_actions = val_actions[i:i+1].to(self.device)  # [1, seq_len, agents]
                sequence_positions = val_positions[i:i+1].to(self.device)  # [1, agents, 2] - 3D tensor, no seq_len
                sequence_goals = val_goals[i:i+1].to(self.device)  # [1, agents, 2] - 3D tensor, no seq_len
                case_obstacles = val_obstacles[i] if i < len(val_obstacles) else []  # Get obstacles for this case
                
                seq_len = sequence_actions.shape[1]
                actions_num_agents = sequence_actions.shape[2]
                positions_num_agents = sequence_positions.shape[1]  # Shape is [1, agents, 2]
                goals_num_agents = sequence_goals.shape[1]  # Shape is [1, agents, 2]
                
                # Use the minimum number of agents to ensure consistent indexing
                num_agents = min(actions_num_agents, positions_num_agents, goals_num_agents)
                
                if actions_num_agents != positions_num_agents or positions_num_agents != goals_num_agents:
                    debug_print(f"⚠️ Agent count mismatch in case {i}: actions={actions_num_agents}, positions={positions_num_agents}, goals={goals_num_agents}. Using min={num_agents}")
                
                # Reset SNN state for each case
                functional.reset_net(self.model)
                
                case_correct = 0
                case_total = 0
                agents_reached_goal = 0
                agent_flow_times = []
                
                # Print initial state matrix for this validation case (debug only)
                if i < 5:  # Print for first 5 cases only to avoid spam
                    initial_positions = sequence_positions[0].cpu().numpy()  # [agents, 2] - no extra index
                    initial_goals = sequence_goals[0].cpu().numpy()  # [agents, 2] - no extra index
                    debug_print("Initial state matrix for validation case {i}")
                    # Move detailed matrix printing to debug mode
                    # self.print_initial_state_matrix(initial_positions, initial_goals, case_obstacles, case_idx=i)
                
                # Track each agent's path to goal and collision status
                agent_reached_times = [-1] * num_agents  # -1 means not reached
                agent_collided = [False] * num_agents  # Track which agents have collided
                current_agent_positions = sequence_positions[0].clone()  # [agents, 2] - start with initial positions
                
                # Import collision detection utility
                from collision_utils import detect_collisions
                
                for t in range(seq_len):
                    fov_t = sequence_fov[:, t]  # [1, agents, channels, h, w]
                    output = self.model(fov_t)
                    
                    # Handle different output shapes
                    debug_print(f"Model output shape: {output.shape}, num_agents: {num_agents}")
                    
                    if output.dim() == 1:
                        # If output is 1D, reshape based on actual number of agents in the batch
                        actual_agents = output.numel() // 5  # Assuming 5 action classes
                        if actual_agents != num_agents:
                            debug_print(f"Agent count mismatch: expected {num_agents}, got {actual_agents}")
                            num_agents = min(num_agents, actual_agents)  # Use the smaller count
                        output = output.view(1, actual_agents, 5)  # [1, agents, action_classes]
                    elif output.dim() == 2:
                        # If output is 2D [batch, features], reshape appropriately
                        batch_size, features = output.shape
                        if features % 5 == 0:  # Assuming 5 action classes
                            actual_agents = features // 5
                            if actual_agents != num_agents:
                                debug_print(f"Agent count mismatch: expected {num_agents}, got {actual_agents}")
                                num_agents = min(num_agents, actual_agents)
                            output = output.view(batch_size, actual_agents, 5)
                        else:
                            # Fallback: use original approach but with safety check
                            if output.numel() % num_agents == 0:
                                output = output.view(1, num_agents, -1)
                            else:
                                debug_print(f"❌ Cannot reshape output of size {output.numel()} for {num_agents} agents")
                                continue
                    else:
                        # If already 3D or higher, try to reshape
                        try:
                            output = output.view(1, num_agents, -1)  # [1, agents, action_classes]
                        except RuntimeError as e:
                            debug_print(f"❌ Reshape error: {e}")
                            continue
                    
                    # Get predicted actions and simulate movement for collision detection
                    predicted_actions = torch.argmax(output[0], dim=1)  # [num_agents]
                    
                    # Update positions based on predicted actions for collision detection
                    prev_positions = current_agent_positions.clone()
                    action_deltas = torch.tensor([
                        [0, 0],   # 0: Stay
                        [1, 0],   # 1: Right  
                        [0, 1],   # 2: Up
                        [-1, 0],  # 3: Left
                        [0, -1]   # 4: Down
                    ], dtype=torch.float32, device=current_agent_positions.device)
                    
                    # Apply actions to update positions
                    for agent_idx in range(num_agents):
                        if agent_idx < len(predicted_actions):
                            action = predicted_actions[agent_idx].item()
                            if action < len(action_deltas):
                                delta = action_deltas[action]
                                new_pos = current_agent_positions[agent_idx] + delta
                                # Clamp to board boundaries
                                new_pos[0] = torch.clamp(new_pos[0], 0, 8)  # Assuming 9x9 board (0-8)
                                new_pos[1] = torch.clamp(new_pos[1], 0, 8)
                                current_agent_positions[agent_idx] = new_pos
                    
                    # Detect collisions at this timestep
                    try:
                        collisions = detect_collisions(
                            current_agent_positions.unsqueeze(0),  # Add batch dimension
                            prev_positions.unsqueeze(0)
                        )
                        
                        # Mark agents that collided (vertex or edge collisions)
                        vertex_collisions = collisions['vertex_collisions'][0]  # Remove batch dimension
                        edge_collisions = collisions['edge_collisions'][0]
                        
                        for agent_idx in range(num_agents):
                            if vertex_collisions[agent_idx] > 0 or edge_collisions[agent_idx] > 0:
                                agent_collided[agent_idx] = True
                                
                    except Exception as e:
                        debug_print(f"Collision detection failed at timestep {t}: {e}")
                    
                    # Get current positions and goals (constant throughout sequence)
                    current_positions = sequence_positions[0].cpu().numpy()  # [agents, 2] - no timestep index
                    current_goals = sequence_goals[0].cpu().numpy()  # [agents, 2] - no timestep index
                    
                    # Check predictions and goal reaching (only for non-collided agents)
                    for agent in range(num_agents):
                        # Bounds check for actions
                        if agent >= sequence_actions.shape[2]:
                            continue
                            
                        predicted = torch.argmax(output[0, agent])
                        true_action = sequence_actions[0, t, agent]
                        if predicted == true_action:
                            case_correct += 1
                        case_total += 1
                        
                        # Check if agent reached goal for the first time (only if not collided)
                        if agent_reached_times[agent] == -1 and not agent_collided[agent]:  # Not reached yet and not collided
                            # Bounds check for positions and goals
                            if agent >= len(current_positions) or agent >= len(current_goals):
                                continue
                                
                            agent_pos = current_positions[agent]
                            agent_goal = current_goals[agent]
                            # Check if agent is at goal position (within tolerance)
                            distance_to_goal = np.linalg.norm(agent_pos - agent_goal)
                            if distance_to_goal < 0.5:  # Agent reached goal
                                agent_reached_times[agent] = t
                                agents_reached_goal += 1
                                debug_print(f"Agent {agent} reached goal at timestep {t} in case {i}")
                
                # Calculate flow times for agents that reached their goals
                for agent in range(num_agents):
                    if agent_reached_times[agent] != -1:
                        agent_flow_times.append(agent_reached_times[agent] + 1)  # +1 because timestep is 0-indexed
                    else:
                        # Agent didn't reach goal, use maximum time as penalty
                        agent_flow_times.append(seq_len)
                
                # Calculate effective agents (exclude collided agents from total count)
                effective_agents = num_agents - sum(agent_collided)
                collided_agents = sum(agent_collided)
                
                # Episode metrics
                total_correct += case_correct
                total_predictions += case_total
                total_agents_reached += agents_reached_goal
                total_agents += effective_agents  # Use effective agents instead of all agents
                
                # Debug info for collision tracking (keep this as it's performance-related)
                if collided_agents > 0:
                    print(f"🚫 Case {i}: {collided_agents}/{num_agents} agents collided, effective agents: {effective_agents}")
                
                # Flow time calculation
                episode_avg_flow_time = sum(agent_flow_times) / len(agent_flow_times) if agent_flow_times else seq_len
                total_flow_time += episode_avg_flow_time
                total_episodes += 1
                
                # Episode success (all non-collided agents reached their goals)
                if agents_reached_goal == effective_agents and effective_agents > 0:
                    episode_success_count += 1
                
                # Create episode data for potential visualization (debug only)
                if i < 3:  # Visualize first 3 validation cases
                    episode_data = {
                        'positions': sequence_positions[0].cpu().numpy(),  # Initial positions - no extra index
                        'goals': sequence_goals[0].cpu().numpy(),  # Initial goals - no extra index
                        'obstacles': case_obstacles,  # Use real obstacle data
                        'current_timestep': 0,
                        'show_full_paths': False
                    }
                    debug_print(f"🔍 Validation case {i} episode data created with {len(case_obstacles)} obstacles")
                    debug_print(f"    Agents reached goal: {agents_reached_goal}/{num_agents}, Avg flow time: {episode_avg_flow_time:.2f}")
        
        # Calculate comprehensive metrics
        action_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
        agents_reached_rate = total_agents_reached / total_agents if total_agents > 0 else 0.0
        episode_success_rate = episode_success_count / total_episodes if total_episodes > 0 else 0.0
        avg_flow_time = total_flow_time / total_episodes if total_episodes > 0 else 0.0
        
        metrics = {
            'action_accuracy': action_accuracy,  # How well the model predicts actions
            'agents_reached_rate': agents_reached_rate,  # Percentage of agents that reached goals
            'episode_success_rate': episode_success_rate,  # Percentage of episodes where all agents reached goals
            'avg_flow_time': avg_flow_time,  # Average time for agents to reach goals
            'total_correct': total_correct,
            'total_predictions': total_predictions,
            'total_agents_reached': total_agents_reached,
            'total_agents': total_agents,
            'episode_success_count': episode_success_count,
            'total_episodes': total_episodes,
            'cases_evaluated': num_cases
        };
        
        print(f"✅ Evaluation complete:")
        print(f"   🎯 Action Accuracy: {action_accuracy:.3f}")
        print(f"   🏁 Agents Reached Rate: {agents_reached_rate:.3f} ({total_agents_reached}/{total_agents})")
        print(f"   ⏱️  Average Flow Time: {avg_flow_time:.2f} timesteps")
        if total_agents != total_agents_reached + (total_episodes * 5 - total_agents):  # If collisions occurred
            collision_rate = 1 - (total_agents / (total_episodes * 5)) if total_episodes > 0 else 0
            print(f"   🚫 Collision Rate: {collision_rate:.3f}")
        debug_print(f"   Episode Success Rate: {episode_success_rate:.3f}")
        debug_print(f"   Episodes: {episode_success_count}/{total_episodes} successful")
        debug_print(f"   Cases evaluated: {num_cases}")
        return metrics
    
    def train(self):
        """Main training loop with enhanced features"""
        print("🚀 Starting Enhanced MAPF Training...")
        
        # Create sequence dataset
        try:
            print("📊 Creating training dataset...")
            train_fovs, train_actions, train_positions, train_goals, train_obstacles = self.create_sequence_dataset(mode="train")
            
            # Apply data augmentation
            if len(train_fovs) > 0:
                train_fovs, train_actions, train_positions, train_goals = self.augment_dataset(
                    train_fovs, train_actions, train_positions, train_goals, augment_factor=3
                )
                # Note: obstacles are not augmented as they stay the same regardless of rotation/flip
                # We need to replicate them to match the augmented data size
                augment_factor = 3
                original_size = len(train_obstacles)
                train_obstacles = train_obstacles * augment_factor  # Replicate obstacle data
            
            print(f"✅ Training dataset ready: {len(train_fovs)} sequences")
            
        except Exception as e:
            print(f"❌ Failed to create training dataset: {e}")
            return
        
        if len(train_fovs) == 0:
            print("❌ No training data available!")
            return
        
        # Create data loader
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(train_fovs, train_actions, train_positions, train_goals)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Store obstacles separately since they can't go in TensorDataset (variable length)
        self.train_obstacles = train_obstacles
        
        print(f"📈 Training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            print(f"\n🔄 Epoch {epoch+1}/{self.num_epochs}")
            
            # Calculate entropy bonus weight for this epoch
            use_entropy_bonus = self.config.get('use_entropy_bonus', False)
            entropy_bonus_weight = 0.0
            if use_entropy_bonus:
                initial_entropy_weight = self.config.get('entropy_bonus_weight', 0.02)
                bonus_epochs = self.config.get('entropy_bonus_epochs', 15)
                decay_type = self.config.get('entropy_bonus_decay_type', 'linear')
                entropy_bonus_weight = entropy_bonus_schedule(epoch, initial_entropy_weight, bonus_epochs, decay_type)
            
            self.model.train()
            epoch_loss = 0.0
            epoch_comm_loss = 0.0
            epoch_coord_score = 0.0
            epoch_entropy_bonus = 0.0  # Track entropy bonus
            epoch_progress_reward = 0.0  # Track progress reward
            epoch_oscillation_penalty = 0.0  # Track oscillation penalty
            # Adjacent coordination ratio no longer tracked in simplified implementation
            num_batches = 0
            
            for batch_idx, (batch_fovs, batch_actions, batch_positions, batch_goals) in enumerate(train_loader):
                batch_fovs = batch_fovs.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_positions = batch_positions.to(self.device)
                batch_goals = batch_goals.to(self.device)
                
                # Get corresponding obstacles for this batch
                # Note: We need to map batch indices to original sequence indices
                batch_start_idx = batch_idx * self.batch_size
                batch_end_idx = min(batch_start_idx + self.batch_size, len(self.train_obstacles))
                batch_obstacles = self.train_obstacles[batch_start_idx:batch_end_idx]
                
                # Reset SNN state
                functional.reset_net(self.model)
                
                batch_loss = 0.0
                sequence_outputs = []
                
                seq_len = batch_fovs.shape[1]
                
                # Track positions throughout sequence for progress reward calculation
                # Initialize with batch_positions (starting positions)
                current_batch_positions = batch_positions.clone()  # [batch_size, num_agents, 2]
                previous_batch_positions = None  # Will be set after first timestep
                
                # Initialize best distances for progress reward at start of each batch
                if self.config.get('use_goal_progress_reward', False):
                    # Initialize best distances to current distances to goals
                    initial_distances = torch.norm(current_batch_positions - batch_goals, dim=2)
                    self.batch_best_distances = initial_distances.clone()  # [batch_size, num_agents]
                
                for t in range(seq_len):
                    fov_t = batch_fovs[:, t]  # [batch, agents, channels, h, w]
                    
                    # Forward pass
                    output = self.model(fov_t)
                    output = output.view(batch_fovs.shape[0], -1, self.model.num_classes)
                    sequence_outputs.append(output)
                    
                    # Compute loss for this timestep
                    targets = batch_actions[:, t]  # [batch, agents]
                    
                    # Ensure targets are Long type for CrossEntropy loss
                    targets = targets.long()
                    
                    timestep_loss = 0.0
                    for agent in range(targets.shape[1]):
                        agent_loss = self.criterion(output[:, agent], targets[:, agent])
                        timestep_loss += agent_loss
                    
                    timestep_loss /= targets.shape[1]  # Average over agents
                    
                    # Add entropy bonus for exploration (encourage diverse actions)
                    entropy_bonus = 0.0
                    if entropy_bonus_weight > 0:
                        # Flatten logits for all agents at this timestep
                        logits_flat = output.view(-1, self.model.num_classes)
                        entropy_bonus = compute_entropy_bonus(logits_flat)
                        timestep_loss -= entropy_bonus_weight * entropy_bonus  # Subtract to encourage high entropy
                        epoch_entropy_bonus += entropy_bonus.item()
                    
                    batch_loss += timestep_loss
                    
                    # Update position tracking for proximity reward
                    if t > 0:  # Skip first timestep (no previous position)
                        previous_batch_positions = current_batch_positions.clone()
                        # Simulate position updates based on actions (simplified)
                        # In a real implementation, you'd use actual position reconstruction
                        # For now, we'll use this as a placeholder for position tracking
                        current_batch_positions = self._update_positions_from_actions(
                            current_batch_positions, targets
                        )
                        
                        # Update position history INSIDE the loop to track evolving positions
                        self.update_agent_position_history(current_batch_positions, batch_id=batch_idx)
                
                # Average loss over time
                batch_loss /= seq_len
                
                # Compute additional losses
                sequence_outputs_tensor = torch.stack(sequence_outputs, dim=1)  # [batch, seq_len, agents, action_dim]
                
                # Communication loss
                comm_loss = self.compute_communication_loss(sequence_outputs_tensor[:, -1], batch_positions)
                
                # Coordination loss with adjacent coordination enhancement
                coord_loss = self.compute_coordination_loss(sequence_outputs_tensor[:, -1], batch_positions)
                
                # Temporal consistency loss
                temporal_loss = self.compute_temporal_consistency_loss(sequence_outputs_tensor)
                
                # Detect oscillation patterns and compute penalty (history already updated in loop)
                oscillation_penalty = self.detect_oscillation_patterns(batch_id=batch_idx)
                
                # Goal progress reward (best-distance-so-far system)
                progress_reward = 0.0
                if self.config.get('use_goal_progress_reward', False):
                    from collision_utils import compute_best_distance_progress_reward
                    
                    # Get progress reward parameters from config
                    goal_progress_weight = float(self.config.get('goal_progress_weight', 1e-3))
                    success_threshold = self.config.get('goal_success_threshold', 1.0)
                    
                    # Compute progress reward with best-distance tracking
                    # best_distances should already be initialized at batch start
                    progress_reward_tensor, self.batch_best_distances = compute_best_distance_progress_reward(
                        current_batch_positions, batch_goals, 
                        self.batch_best_distances,
                        progress_weight=goal_progress_weight, 
                        success_threshold=success_threshold
                    )
                    # Convert tensor to scalar
                    progress_reward = progress_reward_tensor.item()
                    # Accumulate progress reward for epoch tracking
                    epoch_progress_reward += progress_reward
                
                # Total loss (subtract progress reward since it should reduce loss)
                total_loss = (batch_loss + 
                             self.communication_weight * comm_loss +
                             self.coordination_weight * coord_loss +
                             self.temporal_consistency_weight * temporal_loss +
                             oscillation_penalty -  # Add oscillation penalty (learnable weight already applied)
                             progress_reward)
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_comm_loss += comm_loss.item()
                epoch_coord_score += (1.0 - coord_loss.item())  # Higher is better for coordination
                epoch_oscillation_penalty += oscillation_penalty.item() if isinstance(oscillation_penalty, torch.Tensor) else 0.0
                # Adjacent coordination ratio no longer tracked in simplified implementation
                num_batches += 1
                
                # Update visualization
                if self.visualize_training and batch_idx % self.vis_update_freq == 0:
                    # Create episode data for visualization
                    episode_data = {
                        'positions': batch_positions[0].cpu().numpy(),
                        'goals': batch_goals[0].cpu().numpy(),
                        'obstacles': batch_obstacles[0] if batch_obstacles else [],  # Use real obstacle data
                        'current_timestep': 0,
                        'show_full_paths': False
                    }
                    
                    # Queue animation
                    self.queue_animation(episode_data, case_idx=batch_idx, epoch=epoch, batch_idx=batch_idx)
                    
                    # Update visualization
                    self.update_visualization(epoch, batch_idx, episode_data)
            
            # Compute epoch averages
            avg_loss = epoch_loss / num_batches
            avg_comm_loss = epoch_comm_loss / num_batches
            avg_coord_score = epoch_coord_score / num_batches
            avg_entropy_bonus = epoch_entropy_bonus / (num_batches * seq_len) if num_batches > 0 and seq_len > 0 else 0.0
            avg_progress_reward = epoch_progress_reward / (num_batches * seq_len) if num_batches > 0 and seq_len > 0 else 0.0
            avg_oscillation_penalty = epoch_oscillation_penalty / num_batches if num_batches > 0 else 0.0
            # Adjacent coordination ratio no longer tracked in simplified implementation
            
            # Update training history
            self.training_history['epoch_losses'].append(avg_loss)
            self.training_history['communication_losses'].append(avg_comm_loss)
            self.training_history['coordination_scores'].append(avg_coord_score)
            self.training_history['entropy_bonuses'].append(avg_entropy_bonus)
            self.training_history['progress_rewards'].append(avg_progress_reward)
            self.training_history['oscillation_penalties'].append(avg_oscillation_penalty)
            # Adjacent coordination no longer tracked in simplified implementation
            
            # Validation evaluation
            if epoch % 5 == 0:  # Evaluate every 5 epochs
                val_metrics = self.comprehensive_evaluate(max_cases=20)
                if val_metrics:
                    # Use the new comprehensive metric names
                    self.training_history['val_success_rates'].append(val_metrics['episode_success_rate'])
                    self.training_history['val_agents_reached_rates'].append(val_metrics['agents_reached_rate'])
                    self.training_history['val_avg_flow_times'].append(val_metrics['avg_flow_time'])
            
            print(f"📊 Epoch {epoch+1} Summary:")
            print(f"   Loss: {avg_loss:.4f}")
            print(f"   Communication: {avg_comm_loss:.4f}")
            print(f"   Coordination: {avg_coord_score:.4f}")
            print(f"   🤝 Adjacent Coordination: Enhanced (2x penalty for close agents)")
            print(f"   🔄 Oscillation Penalty: {avg_oscillation_penalty:.4f} (weight: {self.oscillation_penalty_weight.item():.3f})")
            if entropy_bonus_weight > 0:
                print(f"   Entropy Weight: {entropy_bonus_weight:.4f} | Entropy Bonus: {avg_entropy_bonus:.4f}")
            if self.config.get('use_goal_progress_reward', False):
                print(f"   Progress Reward: {avg_progress_reward:.4f}")
            
            # Clean up old position history to prevent memory buildup
            self.clear_old_position_history(max_batches_to_keep=5)
            
        print("✅ Training completed!")
        
        # Stop animation worker
        self.stop_animation_worker()
    
    def save_model(self, path):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }, path)
        print(f"💾 Model saved to {path}")
    
    def _update_positions_from_actions(self, positions, actions):
        """
        Update agent positions based on predicted actions.
        
        Args:
            positions: Current positions [batch_size, num_agents, 2]
            actions: Action indices [batch_size, num_agents]
            
        Returns:
            updated_positions: New positions after applying actions [batch_size, num_agents, 2]
        """
        # Action mapping: 0=stay, 1=right, 2=up, 3=left, 4=down
        action_deltas = torch.tensor([
            [0, 0],   # 0: Stay
            [1, 0],   # 1: Right 
            [0, 1],   # 2: Up
            [-1, 0],  # 3: Left
            [0, -1]   # 4: Down
        ], device=positions.device, dtype=positions.dtype)
        
        # Get position deltas for each agent's action
        deltas = action_deltas[actions]  # [batch_size, num_agents, 2]
        
        # Update positions
        new_positions = positions + deltas
        
        # Apply boundary constraints (clamp to grid)
        grid_size = self.grid_size
        new_positions = torch.clamp(new_positions, 0, grid_size - 1)
        
        return new_positions
        

def main():
    """Main function to run enhanced MAPF training"""
    import argparse
    
    # Parse command line arguments
    """Main function to run enhanced MAPF training"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced MAPF Training with Real-time Visualization')
    parser.add_argument('--config', type=str, default='configs/config_snn.yaml',
                        help='Path to config file (default: configs/config_snn.yaml)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Enable real-time visualization during training')
    parser.add_argument('--eval-only', action='store_true', default=False,
                        help='Run evaluation only (no training)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the trained model')
    parser.add_argument('--load-path', type=str, default=None,
                        help='Path to load a pre-trained model')
    args = parser.parse_args()
    
    try:
        print("🚀 Starting Enhanced MAPF Training System...")
        print(f"📋 Using config: {args.config}")
        
        # Create enhanced trainer
        trainer = EnhancedTrainer(config_path=args.config)
        
        # Override epochs if specified
        if args.epochs is not None:
            trainer.num_epochs = args.epochs
            print(f"📊 Epochs overridden to: {args.epochs}")
        
        # Enable visualization if requested
        if args.visualize:
            trainer.visualize_training = True
            print("🎬 Real-time visualization enabled")
        
        # Load pre-trained model if specified
        if args.load_path:
            try:
                trainer.model.load_state_dict(torch.load(args.load_path, map_location=trainer.device))
                print(f"✅ Loaded pre-trained model from: {args.load_path}")
            except Exception as e:
                print(f"❌ Failed to load model from {args.load_path}: {e}")
                return
        
        if args.eval_only:
            # Run evaluation only
            print("🧪 Running evaluation only...")
            metrics = trainer.comprehensive_evaluate(max_cases=100)
            
            print("\n📊 EVALUATION RESULTS:")
            print("=" * 50)
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            print("=" * 50)
        else:
            # Run training
            print("🏋️ Starting training...")
            trainer.train()
            
            # Run final evaluation
            print("\n🧪 Running final evaluation...")
            final_metrics = trainer.comprehensive_evaluate(max_cases=50)
            
            print("\n📊 FINAL EVALUATION RESULTS:")
            print("=" * 50)
            for key, value in final_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            print("=" * 50)
            
            # Save model if path specified
            save_path = args.save_path or f"trained_models/enhanced_mapf_model_epoch_{trainer.num_epochs}.pth"
            try:
                trainer.save_model(save_path)
                print(f"💾 Model saved to: {save_path}")
            except Exception as e:
                print(f"❌ Failed to save model: {e}")
        
        # Stop animation worker if running
        if hasattr(trainer, 'stop_animation_worker'):
            trainer.stop_animation_worker()
        
        print("✅ Enhanced MAPF Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up matplotlib
        try:
            plt.close('all')
        except:
            pass


if __name__ == "__main__":
    main()
