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

# Configure matplotlib FIRST before any other imports
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better GUI support
import matplotlib.pyplot as plt
# Enable interactive mode
plt.ion()
# Configure matplotlib settings
plt.rcParams['figure.raise_window'] = True  # Allow window to be raised
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
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
from loss_compute import (
    compute_entropy_bonus, entropy_bonus_schedule, compute_all_losses,
    compute_goal_progress_reward, compute_incremental_rewards,
    compute_progress_shaping_reward
    # Removed compute_spike_sparsity_loss - overwhelming the model
)
#from pretraining import pretrain_model
from debug_utils import (
    debug_print, print_tensor_info, check_gradient_flow, log_training_step,
    analyze_agent_positions, detect_position_anomalies, monitor_loss_components,
    validate_model_output, check_collision_consistency, summarize_training_metrics,
    log_system_resources
)
from visualize import MAPFVisualizer
from spike_monitor import create_spike_monitor, SpikeTracker

from pretraining import pretrain_model, generate_pretraining_batch, visualize_pretrain_scenario, progressive_pretrain_model
# Import entropy functions now available in loss_compute.py
# from train import entropy_bonus_schedule, compute_entropy_bonus

# SNN-specific imports
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.base import MemoryModule

# Export the main trainer class
__all__ = ['EnhancedTrainer']

def safe_reset_snn(model):
    """
    Safely reset SNN state, only resetting modules that are actual memory modules.
    This avoids warnings about trying to reset non-memory modules.
    """
    for module in model.modules():
        if isinstance(module, MemoryModule):
            module.reset()

class EnhancedTrainer:
    """Enhanced trainer with multi-step learning and visualization"""

    def __init__(self, config_or_path="configs/config_snn.yaml", debug_mode=False,
                visualizer = None):
        # Load configuration with proper fallback
        if isinstance(config_or_path, dict):
            # Config dict was passed directly
            self.config = config_or_path
            print(f"📋 Using provided config dictionary")
        elif isinstance(config_or_path, str) and os.path.exists(config_or_path):
            # Config file path was passed
            with open(config_or_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"📋 Loaded config from {config_or_path}")
        else:
            print(f"⚠️  Config file {config_or_path} not found, using global config")
            self.config = config

        # Extract board size and grid size from config
        self.board_size = self.config.get('board_size', [9, 9])
        self.grid_size = self.board_size[0]  # Assuming square grid
        print(f"🗺️  Using board size: {self.board_size}, grid size: {self.grid_size}")

        # Extract experiment name for checkpoint saving
        self.exp_name = self.config.get('exp_name', 'trained_models/snn_experiment')
        print(f"📁 Experiment name: {self.exp_name}")

        self.device = self.config['device']
        print(f"🚀 Enhanced Trainer initialized on {self.device}")

        self.visualizer = visualizer
        self.visualize_training = self.config.get('visualize_training', True) and (self.visualizer is not None)


        # Training parameters
        # Use separate sequence_length for multi-step training (independent of episode length)
        dataset_min_time = self.config.get('min_time', 40)
        self.sequence_length = self.config.get('sequence_length', 60)  # Multi-step training window
        self.batch_size = self.config.get('batch_size', 16)
        self.learning_rate = float(self.config.get('learning_rate', 0.001))
        self.num_epochs = self.config.get('epochs', 50)

        print(f"   📏 Using sequence length: {self.sequence_length} (dataset min_time: {dataset_min_time})")

        # Communication learning parameters - BALANCED WEIGHTS
        self.communication_weight = self.config.get('communication_weight', 5.0)  # Increased from 0.3
        self.coordination_weight = self.config.get('coordination_weight', 5.0)    # Increased from 0.2
        self.temporal_consistency_weight = self.config.get('temporal_consistency_weight', 3.0)  # Increased from 0.1

        # Pre-training parameters
        self.enable_pretraining = self.config.get('enable_pretraining', True)
        self.use_progressive_pretraining = self.config.get('use_progressive_pretraining', True)
        self.pretraining_epochs = self.config.get('pretraining_epochs', 5)
        self.pretraining_batches_per_epoch = self.config.get('pretraining_batches_per_epoch', 50)
        print(f"   🎯 Pre-training: {'enabled' if self.enable_pretraining else 'disabled'}")
        if self.enable_pretraining:
            if self.use_progressive_pretraining:
                print(f"       Progressive: 4 epochs 3x3, 3 epochs 5x5, 3 epochs 7x7")
            else:
                print(f"       Standard: {self.pretraining_epochs} epochs, {self.pretraining_batches_per_epoch} batches/epoch")

        # Initialize model first
        self.model = Network(self.config).to(self.device)

        # Disable torch.compile for SNN compatibility and enable anomaly detection
        print(f"   🧠 SNN model initialized (torch.compile disabled for SNN compatibility)")
        torch.autograd.set_detect_anomaly(True)
        print(f"   🔍 Anomaly detection enabled for gradient debugging")

        # Enhanced oscillation penalty weight - BALANCED
        oscillation_weight = self.config.get('oscillation_penalty_weight', 7.0)  # Balanced weight
        self.oscillation_penalty_weight = nn.Parameter(torch.tensor(oscillation_weight, device=self.device))
        self.model.register_parameter('oscillation_penalty_weight', self.oscillation_penalty_weight)

        # Enhanced oscillation detection parameters
        self.use_enhanced_oscillation_detection = self.config.get('use_enhanced_oscillation_detection', True)
        self.oscillation_history_length = self.config.get('oscillation_history_length', 8)
        self.oscillation_min_cycle_length = self.config.get('oscillation_min_cycle_length', 2)
        self.oscillation_max_cycle_length = self.config.get('oscillation_max_cycle_length', 6)
        self.oscillation_stagnation_radius = self.config.get('oscillation_stagnation_radius', 1.5)
        self.oscillation_stagnation_threshold = self.config.get('oscillation_stagnation_threshold', 0.7)
        self.oscillation_enable_diagnostics = self.config.get('oscillation_enable_diagnostics', True)

        # Register stress inhibition weight as a learnable parameter - BALANCED
        initial_stress_weight = self.config.get('stress_inhibition_weight', 6.0)  # Increased from 0.5
        self.stress_inhibition_weight = nn.Parameter(torch.tensor(initial_stress_weight, device=self.device))
        self.model.register_parameter('stress_inhibition_weight', self.stress_inhibition_weight)

        # Register selective movement weight as a learnable parameter - BALANCED
        initial_selective_weight = self.config.get('selective_movement_weight', 4.0)  # Increased from 0.8
        self.selective_movement_weight = nn.Parameter(torch.tensor(initial_selective_weight, device=self.device))
        self.model.register_parameter('selective_movement_weight', self.selective_movement_weight)

        if self.use_enhanced_oscillation_detection:
            print(f"   🔄 Enhanced oscillation penalty: weight={self.oscillation_penalty_weight.item():.1f} (learnable, detects all cyclic patterns)")
            print(f"      - History length: {self.oscillation_history_length}")
            print(f"      - Cycle length range: {self.oscillation_min_cycle_length}-{self.oscillation_max_cycle_length}")
            print(f"      - Stagnation radius: {self.oscillation_stagnation_radius}")
            print(f"      - Diagnostics: {'enabled' if self.oscillation_enable_diagnostics else 'disabled'}")
        else:
            print(f"   🔄 Simple oscillation penalty: weight={self.oscillation_penalty_weight.item():.1f} (learnable, A->B->A->B pattern detection)")
        print(f"   🧠 Stress inhibition: weight={self.stress_inhibition_weight.item():.3f} (learnable)")
        print(f"   🎯 Selective movement: weight={self.selective_movement_weight.item():.3f} (learnable)")

        # Initialize STDP coordination system for learning rate synchronization
        try:
            from stdp_coordination import STDPCoordinationSystem
            self.stdp_system = STDPCoordinationSystem(self.config).to(self.device)
            print(f"   🧠 STDP coordination system initialized")
        except ImportError:
            self.stdp_system = None
            print(f"   ⚠️  STDP coordination system not available")

        # Initialize debug mode early
        # Set debug mode for all modules
        self.debug_enabled = debug_mode  # Use parameter instead of config
        from debug_utils import set_debug_mode
        set_debug_mode(debug_mode)

        # Visualization parameters
        self.visualize_training = self.config.get('visualize_training', True)
        self.vis_update_freq = self.config.get('vis_update_freq', 10)  # Update every N batches
        self.max_vis_episodes = self.config.get('max_vis_episodes', 3)  # Show max 3 episodes
        self.vis_episode_frequency = self.config.get('vis_episode_frequency', 5)  # Episode animation frequency
        self.animate_episodes = self.config.get('animate_episodes', True)  # Enable episode animation

        # Initialize visualizer
        #self.visualizer = MAPFVisualizer(self.config, self.grid_size, debug_mode=self.debug_enabled)
        #self.visualizer.start()  # Start the visualization process immediately

        # Initialize optimizer after all parameters are registered
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Initialize ReduceLROnPlateau scheduler based on agents reached percentage
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Reduce LR when agents_reached_rate stops increasing
            factor=self.config.get('lr_scheduler_factor', 0.7),  # Reduce LR by 70% (more conservative)
            patience=self.config.get('lr_scheduler_patience', 8),  # Wait 8 epochs (wider patience)
            min_lr=self.config.get('lr_scheduler_min_lr', 1e-6),  # Minimum LR
            threshold=self.config.get('lr_scheduler_threshold', 0.02)  # 2% improvement threshold
        )
        print(f"📉 ReduceLROnPlateau scheduler initialized (agents_reached_rate): factor={self.scheduler.factor}, patience={self.scheduler.patience}, threshold={self.scheduler.threshold}")

        # Track for STDP learning rate synchronization
        self.stdp_lr_multiplier = self.config.get('stdp_lr_multiplier', 4.0)  # STDP LR = 4x main LR
        print(f"🧠 STDP learning rate multiplier: {self.stdp_lr_multiplier}x")

        # Initialize STDP learning rate sync
        if self.stdp_system is not None:
            initial_stdp_lr = self.learning_rate * self.stdp_lr_multiplier
            self.stdp_system.update_learning_rate(initial_stdp_lr)

        self.criterion = nn.CrossEntropyLoss()

        # Training history
        self.training_history = {
            'epoch_losses': [],
            'communication_losses': [],
            'coordination_scores': [],
            'collision_rates': [],
            'collision_losses': [],  # Add collision loss tracking
            'success_rates': [],  # training success rates (placeholder)
            'entropy_bonuses': [],  # Add entropy bonus tracking
            'progress_rewards': [],  # Add progress reward tracking
            'stay_rewards': [],  # Add stay action reward tracking
            'stay_action_rates': [],  # Add stay action rate tracking
            'avg_stay_probabilities': [],  # Add average stay probability tracking
            'oscillation_penalties': [],  # Add oscillation penalty tracking
            'stress_inhibition_losses': [],  # Track stress inhibition mechanism
            'selective_movement_losses': [],  # Track selective movement coordination
            'stress_zones_detected': [],  # Track stress zone detection frequency
            'avg_stress_intensity': [],  # Track average stress intensity
            # STDP coordination metrics
            'stdp_rewards': [],  # Track STDP coordination rewards
            'coordination_efficiency': [],  # Track coordination efficiency
            'goal_proximity_bonus': [],  # Track goal proximity bonus
            'conflict_resolution': [],  # Track conflict resolution
            # Incremental reward tracking
            'goal_approach_rewards': [],  # Track goal approach rewards
            'collision_avoidance_rewards': [],  # Track collision avoidance rewards
            'grid_movement_rewards': [],  # Track grid movement rewards
            'movement_away_rewards': [],  # Track movement away rewards
            # Removed spike_sparsity_losses - was overwhelming the model
            'distance_away_penalties': [],  # Track distance-away penalties
            # Adjacent coordination tracking removed in simplified implementation
            # validation metrics
            'val_success_rates': [],
            'val_agents_reached_rates': [],
            'val_avg_flow_times': []
        }

        # Share training history with visualizer - IMPORTANT: this must be a reference, not a copy
        if self.visualizer is not None:
            self.visualizer.training_history = self.training_history

        # Initialize debug tracking (debug_enabled already set above)
        self.metrics_history = {}  # For debug utility tracking
        self.resource_log_freq = self.config.get('resource_log_freq', 50)  # Log every N steps

        # Initialize spike monitor for SNN layer monitoring
        spike_config = {
            'max_history': self.config.get('spike_history_size', 1000),
            'save_dir': self.config.get('spike_log_dir', 'spike_logs')
        }
        self.spike_monitor = create_spike_monitor(spike_config)
        self.spike_tracker = SpikeTracker(self.spike_monitor)

        # Register SNN layers for monitoring
        self.spike_monitor.register_layer('global_features')
        self.spike_monitor.register_layer('snn_output')
        self.spike_monitor.register_layer('action_logits')
        self.spike_monitor.register_layer('model_output')

        enable_spike_monitoring = self.config.get('enable_spike_monitoring', True)
        if enable_spike_monitoring:
            print("🧠 Spike monitoring enabled for SNN layers")
        else:
            self.spike_tracker.disable()
            print("🧠 Spike monitoring disabled")

        print("✅ Enhanced Trainer ready!")

    def debug_print(self, message):
        """Conditional debug printing based on debug mode"""
        if self.debug_enabled:
            from debug_utils import debug_print
            debug_print(message)

    def setup_visualization_if_enabled(self):
        """Setup visualization after command line args are processed"""
        if self.visualize_training:
            print("🎬 Setting up visualization with test data...")
            print(f"   📊 Visualization update frequency: every {self.vis_update_freq} batches")
            print(f"   🎯 Grid size: {self.grid_size}")

            try:
                success = self.visualizer.setup_visualization()
                if not success:
                    print("⚠️  Visualization setup returned False, but continuing with training...")
                else:
                    print("✅ Visualization setup successful!")

                # Check if visualizer methods are available
                if hasattr(self.visualizer, 'queue_animation'):
                    print("   ✅ queue_animation method available")
                else:
                    print("   ❌ queue_animation method missing!")

                if hasattr(self.visualizer, 'queue_plot_update'):
                    print("   ✅ queue_plot_update method available")
                else:
                    print("   ❌ queue_plot_update method missing!")

                # Enable synchronous mode for coordinated training-visualization
                if hasattr(self.visualizer, 'set_synchronous_mode'):
                    self.visualizer.set_synchronous_mode(True)
                    print("🔄 Synchronous training-visualization mode enabled")
                else:
                    print("⚠️  set_synchronous_mode not available")

            except Exception as viz_error:
                print(f"❌ Visualization setup failed with error: {viz_error}")
                import traceback
                traceback.print_exc()

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
                        delta = action_deltas[int(action)]
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
            fov_data: tensor of shape [num_agents, 2, 7, 7] (used for determining case info)
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
            fov_data: tensor of shape [num_agents, 2, 7, 7] (used for determining case info)
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
            fov_data: tensor of shape [num_agents, 2, 7, 7] (used for determining case info)
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

    def load_expert_actions_for_training(self, train_fovs, train_actions, train_positions, train_goals):
        """
        Load expert actions from solution.yaml files for all training sequences.

        Args:
            train_fovs: Training FOVs tensor
            train_actions: Training actions tensor (from trajectory_record.npy - not used for loss)
            train_positions: Training positions tensor
            train_goals: Training goals tensor

        Returns:
            Expert actions tensor [num_sequences, sequence_length, num_agents] from solution.yaml
        """
        print("📜 Loading expert actions from solution.yaml files...")

        # Import expert utilities
        from expert_utils import extract_expert_actions_from_solution
        import os

        dataset_root = self.config.get('train', {}).get('root_dir', 'dataset/5_8_28')
        mode = 'train'

        num_sequences = len(train_fovs)
        sequence_length = train_fovs.shape[1]
        num_agents = train_fovs.shape[2]

        # Initialize expert actions tensor
        expert_actions = torch.zeros(num_sequences, sequence_length, num_agents, dtype=torch.long)

        # Load expert actions for each sequence
        loaded_count = 0
        for seq_idx in range(num_sequences):
            # For this simple implementation, we'll assume sequence index matches case index
            # In a more sophisticated implementation, we'd need to track case indices
            case_idx = seq_idx

            case_name = f"case_{case_idx}"
            solution_path = os.path.join(dataset_root, mode, case_name, "solution.yaml")

            if os.path.exists(solution_path):
                try:
                    # Extract expert actions from solution.yaml
                    expert_trajectory = extract_expert_actions_from_solution(
                        solution_path, num_agents, sequence_length
                    )

                    if expert_trajectory is not None:
                        # expert_trajectory is [time_steps, num_agents], we need [num_agents, time_steps]
                        expert_actions[seq_idx] = expert_trajectory  # Shape: [sequence_length, num_agents]
                        loaded_count += 1
                    else:
                        # Fallback to trajectory_record actions if solution.yaml parsing fails
                        print(f"⚠️ Failed to parse solution.yaml for case {case_idx}, using trajectory_record fallback")
                        expert_actions[seq_idx] = train_actions[seq_idx]  # Use original actions as fallback

                except Exception as e:
                    print(f"⚠️ Error loading expert actions for case {case_idx}: {e}")
                    # Fallback to trajectory_record actions
                    expert_actions[seq_idx] = train_actions[seq_idx]  # Use original actions as fallback
            else:
                # No solution.yaml found, use trajectory_record actions as fallback
                if seq_idx % 100 == 0:  # Log warning every 100 cases to avoid spam
                    print(f"⚠️ No solution.yaml found for case {case_idx}, using trajectory_record fallback")
                expert_actions[seq_idx] = train_actions[seq_idx]  # Use original actions as fallback

        print(f"✅ Loaded expert actions: {loaded_count}/{num_sequences} from solution.yaml, "
              f"{num_sequences - loaded_count} fallback to trajectory_record")

        return expert_actions

    def create_sequence_dataset(self, dataset_root="dataset/5_8_28", mode="train"):
        """Create multi-step sequence dataset"""
        print(f"📊 Creating sequence dataset from {dataset_root}/{mode}")

        # Create proper data loader using the same approach as train.py
        try:
            data_loader = SNNDataLoader(self.config)

            sequences_fov = []
            sequences_actions = []
            sequences_positions = []
            sequences_goals = []
            sequences_obstacles = []

            # Choose the appropriate loader based on mode
            if mode == "train":
                loader = data_loader.train_loader
            else:
                loader = data_loader.valid_loader

            if loader is None:
                print(f"Warning: No {mode} loader available")
                return None, None, None, None, None

        except Exception as e:
            print(f"Error creating data loader: {e}")
            return None, None, None, None, None
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
                    # fovs: [batch, T, num_agents, 2, 7, 7], actions: [batch, T, num_agents]
                    case_fov = fovs[b]  # [T, num_agents, 2, 7, 7]
                    case_actions = actions[b]  # [T, num_agents]
                    case_idx = case_indices[b] if isinstance(case_indices, (list, torch.Tensor)) else case_count

                    T = case_fov.shape[0]
                    num_agents = case_fov.shape[1]

                    debug_print(f"Case {case_idx}: T={T}, num_agents={num_agents}, sequence_length={self.sequence_length}")

                    # Use adaptive sequence length instead of skipping
                    if T < self.sequence_length:
                        actual_sequence_length = max(1, T)  # Use at least 1 timestep, or full available length
                        debug_print(f"Case {case_idx}: Using adaptive length {actual_sequence_length} instead of configured {self.sequence_length}")
                    else:
                        actual_sequence_length = self.sequence_length
                        debug_print(f"Case {case_idx}: Using full sequence length {actual_sequence_length}")

                    # Extract real positions from trajectory and YAML data
                    try:
                        # Build the correct dataset path for this mode
                        mode_dataset_path = os.path.join(dataset_root, mode)

                        # Extract agent positions and goals using real case data
                        first_fov = case_fov[0]  # [num_agents, 2, 7, 7]
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

                    # Create overlapping sequences with adaptive length
                    for start_t in range(T - actual_sequence_length + 1):
                        end_t = start_t + actual_sequence_length

                        seq_fov = case_fov[start_t:end_t]  # [seq_len, num_agents, 2, 7, 7]
                        seq_actions = case_actions[start_t:end_t]  # [seq_len, num_agents]

                        # Ensure sequence has consistent length by padding if necessary
                        if seq_fov.shape[0] < self.sequence_length:
                            # Pad sequence to required length
                            pad_length = self.sequence_length - seq_fov.shape[0]

                            # Pad with last frame repeated
                            last_fov = seq_fov[-1:].repeat(pad_length, 1, 1, 1, 1)
                            last_actions = seq_actions[-1:].repeat(pad_length, 1)

                            seq_fov = torch.cat([seq_fov, last_fov], dim=0)
                            seq_actions = torch.cat([seq_actions, last_actions], dim=0)

                            debug_print(f"Padded sequence from {seq_fov.shape[0] - pad_length} to {seq_fov.shape[0]} timesteps")

                        sequences_fov.append(seq_fov)
                        sequences_actions.append(seq_actions)
                        # Debug: Track which cases produce non-5 agent tensors
                        if positions.shape[0] != 5:
                            print(f"🚨 Case {case_idx} produced {positions.shape[0]} agents instead of 5!")
                            print(f"   Positions: {positions}")
                            print(f"   Goals: {goals}")

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

        # Debug: Check tensor shapes before stacking and fix mismatches
        print(f"🔍 About to stack {len(sequences_positions)} position tensors...")

        # Fix tensor shape mismatches by ensuring all tensors have exactly 5 agents
        fixed_positions = []
        fixed_goals = []

        target_agents = 5  # All cases should have 5 agents

        for i, (pos, goal) in enumerate(zip(sequences_positions, sequences_goals)):
            # Check and fix positions
            if pos.shape[0] != target_agents:
                print(f"⚠️  Entry {i}: position shape {pos.shape}, expected [{target_agents}, 2]")
                if pos.shape[0] < target_agents:
                    # Pad with dummy positions
                    padding_size = target_agents - pos.shape[0]
                    dummy_positions = torch.zeros(padding_size, 2, dtype=torch.float32)
                    pos = torch.cat([pos, dummy_positions], dim=0)
                    print(f"   Padded to {pos.shape}")
                elif pos.shape[0] > target_agents:
                    # Truncate to target size
                    pos = pos[:target_agents]
                    print(f"   Truncated to {pos.shape}")
            fixed_positions.append(pos)

            # Check and fix goals (same logic)
            if goal.shape[0] != target_agents:
                print(f"⚠️  Entry {i}: goal shape {goal.shape}, expected [{target_agents}, 2]")
                if goal.shape[0] < target_agents:
                    # Pad with dummy goals
                    padding_size = target_agents - goal.shape[0]
                    dummy_goals = torch.zeros(padding_size, 2, dtype=torch.float32)
                    goal = torch.cat([goal, dummy_goals], dim=0)
                    print(f"   Goal padded to {goal.shape}")
                elif goal.shape[0] > target_agents:
                    # Truncate to target size
                    goal = goal[:target_agents]
                    print(f"   Goal truncated to {goal.shape}")
            fixed_goals.append(goal)

        # Use fixed tensors
        sequences_positions = fixed_positions
        sequences_goals = fixed_goals

        # Convert to tensors
        fovs_tensor = torch.stack(sequences_fov)  # [N, seq_len, num_agents, 2, 7, 7]
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
            torch.empty(0, self.sequence_length, 5, 2, 7, 7),  # Empty FOV tensor
            torch.empty(0, self.sequence_length, 5, dtype=torch.long),  # Empty actions tensor
            torch.empty(0, 5, 2),  # Empty positions tensor
            torch.empty(0, 5, 2),   # Empty goals tensor
            []  # Empty obstacles list
        )

    # All loss computations moved to loss_compute.py for cleaner architecture
    # Oscillation detection now uses simple A->B->A->B pattern detection in loss_compute.py

    # Removed setup_visualization method - now handled by MAPFVisualizer

    # Removed update_visualization method - now handled by MAPFVisualizer

    # Removed visualize_episode method - now handled by MAPFVisualizer

    # Removed _to_np method - now handled by MAPFVisualizer

    # Removed print_initial_state_matrix method - now handled by MAPFVisualizer

    # Removed setup_animation_queue method - now handled by MAPFVisualizer

    # Removed queue_animation method - now handled by MAPFVisualizer

    # Removed start_show_once_animation_worker method - now handled by MAPFVisualizer

    # Removed show_once_animation_worker method - now handled by MAPFVisualizer

    # Removed process_single_animation method - now handled by MAPFVisualizer

    # Removed visualize_initial_state method - now handled by MAPFVisualizer

    # Removed stop_animation_worker method - now handled by MAPFVisualizer

    # Removed play_single_animation_seamlessly method - now handled by MAPFVisualizer

    # Removed unlock_plot_and_restore method - now handled by MAPFVisualizer

    # Removed process_pending_animations method - now handled by MAPFVisualizer

    def comprehensive_validation(self, max_cases=None, extended_time=True):
        """Comprehensive validation on validation set

        Args:
            max_cases: Maximum number of cases to validate (None = use entire validation set)
            extended_time: Whether to use extended validation time
        """
        # Import debug utilities at the beginning
        from debug_utils import debug_print
        from loss_compute import clear_oscillation_history

        # Clear oscillation history at start of validation
        clear_oscillation_history()  # This now clears both simple and enhanced oscillation history

        if not hasattr(self, 'model') or self.model is None:
            print("❌ No model available for validation")
            return {}

        # Create validation data
        try:
            val_fovs, val_actions, val_positions, val_goals, val_obstacles = self.create_sequence_dataset(mode="valid")
            if val_fovs is None or len(val_fovs) == 0:
                print("❌ No validation data available")
                return {}
        except Exception as e:
            print(f"❌ Failed to create validation dataset: {e}")
            return {}

        # Use entire validation dataset unless max_cases is specified
        if max_cases is None:
            num_cases = len(val_fovs)
            print(f"🧪 Running comprehensive validation on entire validation set ({num_cases} cases)...")
        else:
            num_cases = min(len(val_fovs), max_cases)
            print(f"🧪 Running comprehensive validation on {num_cases}/{len(val_fovs)} cases...")

        if extended_time:
            print("⏱️  Using extended validation time for better goal-reaching assessment")

        # Limit to max_cases if specified
        if max_cases is not None:
            num_cases = min(len(val_fovs), max_cases)
            val_fovs = val_fovs[:num_cases]
            val_actions = val_actions[:num_cases]
        val_positions = val_positions[:num_cases]
        val_goals = val_goals[:num_cases]
        val_obstacles = val_obstacles[:num_cases]

        self.model.eval()
        total_agents_reached = 0
        total_agents = 0
        total_collisions = 0  # Track total collisions across all episodes
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
                safe_reset_snn(self.model)

                # Initialize per-case metrics (RL style - no accuracy tracking)
                agents_reached_goal = 0
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

                # Main validation loop (using sequence data)
                for t in range(seq_len):
                    fov_t = sequence_fov[:, t]  # [1, agents, channels, h, w]
                    # Pass goals for STDP coordination
                    output = self.model(fov_t, goals=sequence_goals)

                    # Handle different output shapes
                    debug_print(f"Model output shape: {output.shape}, num_agents: {num_agents}")

                    if output.dim() == 1:
                        # If output is 1D, reshape based on actual number of agents in the batch
                        actual_agents = output.numel() // 5  # Assuming 5 action classes
                        if actual_agents != num_agents:
                            debug_print(f"Agent count mismatch: expected {num_agents}, got {actual_agents}")
                            num_agents = min(num_agents, actual_agents)  # Use the smaller count
                        output = output.reshape(1, actual_agents, 5)  # [1, agents, action_classes]
                    elif output.dim() == 2:
                        # If output is 2D [batch, features], reshape appropriately
                        batch_size, features = output.shape
                        if features % 5 == 0:  # Assuming 5 action classes
                            actual_agents = features // 5
                            if actual_agents != num_agents:
                                debug_print(f"Agent count mismatch: expected {num_agents}, got {actual_agents}")
                                num_agents = min(num_agents, actual_agents)
                            output = output.reshape(batch_size, actual_agents, 5)
                        else:
                            # Fallback: use original approach but with safety check
                            if output.numel() % num_agents == 0:
                                output = output.reshape(1, num_agents, -1)
                            else:
                                debug_print(f"❌ Cannot reshape output of size {output.numel()} for {num_agents} agents")
                                continue
                    else:
                        # If already 3D or higher, try to reshape
                        try:
                            output = output.reshape(1, num_agents, -1)  # [1, agents, action_classes]
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
                                delta = action_deltas[int(action)]
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

                    # Get current (updated) positions and goals
                    current_positions = current_agent_positions.cpu().numpy()  # Use updated positions, not initial
                    current_goals = sequence_goals[0].cpu().numpy()  # [agents, 2] - goals are constant

                    # Check predictions and goal reaching (only for non-collided agents)
                    for agent in range(num_agents):
                        # Bounds check for actions
                        if agent >= sequence_actions.shape[2]:
                            continue

                        # Skip accuracy tracking in RL mode - focus on goal reaching

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

                # Extended validation: Continue running the model for additional timesteps if enabled
                if extended_time and agents_reached_goal < num_agents:
                    debug_print(f"🔄 Running extended validation for case {i} ({agents_reached_goal}/{num_agents} reached in {seq_len} steps)")
                    extended_timesteps = min(60, seq_len * 2)  # Run for up to 60 more steps or double the original sequence

                    # Generate additional FOV observations and continue validation
                    for t_ext in range(extended_timesteps):
                        # Create FOV from current positions (we'll use last FOV as template and update positions)
                        current_fov = sequence_fov[:, -1].clone()  # Use last FOV as template

                        # Update positions in FOV (simplified - this would need proper FOV generation in practice)
                        # For now, just use the last FOV

                        output = self.model(current_fov, goals=sequence_goals)

                        # Process output (same logic as before)
                        if output.dim() == 1:
                            actual_agents = output.numel() // 5
                            output = output.reshape(1, actual_agents, 5)
                        elif output.dim() == 2:
                            batch_size, features = output.shape
                            if features % 5 == 0:
                                actual_agents = features // 5
                                output = output.reshape(batch_size, actual_agents, 5)

                        predicted_actions = torch.argmax(output[0], dim=1)

                        # Update positions
                        prev_positions = current_agent_positions.clone()
                        for agent_idx in range(min(num_agents, len(predicted_actions))):
                            action = predicted_actions[agent_idx].item()
                            if action < len(action_deltas):
                                delta = action_deltas[int(action)]
                                new_pos = current_agent_positions[agent_idx] + delta
                                new_pos[0] = torch.clamp(new_pos[0], 0, 8)
                                new_pos[1] = torch.clamp(new_pos[1], 0, 8)
                                current_agent_positions[agent_idx] = new_pos

                        # Check for new goal arrivals
                        current_positions = current_agent_positions.cpu().numpy()
                        for agent in range(num_agents):
                            if agent_reached_times[agent] == -1 and not agent_collided[agent]:
                                if agent < len(current_positions) and agent < len(current_goals):
                                    agent_pos = current_positions[agent]
                                    agent_goal = current_goals[agent]
                                    distance_to_goal = np.linalg.norm(agent_pos - agent_goal)
                                    if distance_to_goal < 0.5:
                                        agent_reached_times[agent] = seq_len + t_ext
                                        agents_reached_goal += 1
                                        debug_print(f"Agent {agent} reached goal at extended timestep {seq_len + t_ext} in case {i}")

                        # Stop early if all agents reached goals
                        if agents_reached_goal >= num_agents:
                            break

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
                total_collisions += collided_agents  # Track total collisions across all episodes

                # Episode metrics (RL style - no accuracy tracking)
                total_agents_reached += agents_reached_goal
                total_agents += num_agents  # Count ALL agents for honest goal rate percentage

                # Debug info for collision and goal-reaching tracking
                if collided_agents > 0:
                    debug_print(f"🚫 Case {i}: {collided_agents}/{num_agents} agents collided, {agents_reached_goal}/{effective_agents} effective agents reached goals")
                else:
                    debug_print(f"✅ Case {i}: {agents_reached_goal}/{num_agents} agents reached goals (no collisions)")

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

        # Calculate comprehensive RL metrics (no imitation learning accuracy)
        agents_reached_rate = total_agents_reached / total_agents if total_agents > 0 else 0.0
        episode_success_rate = episode_success_count / total_episodes if total_episodes > 0 else 0.0
        avg_flow_time = total_flow_time / total_episodes if total_episodes > 0 else 0.0
        collision_rate = total_collisions / total_agents if total_agents > 0 else 0.0  # Fixed collision rate calculation

        metrics = {
            # RL-focused metrics (removed imitation learning accuracy)
            'agents_reached_rate': agents_reached_rate,  # Percentage of agents that reached goals
            'episode_success_rate': episode_success_rate,  # Percentage of episodes where all agents reached goals
            'avg_flow_time': avg_flow_time,  # Average time for agents to reach goals
            'collision_rate': collision_rate,  # Percentage of agents that collided
            'total_agents_reached': total_agents_reached,
            'total_agents': total_agents,
            'total_collisions': total_collisions,
            'episode_success_count': episode_success_count,
            'total_episodes': total_episodes,
            'cases_validated': num_cases
        };

        print(f"✅ Validation complete (RL metrics):")
        print(f"   🏁 Agents Reached: {total_agents_reached}/{total_agents} ({agents_reached_rate:.3f})")
        print(f"   🚫 Collision Rate: {collision_rate:.3f} ({total_collisions}/{total_agents})")
        print(f"   ⏱️  Average Flow Time: {avg_flow_time:.2f} timesteps")
        print(f"   📈 Episode Success Rate: {episode_success_rate:.3f}")
        debug_print(f"   Episodes: {episode_success_count}/{total_episodes} successful")
        debug_print(f"   Cases validated: {num_cases}")
        return metrics

    def run_pretraining(self):
        """
        Run pre-training phase on simple 3x3 scenarios to teach basic goal-seeking behavior.
        This helps the SNN learn fundamental movement patterns before complex training.
        """
        if not self.enable_pretraining:
            print("⏭️  Pre-training disabled, skipping...")
            return

        print("\n🎯 === PRE-TRAINING PHASE ===")
        print("Teaching SNN basic goal-seeking on 3x3 grids...")

        # Store initial weights before pretraining for comparison
        self.initial_pretrain_state = {}
        for name, param in self.model.named_parameters():
            if 'global_features' in name and param.requires_grad:
                self.initial_pretrain_state[name] = param.clone().detach()
                initial_magnitude = param.abs().mean().item()
                print(f"   📝 Initial {name}: mean_magnitude={initial_magnitude:.6f}")
                break  # Just track one example layer

        try:
            # Choose pre-training method
            if self.use_progressive_pretraining:
                # Run progressive pre-training
                history = progressive_pretrain_model(
                    model=self.model,
                    optimizer=self.optimizer,
                    config=self.config,
                    device=self.device
                )
            else:
                # Run standard pre-training
                history = pretrain_model(
                    model=self.model,
                    optimizer=self.optimizer,
                    config=self.config,
                    device=self.device,
                    epochs=self.pretraining_epochs,
                    batches_per_epoch=self.pretraining_batches_per_epoch
                )

            # Log pre-training results
            print(f"\n✅ Pre-training completed!")
            print(f"   📊 Final Loss: {history['loss'][-1]:.4f}")
            print(f"   🎯 Final Goal Reaching Rate: {history['agents_reached_goal'][-1]:.4f}")
            print(f"   📈 Goal Progress: {history['goal_progress'][-1]:.4f}")
            print(f"   💥 Collision Rate: {history['collision_rate'][-1]:.4f}")

            # Store pre-training history for analysis
            self.pretraining_history = history

            # Save pretrained model checkpoint to verify weight transfer
            print("💾 Saving pretrained model checkpoint...")
            pretrained_checkpoint_path = f"{self.exp_name}_pretrained.pth"
            pretrained_state = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'pretraining_history': history,
                'config': self.config,
                'pretraining_completed': True
            }
            torch.save(pretrained_state, pretrained_checkpoint_path)
            print(f"   ✅ Pretrained checkpoint saved: {pretrained_checkpoint_path}")

            # Verify model weights have changed from initialization
            total_params = sum(p.numel() for p in self.model.parameters())
            total_nonzero_params = sum((p != 0).sum().item() for p in self.model.parameters())
            print(f"   🔍 Model weight verification: {total_nonzero_params}/{total_params} params non-zero")

            # Compare weights before/after pretraining to verify learning occurred
            for name, param in self.model.named_parameters():
                if 'global_features' in name and param.requires_grad and name in self.initial_pretrain_state:
                    final_magnitude = param.abs().mean().item()
                    weight_change = (param - self.initial_pretrain_state[name]).abs().mean().item()
                    weight_std = param.std().item()
                    print(f"   🧠 After pretraining {name}:")
                    print(f"      📊 Final magnitude: {final_magnitude:.6f}")
                    print(f"      📈 Weight change: {weight_change:.6f}")
                    print(f"      📊 Standard deviation: {weight_std:.6f}")

                    if weight_change < 1e-6:
                        print(f"      ❌ WARNING: Weights barely changed - pretraining may have failed!")
                    else:
                        print(f"      ✅ Weights changed significantly - pretraining worked!")
                    break  # Just show one example layer

            # Optionally visualize a sample pre-training scenario
            if self.config.get('visualize_pretraining', False):
                print("🎬 Visualizing sample pre-training scenario...")
                try:
                    # Generate a sample scenario for visualization
                    sample_batch = generate_pretraining_batch(batch_size=1, sequence_length=10, grid_size=3, num_agents=2, fov_size=7)
                    batch_positions, batch_goals = sample_batch[2], sample_batch[3]  # Get tensors

                    # Visualize the scenario
                    visualize_pretrain_scenario(
                        positions=batch_positions[0, 0],  # First timestep, first batch
                        goals=batch_goals[0],  # First batch
                        title="Sample Pre-training Scenario (3x3 Grid)",
                        save_path=f"plots/pretraining_sample_scenario.png"
                    )
                    print("   📊 Sample scenario saved to plots/pretraining_sample_scenario.png")
                except Exception as e:
                    print(f"   ⚠️ Visualization failed: {e}")
                    import traceback
                    traceback.print_exc()

            print("🚀 Moving to main training phase...\n")

        except Exception as e:
            print(f"❌ Pre-training failed: {e}")
            print("⚠️  Continuing with main training...")
            import traceback
            traceback.print_exc()

    def train(self):
        """Main training loop with enhanced features"""
        print("🚀 Starting Enhanced MAPF Training...")

        # Check for existing pretrained checkpoint first
        pretrained_checkpoint_path = f"{self.exp_name}_pretrained.pth"
        if os.path.exists(pretrained_checkpoint_path) and self.enable_pretraining:
            print(f"🔄 Found existing pretrained checkpoint: {pretrained_checkpoint_path}")
            try:
                checkpoint = torch.load(pretrained_checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.pretraining_history = checkpoint.get('pretraining_history', {})
                print("   ✅ Pretrained checkpoint loaded successfully!")
                print("   ⏭️  Skipping pretraining phase...")
            except Exception as e:
                print(f"   ⚠️  Failed to load pretrained checkpoint: {e}")
                print("   🔄 Will run pretraining from scratch...")
                # Run pre-training phase first
                self.run_pretraining()
        else:
            # Run pre-training phase first
            self.run_pretraining()

        # Test visualization before starting main training
        if self.visualize_training:
            print("🧪 Testing visualization before main training...")
            try:
                # Create test episode data
                test_positions = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]], device=self.device)  # [1, 2, 2]
                test_goals = torch.tensor([[[5.0, 5.0], [1.0, 1.0]]], device=self.device)     # [1, 2, 2]
                test_episode_data = {
                    'positions': test_positions[0].cpu().numpy(),
                    'goals': test_goals[0].cpu().numpy(),
                    'obstacles': [],
                    'current_timestep': 0,
                    'show_full_paths': False,
                    'training_predictions': [[0, 1], [1, 0], [0, 1]]  # Simple test predictions
                }

                print("   📊 Attempting test animation queue...")
                self.visualizer.queue_animation(test_episode_data, case_idx=0, epoch=0, batch_idx=0)
                print("   ✅ Test visualization queue successful!")

            except Exception as e:
                print(f"   ❌ Test visualization failed: {e}")
                import traceback
                print(f"   Full error trace: {traceback.format_exc()}")

        # Verify pretrained weights are loaded for main training
        print("🔍 Verifying pretrained weights transfer to main training...")
        if hasattr(self, 'pretraining_history') and self.pretraining_history:
            print("   ✅ Pretraining completed - weights should be transferred")

            # Compare current weights with initial pretrain state if available
            if hasattr(self, 'initial_pretrain_state'):
                for name, param in self.model.named_parameters():
                    if name in self.initial_pretrain_state:
                        current_magnitude = param.abs().mean().item()
                        initial_magnitude = self.initial_pretrain_state[name].abs().mean().item()
                        weight_change = (param - self.initial_pretrain_state[name]).abs().mean().item()

                        print(f"   🧠 Main training {name}:")
                        print(f"      📊 Current magnitude: {current_magnitude:.6f}")
                        print(f"      📝 Initial magnitude: {initial_magnitude:.6f}")
                        print(f"      📈 Total change from init: {weight_change:.6f}")

                        if weight_change < 1e-6:
                            print(f"      ⚠️  WARNING: {name} weights unchanged - pretraining had no effect!")
                        else:
                            print(f"      ✅ {name} weights successfully updated by pretraining")
                        break  # Just check one layer as example
            else:
                # Fallback check - just look at weight magnitudes
                for name, param in self.model.named_parameters():
                    if 'global_features' in name and param.requires_grad:
                        weight_magnitude = param.abs().mean().item()
                        weight_std = param.std().item()
                        print(f"   🧠 Main training {name}: mean_magnitude={weight_magnitude:.6f}, std={weight_std:.6f}")
                        if weight_magnitude < 1e-6:
                            print(f"   ⚠️  WARNING: {name} weights are very small - pretraining may not have worked!")
                        else:
                            print(f"   ✅ {name} weights appear to have been trained (magnitude > 1e-6)")
                        break  # Just check one layer as example
        else:
            print("   ⚠️  No pretraining history found - using randomly initialized weights")

        # Create sequence dataset
        try:
            print("📊 Creating training dataset...")
            train_fovs, train_actions, train_positions, train_goals, train_obstacles = self.create_sequence_dataset(mode="train")

            # No data augmentation - we want to use teacher forcing with expert trajectories
            print(f"✅ Training dataset ready: {len(train_fovs)} sequences (no augmentation for teacher forcing)")

        except Exception as e:
            print(f"❌ Failed to create training dataset: {e}")
            return

        if len(train_fovs) == 0:
            print("❌ No training data available!")
            return

        # Load expert actions for all training cases
        print("🎯 Loading expert actions for teacher forcing...")
        train_expert_actions = self.load_expert_actions_for_training(train_fovs, train_actions, train_positions, train_goals)

        # Create data loader with expert actions
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(train_fovs, train_actions, train_positions, train_goals, train_expert_actions)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Store obstacles separately since they can't go in TensorDataset (variable length)
        self.train_obstacles = train_obstacles

        print(f"📈 Training for {self.num_epochs} epochs...")

        # Use tqdm for epoch progress
        epoch_pbar = tqdm(range(self.num_epochs), desc="Training Epochs", unit="epoch")

        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch+1}/{self.num_epochs}")

            # Clear oscillation and stay history at the start of each epoch
            from loss_compute import clear_oscillation_history, clear_stay_history
            clear_oscillation_history()  # This now clears both simple and enhanced oscillation history
            clear_stay_history()

            # Reset spike monitor for new epoch
            if self.spike_tracker.enabled:
                self.spike_monitor.reset_epoch()

            # Calculate entropy bonus weight for this epoch
            use_entropy_bonus = self.config.get('use_entropy_bonus', False)
            entropy_bonus_weight = 0.0
            if use_entropy_bonus:
                initial_entropy_weight = self.config.get('entropy_bonus_weight', 4.0)  # Increased from 2.0
                bonus_epochs = self.config.get('entropy_bonus_epochs', 8)  # Reduced from 15 to 8 for faster annealing
                decay_type = self.config.get('entropy_bonus_decay_type', 'fast_exponential')  # Changed to fast_exponential
                entropy_bonus_weight = entropy_bonus_schedule(epoch, initial_entropy_weight, bonus_epochs, decay_type, self.config)            # Multi-sequence training: Always use predicted actions for position updates
            # ALWAYS use TRUE expert actions from solution.yaml for loss computation
            # "Teacher forcing" now only affects logging and reward application timing
            print(f"🎯 Multi-sequence training for epoch {epoch+1}: agents follow predictions, learn from TRUE expert actions")
            teacher_forcing_ratio = 0.0  # Not used for loss computation anymore
            use_teacher_forcing = False  # Not used for loss computation anymore

            self.model.train()
            epoch_loss = 0.0
            epoch_comm_loss = 0.0
            epoch_coord_score = 0.0
            epoch_entropy_bonus = 0.0  # Track entropy bonus
            epoch_progress_reward = 0.0  # Track progress reward
            epoch_stay_reward = 0.0  # Track stay action reward
            epoch_collision_loss = 0.0  # Track collision loss
            # Oscillation penalty now applied per timestep, not tracked per epoch
            epoch_stress_inhibition = 0.0  # Track stress inhibition loss
            epoch_selective_movement = 0.0  # Track selective movement loss
            epoch_stress_zones = 0  # Track stress zone detection count
            epoch_stress_intensity = 0.0  # Track average stress intensity
            epoch_stdp_reward = 0.0  # Track STDP coordination rewards
            epoch_coordination_efficiency = 0.0  # Track coordination efficiency
            epoch_goal_proximity_bonus = 0.0  # Track goal proximity bonus
            epoch_conflict_resolution = 0.0  # Track conflict resolution
            # Initialize stay action statistics for epoch tracking
            self.epoch_stay_reward = 0.0

            # Initialize incremental reward tracking
            epoch_goal_approach_reward = 0.0
            epoch_collision_avoidance_reward = 0.0
            epoch_grid_movement_reward = 0.0
            epoch_movement_away_reward = 0.0
            # Removed spike sparsity loss tracking - was overwhelming the model
            epoch_distance_away_penalty = 0.0  # Track distance-away penalty
            incremental_reward_weight = self.config.get('incremental_reward_weight', 4.0)  # Balanced weight
            # Removed spike_sparsity_weight - was overwhelming the model
            self.epoch_stay_actions = 0
            self.epoch_total_actions = 0
            self.epoch_avg_stay_prob = 0.0
            # Adjacent coordination ratio no longer tracked in simplified implementation
            num_batches = 0

            for batch_idx, (batch_fovs, batch_actions, batch_positions, batch_goals, batch_expert_actions) in enumerate(train_loader):
                # Clear oscillation history at the start of each batch to prevent false detections
                from loss_compute import clear_oscillation_history
                clear_oscillation_history()  # This now clears both simple and enhanced oscillation history

                batch_fovs = batch_fovs.to(self.device)
                batch_actions = batch_actions.to(self.device)  # Original trajectory_record actions (not used for loss)
                batch_positions = batch_positions.to(self.device)
                batch_goals = batch_goals.to(self.device)
                batch_expert_actions = batch_expert_actions.to(self.device)  # True expert actions from solution.yaml

                # Get corresponding obstacles for this batch
                # Note: We need to map batch indices to original sequence indices
                batch_start_idx = batch_idx * self.batch_size
                batch_end_idx = min(batch_start_idx + self.batch_size, len(self.train_obstacles))
                batch_obstacles = self.train_obstacles[batch_start_idx:batch_end_idx]

                # Reset SNN state
                safe_reset_snn(self.model)

                batch_loss = 0.0
                sequence_outputs = []
                total_batch_stay_penalty = 0.0  # Track total stay penalty for batch
                total_batch_oscillation_penalty = 0.0  # Track total oscillation penalty for batch

                seq_len = batch_fovs.shape[1]

                # TBPTT (Truncated Back-Propagation Through Time) setup
                # Problem: Unrolling whole 60-step episode causes gradient vanishing and slow CUDA compilation
                # Solution: Use TBPTT with window=12 for 5-7x stronger learning signal
                tbptt_window = self.config.get('tbptt_window', 12)
                tbptt_accumulated_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                tbptt_step_count = 0

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

                    # Forward pass with spike monitoring
                    if self.spike_tracker.enabled:
                        # Get model output with spike information
                        output, spike_info = self.model(fov_t, return_spikes=True, positions=None, goals=batch_goals)

                        # Record spike activity for each layer
                        if spike_info:
                            # Debug print to see what's in spike_info
                            if batch_idx == 0 and t == 0:
                                print(f"   🔍 DEBUG spike_info keys: {list(spike_info.keys())}")
                                for key, value in spike_info.items():
                                    if isinstance(value, torch.Tensor):
                                        print(f"   🔍 DEBUG {key}: shape={value.shape}, mean={value.mean().item():.6f}, max={value.max().item():.6f}")

                            self.spike_tracker('global_features', spike_info.get('global_features'))
                            self.spike_tracker('snn_output', spike_info.get('snn_output'))
                            self.spike_tracker('action_logits', spike_info.get('action_logits'))


                            # Track raw output values for model_output (for debugging only, no threshold adjustment)
                            # Use action_logits (100 neurons) as model_output for spike tracking
                            model_output_100 = spike_info.get('action_logits')
                            self.spike_tracker('model_output', model_output_100)

                            # Debug: Log real output magnitudes every few batches
                            #if batch_idx % 10 == 0:
                                #print(f"🔍 Output range: {output.abs().min().item():.3f}-{output.abs().max().item():.3f}, mean: {output.abs().mean().item():.3f}")

                            # Store spike output for sparsity regularization
                            self._last_spike_output = spike_info.get('snn_output')

                        # Also track SNN framework layers (same as pretraining)
                        if hasattr(self.model, 'snn_block'):
                            for name, module in self.model.snn_block.named_modules():
                                if hasattr(module, '_last_spike_output'):
                                    try:
                                        # Get spike output from AdaptiveLIFNode
                                        membrane_spikes = module._last_spike_output
                                        if isinstance(membrane_spikes, torch.Tensor) and membrane_spikes.numel() > 0:
                                            # Register layer and record spikes
                                            layer_name = f'snn_{name}'
                                            self.spike_tracker.monitor.register_layer(layer_name)
                                            self.spike_tracker(layer_name, membrane_spikes)
                                    except Exception as e:
                                        # Handle temporal_lif IndexError and other membrane tracking issues
                                        if "temporal_lif" in name and "tuple index out of range" in str(e):
                                            if t == 0 and batch_idx == 0:
                                                print(f"   ⚠️ Membrane tracking warning for {name}: {e}")
                                        elif t == 0 and batch_idx == 0:
                                            print(f"   ⚠️ SNN layer {name} tracking error: {e}")
                    else:
                        # Standard forward pass without spike monitoring
                        output = self.model(fov_t, positions=None, goals=batch_goals)
                        self._last_spike_output = None

                    output = output.reshape(batch_fovs.shape[0], -1, self.model.num_classes)

                    # Debug utilities for model output validation
                    if self.debug_enabled and batch_idx == 0 and t == 0:  # First batch, first timestep
                        expected_shape = (batch_fovs.shape[0], batch_fovs.shape[2], self.model.num_classes)
                        validate_model_output(output, expected_shape, f"model_output_epoch_{epoch + 1}")

                    sequence_outputs.append(output)

                    # Compute loss for this timestep
                    targets = batch_expert_actions[:, t]  # [batch, agents] - TRUE expert actions from solution.yaml

                    # CRITICAL FIX: Check if we're beyond the expert trajectory length
                    # Expert trajectories are typically 10-15 steps, but model runs for 60 steps
                    # After expert trajectory ends, we should stop computing loss to avoid learning from empty/default actions

                    # More robust check: expert trajectory has ended if:
                    # 1. All agents have stay actions (0), AND we're past minimum length, OR
                    # 2. We're past the expected expert trajectory length (more reliable)
                    min_trajectory_length = 8   # Minimum expected trajectory length
                    # SIMPLIFIED EXPERT END DETECTION: Switch to RL immediately when expert ends
                    expected_expert_length = self.config.get('expected_expert_trajectory_length', 16)

                    # Expert trajectory ends when we reach the expected length OR all agents have stay actions
                    expert_trajectory_ended = (t >= expected_expert_length) or torch.all(targets == 0)

                    # Log when expert trajectory ends (once per batch)
                    if expert_trajectory_ended and batch_idx == 0 and not hasattr(self, '_trajectory_end_logged'):
                        if t >= expected_expert_length:
                            print(f"   🎯 Expert trajectory ended at timestep {t} (reached expected length {expected_expert_length}) - SWITCHING TO PURE RL")
                        else:
                            print(f"   🎯 Expert trajectory ended at timestep {t} (all agents have stay actions) - SWITCHING TO PURE RL")
                        self._trajectory_end_logged = True
                    elif not expert_trajectory_ended and hasattr(self, '_trajectory_end_logged'):
                        delattr(self, '_trajectory_end_logged')

                    # PURE RL MODE: Once expert ends, use ONLY model predictions for the rest of the episode

                    # Initialize timestep loss
                    timestep_loss = 0.0

                    # Check which agents have reached their goals (stop training them!)
                    current_positions = current_batch_positions  # [batch, agents, 2] - tracked positions
                    goal_distances = torch.norm(current_positions - batch_goals, dim=-1)  # [batch, agents]
                    goal_tolerance = self.config.get('goal_tolerance', 0.8)
                    agents_at_goal = goal_distances <= goal_tolerance  # [batch, agents]

                    # Get model's predicted actions FIRST (before applying rewards/penalties)
                    predicted_actions = torch.argmax(output, dim=-1)  # [batch, agents]

                    # Calculate moved_away_mask FIRST (needed for both distance reward and penalty)
                    moved_away_mask = None  # Track which agents moved away from goals
                    if self.config.get('use_distance_away_penalty', True) and t > 0:
                        # Calculate if agents moved away from goals compared to previous timestep
                        if hasattr(self, 'prev_goal_distances'):
                            distance_increase = goal_distances - self.prev_goal_distances
                            # Only penalize when distance actually increased (moved away)
                            moved_away_mask = distance_increase > 0.1  # Small threshold to avoid noise

                    # Add DIRECT distance-based reward to make goal-seeking CRYSTAL CLEAR
                    # This gives immediate reward for being closer to goal, making the objective obvious
                    # ONLY apply rewards/penalties during expert trajectory (not after it ends)
                    if self.config.get('use_direct_distance_reward', True) and not expert_trajectory_ended:
                        # Calculate inverse distance reward (closer = higher reward)
                        max_distance = 20.0  # Maximum possible distance on grid
                        distance_reward_weight = self.config.get('direct_distance_reward_weight', 8.0)  # Reduced from 100.0
                        inverse_distance_reward = distance_reward_weight * (max_distance - goal_distances) / max_distance

                        # Don't reward proximity if we're about to penalize for moving away
                        if moved_away_mask is not None and moved_away_mask.any():
                            # Zero out rewards for agents moving away from goals
                            inverse_distance_reward = inverse_distance_reward * (~moved_away_mask).float()

                            # Log how many agents had their direct distance rewards blocked
                            blocked_agents = moved_away_mask.sum().item()
                            total_agents = moved_away_mask.numel()
                            if batch_idx == 0 and t % 20 == 0:  # Log every 20 timesteps
                                print(f"   🚫 Direct distance rewards blocked for {blocked_agents}/{total_agents} agents moving away")

                        # Apply reward through loss function (not direct logit manipulation)
                        # This preserves SNN's natural learning dynamics
                        direct_distance_loss = -inverse_distance_reward.mean()
                        timestep_loss = timestep_loss + direct_distance_loss  # Avoid in-place operation

                        # Log the direct distance reward occasionally
                        if batch_idx == 0 and t % 10 == 0:  # Log every 10 timesteps for first batch
                            avg_distance = goal_distances.mean().item()
                            avg_reward = inverse_distance_reward.mean().item()
                            print(f"   🎯 Direct Distance Reward: avg_dist={avg_distance:.2f}, reward={avg_reward:.2f}")

                    # Add MASSIVE penalty for moving away from goal (decreases over epochs)
                    # ONLY apply during expert trajectory (not after it ends)
                    distance_away_penalty_applied = 0.0  # Track penalty for this timestep
                    if self.config.get('use_distance_away_penalty', True) and t > 0 and not expert_trajectory_ended:
                        # Calculate current penalty weight with epoch-based decay
                        initial_penalty_weight = self.config.get('distance_away_penalty_weight', 6.0)  # Balanced weight
                        decay_rate = self.config.get('distance_away_penalty_decay_rate', 0.95)
                        min_penalty_weight = self.config.get('distance_away_penalty_min_weight', 2.0)  # Reduced from 50.0

                        # Apply slow exponential decay: weight = initial * (decay_rate ^ epoch)
                        current_penalty_weight = max(
                            initial_penalty_weight * (decay_rate ** epoch),
                            min_penalty_weight
                        )

                        # Use the moved_away_mask computed above
                        if moved_away_mask is not None and moved_away_mask.any():
                            # Get the distance increase for penalty calculation
                            distance_increase = goal_distances - self.prev_goal_distances

                            # Apply penalty through loss function (not direct logit manipulation)
                            # This preserves SNN's natural learning dynamics
                            distance_away_penalty = current_penalty_weight * distance_increase[moved_away_mask].mean()
                            timestep_loss += distance_away_penalty
                            distance_away_penalty_applied = distance_away_penalty.item()

                            # Log distance-away penalty occasionally
                            if batch_idx == 0 and t % 20 == 0:  # Log every 20 timesteps
                                agents_moved_away = moved_away_mask.sum().item()
                                avg_distance_increase = distance_increase[moved_away_mask].mean().item()
                                print(f"   🚫 Distance-Away Penalty: {agents_moved_away} agents moved away, "
                                      f"avg_increase={avg_distance_increase:.2f}, penalty_weight={current_penalty_weight:.1f}")

                        # Store current distances for next timestep comparison
                        self.prev_goal_distances = goal_distances.clone().detach()

                    # Initialize distance tracking for first timestep
                    if t == 0:
                        self.prev_goal_distances = goal_distances.clone().detach()

                    # Mask out targets for agents who reached goals (set to -1 to ignore in loss)
                    masked_targets = targets.clone()
                    masked_targets[agents_at_goal] = -1  # -1 will be ignored in loss computation

                    goals_reached_count = torch.sum(agents_at_goal).item()
                    if goals_reached_count > 0 and batch_idx == 0:  # Log only for first batch
                        debug_print(f"   🎯 {goals_reached_count} agents reached goals at timestep {t}, stopping their training")

                    # SIMPLE LOGIC: Use expert targets when available, model predictions when expert trajectory ends
                    actions_for_position_update = predicted_actions  # Always use model predictions for position updates

                    # SWITCH TO PURE RL: Once expert trajectory ends, use model predictions as targets
                    if expert_trajectory_ended:
                        # PURE RL MODE: Use model predictions as targets (learn from own actions)
                        loss_targets = predicted_actions.long()  # Use model's own predictions as targets
                        if batch_idx == 0 and not hasattr(self, '_rl_mode_logged'):
                            print(f"   🤖 PURE RL MODE: Using model predictions as targets from timestep {t} onwards")
                            self._rl_mode_logged = True
                    else:
                        # EXPERT MODE: Use expert actions as targets (imitation learning)
                        loss_targets = masked_targets.long()  # Use expert actions from solution.yaml

                    # ALWAYS compute loss - whether from expert or model predictions
                    valid_agents = 0
                    for agent in range(loss_targets.shape[1]):
                        # Skip agents who reached goals (targets == -1)
                        agent_targets = loss_targets[:, agent]
                        valid_mask = agent_targets >= 0
                        if valid_mask.sum() > 0:  # Only compute loss for valid targets
                            agent_loss = self.criterion(output[:, agent][valid_mask], agent_targets[valid_mask])
                            timestep_loss = timestep_loss + agent_loss  # Avoid in-place operation
                            valid_agents += 1

                    if valid_agents > 0:
                        timestep_loss /= valid_agents  # Average over valid agents only
                    else:
                        timestep_loss = torch.tensor(0.0, device=self.device)  # No valid agents

                    # Log training mode
                    if batch_idx == 0 and t % 10 == 0:  # Log occasionally for first batch
                        if expert_trajectory_ended:
                            print(f"   🤖 PURE RL mode at timestep {t}, learning from model predictions for {valid_agents} agents")
                        else:
                            print(f"   📚 Expert mode at timestep {t}, learning from expert actions for {valid_agents} agents")

                    # Removed spike sparsity regularization - was overwhelming the model
                    # Instead, the model uses adaptive thresholds in the SNN layers

                    # Add entropy bonus for exploration (encourage diverse actions)
                    entropy_bonus = 0.0
                    if entropy_bonus_weight > 0:
                        # Flatten logits for all agents at this timestep
                        logits_flat = output.reshape(-1, self.model.num_classes)
                        entropy_bonus = compute_entropy_bonus(logits_flat)
                        timestep_loss = timestep_loss - entropy_bonus_weight * entropy_bonus  # Subtract to encourage high entropy - avoid in-place
                        epoch_entropy_bonus += entropy_bonus.item()

                    # TBPTT (Truncated Back-Propagation Through Time) implementation
                    # Accumulate gradients for tbptt_window steps, then apply optimizer
                    tbptt_accumulated_loss = tbptt_accumulated_loss + timestep_loss  # Avoid in-place operation
                    tbptt_step_count += 1

                    # Backward pass every tbptt_window steps or at sequence end
                    if tbptt_step_count % tbptt_window == 0 or t == seq_len - 1:
                        # Average accumulated loss over steps
                        avg_tbptt_loss = tbptt_accumulated_loss / tbptt_step_count

                        # Backward pass
                        avg_tbptt_loss.backward()

                        # Adaptive gradient clipping - only clip if gradients are truly exploding
                        max_norm = self.config.get('gradient_clip_max_norm', 2.0)
                        # First measure unclipped gradient norm without modifying gradients
                        unclipped_grad_norm = sum(p.grad.norm().item() ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5

                        # Only apply clipping if gradients are significantly large (more than 3x threshold)
                        explosion_threshold = max_norm * 3.0  # Only clip if > 6.0 (was > 3.0)
                        if unclipped_grad_norm > explosion_threshold:
                            # Aggressive clipping for true gradient explosion
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                            # Recompute gradient norm after clipping
                            clipped_grad_norm = sum(
                                p.grad.norm().item() ** 2
                                for p in self.model.parameters() if p.grad is not None
                            ) ** 0.5
                            grad_norm = clipped_grad_norm
                            if batch_idx % 25 == 0:
                                print(f"   🚨 TBPTT EXPLOSION! Gradient clipped: {unclipped_grad_norm:.3f} → {grad_norm:.3f}")
                        else:
                            # No clipping - let gradients flow naturally
                            grad_norm = unclipped_grad_norm
                            if batch_idx % 50 == 0:
                                print(f"   📊 TBPTT gradient norm: {grad_norm:.3f} (no clipping needed)")

                        # Log detailed gradient info
                        if batch_idx % 50 == 0:
                            # Check specific layer gradients that might be getting suppressed
                            for name, param in self.model.named_parameters():
                                if param.grad is not None and 'global_features' in name:
                                    layer_grad_norm = param.grad.norm().item()
                                    print(f"   🧠 {name} grad norm: {layer_grad_norm:.6f}")

                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # Reset SNN state but keep hidden states (detach computation graph)
                        self.model.reset_state(detach=True)  # keep hidden but detach graph

                        # Track loss for epoch averaging
                        batch_loss += tbptt_accumulated_loss.item()

                        # Reset TBPTT accumulators
                        tbptt_accumulated_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        tbptt_step_count = 0

                        if batch_idx == 0 and t % (tbptt_window * 2) == 0:  # Log occasionally
                            print(f"   🔄 TBPTT step at timestep {t}, avg_loss: {avg_tbptt_loss.item():.4f}")

                    # Accumulate distance-away penalty for epoch tracking
                    epoch_distance_away_penalty += distance_away_penalty_applied

                    # Update position tracking for incremental rewards and proximity reward
                    if t > 0:  # Skip first timestep (no previous position)
                        previous_batch_positions = current_batch_positions.clone()
                        # Always use predicted actions for position updates in multi-sequence training
                        # This ensures rewards/losses are applied to the agent's chosen sequence
                        current_batch_positions = self._update_positions_from_actions(
                            current_batch_positions, actions_for_position_update
                        )

                        # Log training approach occasionally
                        if batch_idx % 20 == 0 and t == 1:  # Log once per batch
                            print(f"🎯 Multi-sequence training: agents follow predictions, learn from expert actions")

                        # Apply progress shaping reward: reward -Δ(distance-to-goal) each step
                        # This keeps gradients alive once collisions are solved
                        # ONLY apply during expert trajectory (not after it ends)
                        if not expert_trajectory_ended and self.config.get('use_progress_shaping_reward', True):
                            progress_shaping_reward = compute_progress_shaping_reward(
                                current_batch_positions, previous_batch_positions, batch_goals,
                                progress_weight=self.config.get('progress_shaping_weight', 5.0)
                            )
                            # Apply progress shaping reward through loss function (preserves SNN dynamics)
                            progress_shaping_loss = -progress_shaping_reward.mean()  # Negative because reward reduces loss
                            timestep_loss = timestep_loss + progress_shaping_loss  # Avoid in-place operation

                            # Log progress shaping reward occasionally
                            if batch_idx == 0 and t % 10 == 0:  # Log every 10 timesteps for first batch
                                avg_progress_reward = progress_shaping_reward.mean().item()
                                print(f"   📈 Progress Shaping Reward: {avg_progress_reward:.4f}")

                        # Compute incremental rewards for movement dynamics
                        # These rewards are based on the agent's actual chosen actions
                        # ONLY apply during expert trajectory (not after it ends)
                        if incremental_reward_weight > 0 and not expert_trajectory_ended:
                            incremental_rewards = compute_incremental_rewards(
                                current_batch_positions, previous_batch_positions, batch_goals
                            )
                            # Apply incremental rewards through loss function (preserves SNN dynamics)
                            incremental_loss = -incremental_reward_weight * incremental_rewards['total_incremental']
                            timestep_loss = timestep_loss + incremental_loss  # Add to timestep_loss for TBPTT accumulation - avoid in-place

                            # Track individual reward components
                            epoch_goal_approach_reward += incremental_rewards['goal_approach'].item()
                            epoch_collision_avoidance_reward += incremental_rewards['collision_avoidance'].item()
                            epoch_grid_movement_reward += incremental_rewards['grid_movement'].item()
                            epoch_movement_away_reward += incremental_rewards['movement_away'].item()

                        # Apply oscillation penalty per timestep based on predicted actions
                        # ONLY apply during expert trajectory (not after it ends)
                        if not expert_trajectory_ended:
                            if self.use_enhanced_oscillation_detection:
                                from loss_compute import compute_enhanced_oscillation_penalty, compute_consecutive_stay_penalty
                                oscillation_weight = self.oscillation_penalty_weight.item()
                                timestep_oscillation_penalty = compute_enhanced_oscillation_penalty(
                                    current_batch_positions, batch_id=batch_idx, oscillation_weight=oscillation_weight,
                                    goal_positions=batch_goals  # Pass goals to exclude agents that reached them
                                )
                            else:
                                from loss_compute import compute_simple_oscillation_penalty, compute_consecutive_stay_penalty
                                oscillation_weight = self.oscillation_penalty_weight.item()
                                timestep_oscillation_penalty = compute_simple_oscillation_penalty(
                                    current_batch_positions, batch_id=batch_idx, oscillation_weight=oscillation_weight,
                                    goal_positions=batch_goals  # Pass goals to exclude agents that reached them
                                )

                            # Add consecutive stay penalty per timestep based on predicted actions
                            # DISABLED - let agents decide freely when they need to stay
                            # timestep_stay_penalty = compute_consecutive_stay_penalty(
                            #     actions_for_position_update, batch_id=batch_idx, stay_penalty_weight=4.0, max_consecutive_stays=3  # Reduced from 5.0
                            # )
                            timestep_stay_penalty = torch.tensor(0.0, device=self.device)  # DISABLED - let agents decide freely

                            # Add penalties to this timestep's loss
                            if timestep_oscillation_penalty.item() > 0.001:
                                timestep_loss = timestep_loss + timestep_oscillation_penalty  # Add to timestep_loss for TBPTT - avoid in-place
                                total_batch_oscillation_penalty += timestep_oscillation_penalty.item()  # Accumulate for batch total

                            # DISABLED stay penalty - let agents decide freely when they need to stay
                            # if timestep_stay_penalty.item() > 0.001:
                            #     timestep_loss = timestep_loss + timestep_stay_penalty  # Add to timestep_loss for TBPTT - avoid in-place
                            #     total_batch_stay_penalty += timestep_stay_penalty.item()  # Accumulate for batch total
                        else:
                            # Expert trajectory ended - no oscillation/stay penalties
                            timestep_oscillation_penalty = torch.tensor(0.0, device=self.device)
                            timestep_stay_penalty = torch.tensor(0.0, device=self.device)

                # TBPTT handles loss averaging internally
                # batch_loss will contain accumulated .item() values from TBPTT steps

                # Compute additional losses using centralized loss computation
                sequence_outputs_tensor = torch.stack(sequence_outputs, dim=1)  # [batch, seq_len, agents, action_dim]

                # Note: Rewards/penalties are now applied directly to model outputs in the timestep loop above
                # The compute_all_losses function will handle only communication, coordination, and collision losses
                batch_size, num_agents = current_batch_positions.shape[:2]
                collided_agents = None  # Don't track collision for reward exclusion anymore

                # Still detect and display collisions for monitoring (but don't use for reward exclusion)
                try:
                    from collision_utils import detect_collisions
                    # Get current and previous positions for collision detection
                    current_positions = current_batch_positions
                    if previous_batch_positions is not None:
                        prev_positions = previous_batch_positions
                    else:
                        prev_positions = current_positions  # First timestep, no previous positions

                    # Detect collisions for monitoring only
                    collisions = detect_collisions(current_positions, prev_positions)
                    vertex_collisions = collisions['vertex_collisions']  # [batch, agents]
                    edge_collisions = collisions['edge_collisions']      # [batch, agents]

                    # Count collisions for display
                    collision_agents = (vertex_collisions > 0) | (edge_collisions > 0)
                    collision_count = torch.sum(collision_agents).item()
                    # Always show collision count for monitoring
                    if collision_count > 0:
                        print(f"🚫 Collisions: {collision_count} agents")

                    # Show batch total penalties if any occurred
                    if total_batch_oscillation_penalty > 0.001:
                        print(f"🔄 Total batch oscillation penalty: {total_batch_oscillation_penalty:.3f}")
                    # DISABLED stay penalty reporting - agents can now decide freely when to stay
                    # if total_batch_stay_penalty > 0.001:
                    #     print(f"🛑 Total batch consecutive stay penalty: {total_batch_stay_penalty:.3f}")

                except Exception as e:
                    print(f"⚠️  Collision detection failed during monitoring: {e}")

                # Debug utilities for position analysis
                if self.debug_enabled and batch_idx % 20 == 0:  # Every 20 batches
                    position_analysis = analyze_agent_positions(current_batch_positions, self.grid_size)
                    anomalies = detect_position_anomalies(current_batch_positions, self.grid_size)

                    if anomalies:
                        debug_print(f"Position anomalies in batch {batch_idx}: {anomalies}")

                    debug_print(f"Position stats: agents={position_analysis['num_positions']}, "
                              f"x_range={position_analysis['x_range']}, y_range={position_analysis['y_range']}, "
                              f"oob={position_analysis['out_of_bounds']}")

                # Store current positions for next timestep collision detection
                # Store current positions as previous for next iteration
                previous_batch_positions = current_batch_positions.clone().detach()

                # Compute sequence-level losses (communication, coordination, collision)
                # Note: Distance/movement rewards/penalties are applied directly to sequence outputs above
                loss_weights = {
                    'communication_weight': self.communication_weight,
                    'coordination_weight': self.coordination_weight,
                    'temporal_consistency_weight': self.temporal_consistency_weight
                }

                loss_results = compute_all_losses(
                    sequence_outputs_tensor, current_batch_positions, batch_goals,
                    loss_weights, self.config, self.device,
                    collided_agents=collided_agents,  # Pass collision info
                    prev_positions=getattr(self, 'previous_batch_positions', None),  # Pass previous positions
                    stress_inhibition_weight=torch.tensor(0.0, device=self.device),  # DISABLE stress inhibition - let agents decide freely
                    selective_movement_weight=torch.tensor(0.0, device=self.device),  # DISABLE selective movement - let agents decide freely
                    oscillation_penalty_weight=self.oscillation_penalty_weight,  # Pass learnable oscillation weight
                    batch_id=batch_idx,  # Use batch_idx as batch_id for tracking
                    epoch=epoch,  # Pass current epoch for collision weight annealing
                    obstacles=batch_obstacles[0] if batch_obstacles else [],  # Pass obstacles for obstacle-aware progress
                    moved_away_mask=moved_away_mask  # Pass mask of agents moving away from goals
                )

                # Extract individual losses for tracking
                comm_loss = loss_results['communication_loss']
                coord_loss = loss_results['coordination_loss']
                temporal_loss = loss_results['temporal_loss']
                collision_loss = loss_results['collision_loss']
                # Oscillation penalty now applied per timestep in the training loop above
                progress_reward = loss_results['progress_reward']
                stay_reward = loss_results['stay_reward']
                movement_bonus = loss_results.get('movement_bonus', 0.0)
                collision_weight = loss_results.get('collision_weight', 1.0)
                stress_inhibition_loss = loss_results['stress_inhibition_loss']
                selective_movement_loss = loss_results['selective_movement_loss']

                # Compute STDP coordination rewards (now allowing all agents to receive rewards)
                # NOTE: These are applied as scalar rewards since they're computed after sequence finalization
                coordination_rewards = self.model.compute_coordination_rewards(
                    positions=current_batch_positions,
                    goals=batch_goals,
                    batch_size=batch_fovs.shape[0],
                    collided_agents=None  # Don't exclude any agents from receiving rewards
                )
                stdp_reward = coordination_rewards['total_reward']
                stdp_metrics = coordination_rewards['metrics']

                # Extract stress-related and stay action statistics
                stress_stats = loss_results['stress_stats']
                movement_stats = loss_results['movement_stats']
                stay_stats = loss_results['stay_stats']

                # Accumulate progress reward, stay reward, movement bonus, and stress metrics for epoch tracking
                epoch_progress_reward += progress_reward
                epoch_stay_reward = getattr(self, 'epoch_stay_reward', 0.0) + stay_reward.item() if isinstance(stay_reward, torch.Tensor) else 0.0
                epoch_movement_bonus = getattr(self, 'epoch_movement_bonus', 0.0) + movement_bonus.item() if isinstance(movement_bonus, torch.Tensor) else movement_bonus
                epoch_stress_inhibition += stress_inhibition_loss.item() if isinstance(stress_inhibition_loss, torch.Tensor) else 0.0
                epoch_selective_movement += selective_movement_loss.item() if isinstance(selective_movement_loss, torch.Tensor) else 0.0

                # Track stay action statistics
                epoch_stay_actions = getattr(self, 'epoch_stay_actions', 0) + stay_stats.get('total_stay_actions', 0)
                epoch_total_actions = getattr(self, 'epoch_total_actions', 0) + stay_stats.get('total_actions', 0)
                epoch_avg_stay_prob = getattr(self, 'epoch_avg_stay_prob', 0.0) + stay_stats.get('avg_stay_probability', 0.0)

                # Store for next iteration
                self.epoch_stay_reward = epoch_stay_reward
                self.epoch_movement_bonus = epoch_movement_bonus
                self.epoch_stay_actions = epoch_stay_actions
                self.epoch_total_actions = epoch_total_actions
                self.epoch_avg_stay_prob = epoch_avg_stay_prob

                # Track stress zone detection and intensity
                if stress_stats.get('stress_zones_detected', False):
                    epoch_stress_zones += 1
                    epoch_stress_intensity += stress_stats.get('avg_stress_intensity', 0.0)

                # Accumulate STDP coordination metrics
                if isinstance(stdp_reward, torch.Tensor):
                    # Take mean across batch and agents, then convert to scalar
                    epoch_stdp_reward += stdp_reward.mean().item()
                else:
                    epoch_stdp_reward += stdp_reward
                epoch_coordination_efficiency += stdp_metrics.get('coordination_efficiency', 0.0)
                epoch_goal_proximity_bonus += stdp_metrics.get('goal_proximity_bonus', 0.0)
                epoch_conflict_resolution += stdp_metrics.get('conflict_resolution_bonus', 0.0)

                # Total loss: Include sequence-level losses + position-based rewards + STDP coordination rewards
                # Position-based rewards are now applied through loss function (preserves SNN dynamics)
                # DISABLED stress inhibition and selective movement - let agents decide freely when close
                total_loss = (batch_loss +
                             comm_loss +  # Already weighted in compute_all_losses
                             coord_loss +  # Already weighted in compute_all_losses
                             temporal_loss +  # Already weighted in compute_all_losses
                             collision_loss +  # Already weighted in compute_all_losses
                             # stress_inhibition_loss +  # DISABLED - let agents decide freely when close
                             # selective_movement_loss +  # DISABLED - let agents decide freely when close
                             -progress_reward -  # Subtract progress reward (rewards reduce loss)
                             stay_reward -  # Subtract stay reward (rewards reduce loss)
                             -(stdp_reward.mean() if isinstance(stdp_reward, torch.Tensor) else stdp_reward))  # Subtract STDP coordination reward

                # Oscillation penalties are now logged per timestep when they occur

                # With TBPTT, optimization already happened during timestep loop
                # Just accumulate epoch-level tracking losses for logging

                # Final reset after batch (different from TBPTT detach resets)
                # This ensures clean state between mini-batches
                self.model.reset_state()  # fresh brains next batch

                epoch_loss += batch_loss  # batch_loss already accumulated from TBPTT steps
                epoch_comm_loss += comm_loss.item()
                epoch_coord_score += (1.0 - coord_loss.item())  # Higher is better for coordination
                epoch_collision_loss += collision_loss.item()  # Track collision loss
                # Oscillation penalty now applied per timestep, not tracked in epoch summary
                # Adjacent coordination ratio no longer tracked in simplified implementation
                num_batches += 1

                # Update visualization (process-safe, non-blocking)
                if (self.visualize_training and batch_idx % self.vis_update_freq == 0 and 
                    self.animate_episodes and batch_idx % self.vis_episode_frequency == 0):
                    with torch.no_grad():
                        self.model.eval()

                        #1. simulate the full episode using the SNN's predicted actions
                        sim_initial_pos = batch_positions[0].clone()
                        sim_goals = batch_goals[0]
                        sim_obstacles = batch_obstacles[0] if batch_obstacles and len(batch_obstacles) > 0 else []

                        snn_positions = sim_initial_pos
                        trajectory = [snn_positions.cpu().numpy()]
                        safe_reset_snn(self.model)

                        for t in range(self.sequence_length):
                            fov_t = batch_fovs[0:1, t].to(self.device)  # [1, num_agents, channels, h, w] - keep batch dimension
                            output = self.model(fov_t, goals=sim_goals.unsqueeze(0))  # Add batch dimension: [1, num_agents, 2]
                            predicted_actions = torch.argmax(output, dim=-1)  # [1, num_agents] -> need to squeeze
                            predicted_actions = predicted_actions.squeeze(0)  # [num_agents]
                            snn_positions = self._update_positions_from_actions(snn_positions.unsqueeze(0), predicted_actions.unsqueeze(0)).squeeze(0)
                            trajectory.append(snn_positions.cpu().numpy())
                        self.model.train()

                        # Send SNN action sequence to new 4-graph visualizer
                        snn_actions_array = []
                        for t in range(self.sequence_length):
                            fov_t = batch_fovs[0:1, t].to(self.device)  # [1, num_agents, channels, h, w] - keep batch dimension
                            with torch.no_grad():
                                output = self.model(fov_t, goals=sim_goals.unsqueeze(0))  # Add batch dimension: [1, num_agents, 2]
                                predicted_actions = torch.argmax(output, dim=-1).squeeze(0).cpu().numpy()  # [num_agents]
                                snn_actions_array.append(predicted_actions)
                        
                        # Send to new visualizer for SNN action animation
                        if self.visualizer is not None:
                            self.visualizer.show_episode(
                                initial_positions=sim_initial_pos.cpu().numpy(),
                                goals=sim_goals.cpu().numpy(), 
                                obstacles=sim_obstacles,
                                snn_actions=snn_actions_array,  # This is the SNN action sequence
                                case_info=f'Epoch{epoch + 1}, Batch{batch_idx}'
                            )

                        print(f"✅ SNN action animation sent to visualizer for batch {batch_idx}")
                
                # Compute epoch averages
            avg_loss = epoch_loss / num_batches
            avg_comm_loss = epoch_comm_loss / num_batches
            avg_coord_score = epoch_coord_score / num_batches
            avg_entropy_bonus = epoch_entropy_bonus / (num_batches * seq_len) if num_batches > 0 and seq_len > 0 else 0.0
            avg_progress_reward = epoch_progress_reward / (num_batches * seq_len) if num_batches > 0 and seq_len > 0 else 0.0
            avg_collision_loss = epoch_collision_loss / num_batches if num_batches > 0 else 0.0
            # Oscillation penalty now applied per timestep, not tracked in epoch summary
            avg_stress_inhibition = epoch_stress_inhibition / num_batches if num_batches > 0 else 0.0
            avg_selective_movement = epoch_selective_movement / num_batches if num_batches > 0 else 0.0
            avg_stress_intensity = epoch_stress_intensity / epoch_stress_zones if epoch_stress_zones > 0 else 0.0
            stress_detection_rate = epoch_stress_zones / num_batches if num_batches > 0 else 0.0

            # Compute STDP coordination averages
            avg_stdp_reward = epoch_stdp_reward / num_batches if num_batches > 0 else 0.0
            avg_coordination_efficiency = epoch_coordination_efficiency / num_batches if num_batches > 0 else 0.0
            avg_goal_proximity_bonus = epoch_goal_proximity_bonus / num_batches if num_batches > 0 else 0.0
            avg_conflict_resolution = epoch_conflict_resolution / num_batches if num_batches > 0 else 0.0

            # Compute stay action reward and movement bonus averages
            avg_stay_reward = self.epoch_stay_reward / num_batches if num_batches > 0 else 0.0
            avg_movement_bonus = self.epoch_movement_bonus / num_batches if num_batches > 0 else 0.0
            stay_action_rate = self.epoch_stay_actions / self.epoch_total_actions if self.epoch_total_actions > 0 else 0.0
            avg_stay_prob = self.epoch_avg_stay_prob / num_batches if num_batches > 0 else 0.0

            # Compute incremental reward averages
            avg_goal_approach_reward = epoch_goal_approach_reward / (num_batches * seq_len) if num_batches > 0 and seq_len > 0 else 0.0
            avg_collision_avoidance_reward = epoch_collision_avoidance_reward / (num_batches * seq_len) if num_batches > 0 and seq_len > 0 else 0.0
            avg_grid_movement_reward = epoch_grid_movement_reward / (num_batches * seq_len) if num_batches > 0 and seq_len > 0 else 0.0
            avg_movement_away_reward = epoch_movement_away_reward / (num_batches * seq_len) if num_batches > 0 and seq_len > 0 else 0.0
            # Removed spike sparsity loss computation - was overwhelming the model
            avg_distance_away_penalty = epoch_distance_away_penalty / (num_batches * seq_len) if num_batches > 0 and seq_len > 0 else 0.0

            # Calculate current distance-away penalty weight for display
            if self.config.get('use_distance_away_penalty', True):
                initial_penalty_weight = self.config.get('distance_away_penalty_weight', 6.0)  # Balanced weight
                decay_rate = self.config.get('distance_away_penalty_decay_rate', 0.95)
                min_penalty_weight = self.config.get('distance_away_penalty_min_weight', 50.0)
                current_distance_penalty_weight = max(
                    initial_penalty_weight * (decay_rate ** epoch),
                    min_penalty_weight
                )
            else:
                current_distance_penalty_weight = 0.0

            total_incremental_reward = avg_goal_approach_reward + avg_collision_avoidance_reward + avg_grid_movement_reward + avg_movement_away_reward

            # Adjacent coordination ratio no longer tracked in simplified implementation

            # Update training history (add stay action tracking)
            self.training_history['epoch_losses'].append(avg_loss)
            self.training_history['communication_losses'].append(avg_comm_loss)
            self.training_history['coordination_scores'].append(avg_coord_score)
            self.training_history['entropy_bonuses'].append(avg_entropy_bonus)
            self.training_history['progress_rewards'].append(avg_progress_reward)
            # Add stay action reward tracking to history
            self.training_history['stay_rewards'].append(avg_stay_reward)
            self.training_history['stay_action_rates'].append(stay_action_rate)
            self.training_history['avg_stay_probabilities'].append(avg_stay_prob)

            self.training_history['collision_losses'].append(avg_collision_loss)
            # Oscillation penalty now applied per timestep, not tracked in epoch history
            self.training_history['stress_inhibition_losses'].append(avg_stress_inhibition)
            self.training_history['selective_movement_losses'].append(avg_selective_movement)
            self.training_history['stress_zones_detected'].append(stress_detection_rate)
            self.training_history['avg_stress_intensity'].append(avg_stress_intensity)
            # Add STDP coordination metrics to history
            self.training_history['stdp_rewards'].append(avg_stdp_reward)
            self.training_history['coordination_efficiency'].append(avg_coordination_efficiency)
            self.training_history['goal_proximity_bonus'].append(avg_goal_proximity_bonus)
            self.training_history['conflict_resolution'].append(avg_conflict_resolution)
            # Add incremental reward tracking to history
            self.training_history['goal_approach_rewards'].append(avg_goal_approach_reward)
            self.training_history['collision_avoidance_rewards'].append(avg_collision_avoidance_reward)
            self.training_history['grid_movement_rewards'].append(avg_grid_movement_reward)
            self.training_history['movement_away_rewards'].append(avg_movement_away_reward)
            # Removed spike sparsity loss tracking - was overwhelming the model
            self.training_history['distance_away_penalties'].append(avg_distance_away_penalty)
            # Adjacent coordination no longer tracked in simplified implementation

            # Force update the visualizer's training history reference (in case it got disconnected)
            if self.visualizer is not None:
                self.visualizer.training_history = self.training_history

            # Update new 4-graph visualizer with training metrics
            if self.visualizer is not None and hasattr(self.visualizer, 'update_training_metrics'):
                self.visualizer.update_training_metrics(
                    epoch=epoch + 1,
                    loss=avg_loss,
                    communication_weight=self.communication_weight,
                    coordination_weight=self.coordination_weight, 
                    oscillation_weight=self.oscillation_penalty_weight.item(),
                    agents_reached_rate=0.0  # Will be updated after validation
                )

            print(f"📊 Training history updated: {len(self.training_history['epoch_losses'])} epochs recorded")
            print(f"   Latest loss: {avg_loss:.4f}, comm: {avg_comm_loss:.4f}, coord: {avg_coord_score:.4f}")

            # Debug utilities integration
            if self.debug_enabled:
                # Log training step with comprehensive metrics
                epoch_metrics = {
                    'loss': avg_loss,
                    'comm_loss': avg_comm_loss,
                    'coord_score': avg_coord_score,
                    'entropy_bonus': avg_entropy_bonus,
                    'progress_reward': avg_progress_reward,
                    'collision_loss': avg_collision_loss,
                    'stress_inhibition': avg_stress_inhibition,
                    'stdp_reward': avg_stdp_reward,
                    'stay_action_rate': stay_action_rate
                }

                log_training_step(epoch + 1, 0, avg_loss, epoch_metrics, log_freq=1)

                # Monitor loss components
                current_losses = {
                    'total_loss': avg_loss,
                    'communication_loss': avg_comm_loss,
                    'coordination_loss': avg_coord_score,
                    'collision_loss': avg_collision_loss,
                    'stdp_reward': avg_stdp_reward
                }

                moving_averages = monitor_loss_components(current_losses, epoch)
                if epoch % 5 == 0:  # Print summary every 5 epochs
                    debug_print(f"Loss moving averages: {moving_averages}")

                # Check gradient flow periodically
                if epoch % 10 == 0:
                    debug_print(f"=== Gradient Flow Check (Epoch {epoch + 1}) ===")
                    check_gradient_flow(self.model, print_grads=(epoch % 20 == 0))

                # Log system resources periodically
                if epoch % self.resource_log_freq == 0:
                    log_system_resources()

                # Store metrics for analysis
                for key, value in epoch_metrics.items():
                    if key not in self.metrics_history:
                        self.metrics_history[key] = []
                    self.metrics_history[key].append(value)

            # Validation check - run every epoch (limited to 30 cases for speed)
            print(f"🧪 Running validation for epoch {epoch+1}...")
            val_metrics = self.comprehensive_validation(max_cases=30)  # Use only 30 cases to save time
            if val_metrics:
                # Use the new comprehensive metric names
                self.training_history['val_success_rates'].append(val_metrics['episode_success_rate'])
                self.training_history['val_agents_reached_rates'].append(val_metrics['agents_reached_rate'])
                self.training_history['val_avg_flow_times'].append(val_metrics['avg_flow_time'])
                
                # Update visualizer with validation results
                if self.visualizer is not None and hasattr(self.visualizer, 'update_training_metrics'):
                    self.visualizer.update_training_metrics(
                        epoch=epoch + 1,
                        loss=avg_loss,
                        communication_weight=self.communication_weight,
                        coordination_weight=self.coordination_weight, 
                        oscillation_weight=self.oscillation_penalty_weight.item(),
                        agents_reached_rate=val_metrics['agents_reached_rate']
                    )
                
                print(f"   📈 Validation Results: Success Rate: {val_metrics['episode_success_rate']:.3f}, "
                      f"Agents Reached: {val_metrics['total_agents_reached']}/{val_metrics['total_agents']} ({val_metrics['agents_reached_rate']:.3f}), "
                      f"Avg Flow Time: {val_metrics['avg_flow_time']:.1f}")
            else:
                print(f"   ⚠️  Validation failed to generate metrics")

            # Step the learning rate scheduler based on agents reached rate (not loss)
            old_lr = self.optimizer.param_groups[0]['lr']

            # Use agents reached rate as the metric for scheduler
            if val_metrics and 'agents_reached_rate' in val_metrics:
                agents_reached_rate = val_metrics['agents_reached_rate']
                self.scheduler.step(agents_reached_rate)

                # Update STDP learning rate to be a multiple of main LR
                new_lr = self.optimizer.param_groups[0]['lr']
                stdp_lr = new_lr * self.stdp_lr_multiplier

                # Update STDP system learning rate if it exists
                if hasattr(self, 'stdp_system') and self.stdp_system is not None:
                    try:
                        self.stdp_system.update_learning_rate(stdp_lr)
                    except AttributeError:
                        pass  # STDP system might not have this method

                # Log LR changes
                if new_lr != old_lr:
                    print(f"📉 Learning rate reduced (agents_reached_rate: {agents_reached_rate:.3f}): {old_lr:.2e} → {new_lr:.2e}")
                    print(f"🧠 STDP learning rate updated: {stdp_lr:.2e} ({self.stdp_lr_multiplier}x main LR)")
                elif (epoch + 1) % 10 == 0:  # Periodic LR reporting
                    print(f"📊 Current learning rates - Main: {new_lr:.2e}, STDP: {stdp_lr:.2e}")
            else:
                # Fallback to loss-based scheduling if validation metrics unavailable
                print(f"⚠️  Using loss-based LR scheduling (validation metrics unavailable)")
                self.scheduler.step(avg_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"📉 Learning rate reduced (loss-based): {old_lr:.2e} → {new_lr:.2e}")

            # Update epoch progress bar with essential metrics
            epoch_pbar.set_postfix({
                'Loss': f'{avg_loss:.3f}',
                'Collision': f'{avg_collision_loss:.3f}',
                'DistPenalty': f'{current_distance_penalty_weight:.0f}',
                'Entropy': f'{entropy_bonus_weight:.2f}',
                'Weights': f'S:{self.stress_inhibition_weight.item():.1f} M:{self.selective_movement_weight.item():.1f}'
            })

            # Show essential summary (always visible)
            print(f"\n📊 Epoch {epoch+1} | Loss: {avg_loss:.4f} | Collision: {avg_collision_loss:.4f} | "
                  f"Distance Penalty Weight: {current_distance_penalty_weight:.1f} | Entropy: {entropy_bonus_weight:.3f}")
            print(f"   Weights - Distance Away Penalty: {avg_distance_away_penalty:.4f}, "
                  f"Stress: {self.stress_inhibition_weight.item():.3f}, "
                  f"Selective: {self.selective_movement_weight.item():.3f}")
            print(f"   📈 Multi-sequence training: agents follow predictions, learn from TRUE expert actions")

            # Update visualization with epoch results (main thread call)
            try:
                if hasattr(self, 'visualizer') and self.visualizer.visualize_training:
                    # Call update_visualization directly from main thread to refresh plots
                    self.visualizer.update_visualization(epoch, batch_idx=0, episode_data=None, model=self.model)
                    print(f"📊 Visualization updated for epoch {epoch+1}")
            except Exception as viz_error:
                print(f"⚠️ Visualization update failed: {viz_error}")

            # Spike monitoring summary and health check
            if self.spike_tracker.enabled:
                network_health = self.spike_tracker.summary(epoch + 1)
                print(f"🧠 Network Health Score: {network_health:.3f}/1.0 {'✅' if network_health > 0.5 else '⚠️' if network_health > 0.2 else '❌'}")

                # Save spike statistics every 5 epochs
                if (epoch + 1) % 5 == 0:
                    self.spike_monitor.save_stats(epoch + 1)

                # Plot spike activity every 10 epochs
                if (epoch + 1) % 10 == 0:
                    try:
                        self.spike_monitor.plot_spike_activity(save=True)
                    except Exception as e:
                        print(f"⚠️ Spike activity plotting failed: {e}")

            # Detailed information only in debug mode
            if self.debug_enabled:
                print(f"   Communication: {avg_comm_loss:.4f}")
                print(f"   Coordination: {avg_coord_score:.4f}")
                print(f"   🤝 STDP Coordination Reward: {avg_stdp_reward:.4f}")
                print(f"   🎯 Coordination Efficiency: {avg_coordination_efficiency:.4f}")
                print(f"   🏁 Goal Proximity Bonus: {avg_goal_proximity_bonus:.4f}")
                print(f"   ⚖️  Conflict Resolution: {avg_conflict_resolution:.4f}")
                print(f"   🤝 Adjacent Coordination: Enhanced (2x penalty for close agents)")
                print(f"   🚨 Stress Inhibition: {avg_stress_inhibition:.4f} (zones detected: {stress_detection_rate:.1%})")
                print(f"   🎯 Selective Movement: {avg_selective_movement:.4f} (avg intensity: {avg_stress_intensity:.3f})")
                if self.config.get('use_stay_action_reward', True):
                    print(f"   ⏸️  Stay Action Reward: {avg_stay_reward:.4f} (rate: {stay_action_rate:.1%}, avg prob: {avg_stay_prob:.3f})")
                if self.config.get('use_goal_progress_reward', False):
                    print(f"   Progress Reward: {avg_progress_reward:.4f}")

                # Show incremental reward breakdown
                print(f"   🎯 Incremental Rewards (total: {total_incremental_reward:.4f}):")
                print(f"      └─ Goal Approach: {avg_goal_approach_reward:.4f}")
                print(f"      └─ Collision Avoidance: {avg_collision_avoidance_reward:.4f}")
                print(f"      └─ Grid Movement: {avg_grid_movement_reward:.4f}")
                print(f"      └─ Movement Away: {avg_movement_away_reward:.4f}")
                print(f"   🚫 Distance-Away Penalty: {avg_distance_away_penalty:.4f} (weight: {current_distance_penalty_weight:.1f})")

                # Removed spike sparsity info - was overwhelming the model
                # Now using adaptive thresholds in SNN layers instead

        print("✅ Training completed!")

        # Clear oscillation history to prevent memory buildup
        from loss_compute import clear_oscillation_history
        clear_oscillation_history()

        # Stop animation worker if running
        if hasattr(self, 'visualizer') and hasattr(self.visualizer, 'stop_animation_worker'):
            self.visualizer.stop_animation_worker()

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
    parser = argparse.ArgumentParser(description='Enhanced MAPF Training with Real-time Visualization')
    parser.add_argument('--config', type=str, default='configs/config_snn.yaml',
                        help='Path to config file (default: configs/config_snn.yaml)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--no-visualize', action='store_true', default=False,
                        help='Disable real-time visualization during training')
    parser.add_argument('--eval-only', action='store_true', default=False,
                        help='Run validation only (no training)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the trained model')
    parser.add_argument('--load-path', type=str, default=None,
                        help='Path to load a pre-trained model')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode with verbose output')
    parser.add_argument('--no-pretrain', '--skip-pretraining', action='store_true', default=False,
                        help='Disable pre-training phase (alias: --skip-pretraining)')
    parser.add_argument('--force-visualize', action='store_true', default=False,
                        help='Force enable visualization with debug output')
    args = parser.parse_args()

    visualizer = None
    try:
        print("🚀 Starting Enhanced MAPF Training System...")
        print(f"📋 Using config: {args.config}")

        # Load config from file
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize visualizer if enabled
        visualizer = None
        if config.get('visualize_training', False):
            try:
                visualizer = MAPFVisualizer(config)
                visualizer.start()
                print("✅ Visualizer initialized and started")
            except Exception as e:
                print(f"❌ Failed to initialize visualizer: {e}")
                visualizer = None

        # Create enhanced trainer with config dictionary
        trainer = EnhancedTrainer(config, debug_mode=args.debug, visualizer = visualizer)

        # Disable pre-training if requested
        if args.no_pretrain:
            trainer.enable_pretraining = False
            print("🚫 Pre-training disabled via command line")

        # Override epochs if specified
        if args.epochs is not None:
            trainer.num_epochs = args.epochs
            print(f"📊 Epochs overridden to: {args.epochs}")

            # Setup visualization now that it's enabled
            trainer.setup_visualization_if_enabled()

        # Load pre-trained model if specified
        if args.load_path:
            try:
                trainer.model.load_state_dict(torch.load(args.load_path, map_location=trainer.device))
                print(f"✅ Loaded pre-trained model from: {args.load_path}")
            except Exception as e:
                print(f"❌ Failed to load model from {args.load_path}: {e}")
                return

        if args.eval_only:
            # Run validation only
            print("🧪 Running validation only...")
            metrics = trainer.comprehensive_validation(max_cases=30)  # Use 30 cases for speed

            print("\n📊 VALIDATION RESULTS:")
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

            # Run final validation
            print("\n🧪 Running final validation...")
            final_metrics = trainer.comprehensive_validation(max_cases=30)  # Use 30 cases for speed

            print("\n📊 FINAL VALIDATION RESULTS:")
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

        print("✅ Enhanced MAPF Training completed successfully!")

        # Save final trained model checkpoint
        print("💾 Saving final trained model...")
        final_checkpoint_path = f"{trainer.exp_name}_final.pth"
        final_state = {
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'training_history': trainer.training_history,
            'pretraining_history': getattr(trainer, 'pretraining_history', {}),
            'config': trainer.config,
            'training_completed': True,
            'final_metrics': {
                'final_loss': trainer.training_history['losses'][-1] if trainer.training_history['losses'] else 0.0,
                'best_success_rate': max(trainer.training_history.get('val_success_rates', [0.0])),
                'total_epochs': len(trainer.training_history['losses'])
            }
        }
        torch.save(final_state, final_checkpoint_path)
        print(f"   ✅ Final checkpoint saved: {final_checkpoint_path}")

        # Verify weight preservation from pretraining to final training
        if hasattr(trainer, 'initial_pretrain_state'):
            print("🔍 Final weight change verification...")
            for name, param in trainer.model.named_parameters():
                if name in trainer.initial_pretrain_state:
                    final_magnitude = param.abs().mean().item()
                    total_change = (param - trainer.initial_pretrain_state[name]).abs().mean().item()
                    print(f"   🧠 {name}: final_magnitude={final_magnitude:.6f}, total_change={total_change:.6f}")
                    break  # Just show one example

        # Debug utilities - training summary
        if hasattr(trainer, 'debug_enabled') and trainer.debug_enabled and hasattr(trainer, 'metrics_history'):
            print("\n📊 === Training Summary ===")
            summary = summarize_training_metrics(trainer.metrics_history, recent_steps=10)
            for metric, value in summary.items():
                if 'trend' in metric:
                    trend_str = "📈" if value > 0 else "📉" if value < 0 else "➡️"
                    print(f"  {metric}: {value:.6f} {trend_str}")
                else:
                    print(f"  {metric}: {value:.4f}")

            # Final gradient flow check
            print("\n🔄 === Final Gradient Flow Check ===")
            check_gradient_flow(trainer.model, print_grads=False)

            # Final resource usage
            print("\n💻 === Final System Resources ===")
            log_system_resources()

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
        # Ensure the visualizer process is stopped
        if visualizer is not None:
            visualizer.stop()
        print("✅ Training script finished.")


if __name__ == "__main__":
    main()
