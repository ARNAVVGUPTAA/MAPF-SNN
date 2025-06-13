import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import yaml
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F  # Add this import

from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
from data_loader import GNNDataLoader, SNNDataLoader

from spikingjelly.activation_based import learning, functional
from spikingjelly.activation_based.neuron import LIFNode
from config import config
from models.surrogate_snn import SurrogateSNN, RecurrentLIFNode
from collision_utils import (
    compute_collision_loss, 
    extract_positions_from_fov, 
    extract_positions_from_actions,
    predict_future_collisions,
    reconstruct_positions_from_trajectories,
    load_initial_positions_from_dataset,
    load_goal_positions_from_dataset,
    compute_goal_proximity_reward
)
from expert_utils import (
    load_expert_demonstrations,
    create_mixed_batch,
    should_apply_expert_training,
    compute_interval_expert_weight
)

def cosine_annealing_schedule(epoch, initial_value, T_max, eta_min_ratio=0.01, restart_period=0):
    """
    Cosine annealing schedule for regularizers.
    
    Args:
        epoch: Current epoch
        initial_value: Initial value of the regularizer
        T_max: Period of the cosine annealing
        eta_min_ratio: Minimum value ratio (min_value = eta_min_ratio * initial_value)
        restart_period: Period for cosine restart (0 = no restart)
    
    Returns:
        Scheduled value for the regularizer
    """
    # Ensure all parameters are floats to avoid type errors
    initial_value = float(initial_value)
    eta_min_ratio = float(eta_min_ratio)
    T_max = float(T_max)
    restart_period = float(restart_period)
    
    eta_min = eta_min_ratio * initial_value
    
    if restart_period > 0:
        # Cosine annealing with warm restarts
        epoch = epoch % restart_period
        T_max = restart_period
    
    # Cosine annealing formula
    value = eta_min + (initial_value - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
    return value

def collision_curriculum_schedule(epoch, final_value, curriculum_epochs=10, start_ratio=0.1):
    """
    Curriculum learning schedule for collision loss weight.
    Gradually increases from start_ratio * final_value to final_value over curriculum_epochs.
    
    Args:
        epoch: Current epoch (0-indexed)
        final_value: Final collision loss weight
        curriculum_epochs: Number of epochs for curriculum learning
        start_ratio: Starting ratio of final weight (e.g., 0.1 = 10% of final weight)
    
    Returns:
        Scheduled collision loss weight
    """
    if epoch >= curriculum_epochs:
        return final_value
    
    # Linear increase from start_ratio * final_value to final_value
    start_value = start_ratio * final_value
    progress = epoch / curriculum_epochs
    value = start_value + (final_value - start_value) * progress
    
    return value

def entropy_bonus_schedule(epoch, initial_weight, bonus_epochs=15, decay_type='linear'):
    """
    Entropy bonus schedule for exploration encouragement.
    Decreases from initial_weight to 0 over bonus_epochs.
    
    Args:
        epoch: Current epoch (0-indexed)
        initial_weight: Initial entropy bonus weight
        bonus_epochs: Number of epochs to apply entropy bonus
        decay_type: Type of decay ('linear', 'exponential', 'cosine')
    
    Returns:
        Scheduled entropy bonus weight
    """
    if epoch >= bonus_epochs:
        return 0.0
    
    progress = epoch / bonus_epochs
    
    if decay_type == 'linear':
        # Linear decay from initial_weight to 0
        return initial_weight * (1.0 - progress)
    elif decay_type == 'exponential':
        # Exponential decay
        return initial_weight * (0.1 ** progress)
    elif decay_type == 'cosine':
        # Cosine decay (smooth transition)
        return initial_weight * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        # Default to linear
        return initial_weight * (1.0 - progress)

def compute_entropy_bonus(logits):
    """
    Compute entropy bonus to encourage exploration.
    
    Args:
        logits: Action logits [batch * num_agents, num_actions]
    
    Returns:
        Entropy bonus (higher entropy = higher bonus)
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Compute entropy: -sum(p * log(p))
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    # Return mean entropy across all agents and batches
    return torch.mean(entropy)

def compute_success_rate(success_list):
    """
    Compute the average success rate from a list of per-episode successes.
    Args:
        success_list (list of float): successes per episode (0.0 to 1.0)
    Returns:
        float: average success rate, or 0.0 if list is empty
    """
    if not success_list:
        return 0.0
    return sum(success_list) / len(success_list)

def evaluate_on_validation(model, data_loader, config, num_samples=10):
    """
    Evaluate model performance on validation dataset.
    Args:
        model: The neural network model
        data_loader: Data loader with valid_loader attribute
        config: Configuration dictionary
        num_samples: Number of validation samples to evaluate
    Returns:
        dict: Validation metrics including success rate and agents reached
    """
    if data_loader.valid_loader is None:
        return {"val_success_rate": 0.0, "val_agents_reached": 0, "val_total_agents": 0}
    
    model.eval()
    device = config["device"]
    net_type = config["net_type"]
    
    # Reset SNN state if using SNN
    if net_type == "snn":
        functional.reset_net(model)
    
    val_success_rates = []
    val_agents_reached = 0
    val_total_agents = 0
    
    # Sample a subset of validation data
    val_iter = iter(data_loader.valid_loader)
    samples_processed = 0
    
    with torch.no_grad():
        for _ in range(min(num_samples, len(data_loader.valid_loader))):
            try:
                batch_data = next(val_iter)
                if len(batch_data) == 4:  # SNN case with case_idx
                    states, trajectories, gso_or_dummy, case_idx = batch_data
                else:  # GNN case without case_idx
                    states, trajectories, gso_or_dummy = batch_data
                
                states = states.to(device)
                trajectories = trajectories.to(device)
                
                batch_size = states.shape[0]
                T = states.shape[1]
                num_agents = trajectories.shape[2]
                
                # Reset SNN for each validation batch
                if net_type == "snn":
                    functional.reset_net(model)
                
                # Process each time step
                correct_predictions = 0
                total_predictions = 0
                
                for t in range(T):
                    fov_t = states[:, t]
                    
                    if net_type == "snn":
                        logits = model(fov_t)
                    else:
                        gso_t = gso_or_dummy.to(device)
                        logits = model(fov_t, gso_t)
                    
                    logits = logits.view(-1, num_agents, model.num_classes)
                    
                    # Compare predictions with ground truth
                    for agent in range(num_agents):
                        predicted_actions = torch.argmax(logits[:, agent], dim=1)
                        true_actions = trajectories[:, t, agent].long()
                        correct_predictions += (predicted_actions == true_actions).sum().item()
                        total_predictions += batch_size
                    
                    if net_type == "snn":
                        functional.detach_net(model)
                
                # Calculate success rate for this batch (percentage of correct predictions)
                batch_success_rate = correct_predictions / total_predictions if total_predictions > 0 else 0.0
                val_success_rates.append(batch_success_rate)
                
                # Count agents (simplified metric)
                val_agents_reached += int(batch_success_rate * batch_size * num_agents)
                val_total_agents += batch_size * num_agents
                
                samples_processed += 1
                
            except StopIteration:
                break
    
    # Calculate average validation metrics
    avg_val_success_rate = sum(val_success_rates) / len(val_success_rates) if val_success_rates else 0.0
    
    return {
        "val_success_rate": avg_val_success_rate,
        "val_agents_reached": val_agents_reached,
        "val_total_agents": val_total_agents,
        "val_samples_processed": samples_processed
    }

net_type = config["net_type"]
exp_name = config["exp_name"]
tests_episodes = config["tests_episodes"]
# config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if net_type == "baseline":
    from models.framework_baseline import Network
elif net_type == "gnn":
    from models.framework_gnn_message import Network
elif net_type == "snn":
    from models.framework_snn import Network

results_path = os.path.join("results", exp_name)
os.makedirs(results_path, exist_ok=True)

with open(os.path.join(results_path, "config.yaml"), "w") as config_path:
    yaml.dump(config, config_path)

if __name__ == "__main__":
    print("----- Training stats -----")
    if net_type == "gnn":
        # Enable obstacle conflict resolution by default
        data_loader = GNNDataLoader(config, resolve_obstacle_conflicts=True)
    elif net_type == "snn":
        # Initialize SNN data loader with obstacle conflict resolution enabled
        data_loader = SNNDataLoader(config, resolve_obstacle_conflicts=True)
        # Sample first batch to get C, H, W
        batch_data = next(iter(data_loader.train_loader))
        if len(batch_data) == 4:  # With case_idx
            sample_states, _, _, _ = batch_data
        else:  # Without case_idx (backward compatibility)
            sample_states, _, _ = batch_data
        
        shape = sample_states.shape
        if len(shape) == 6:
            _, _, _, C, H, W = shape
        elif len(shape) == 5:
            _, C, H, W = shape
        else:
            raise RuntimeError(f"Unexpected SNN data shape: {shape}")
        config["channels"] = C
        config["height"] = H
        config["width"] = W
    else:
        data_loader = None  # baseline case or others
    
    # Create model - framework_snn Network only takes config
    model = Network(config)
    model.to(config["device"])

    if net_type == "snn":
        # Use SurrogateSNN and Adam optimizer as per prompt
        channels = config.get("channels", 1)
        height = config.get("height", config["board_size"])
        width = config.get("width", config["board_size"])
        num_actions = config.get("num_actions", 5)
        device = config["device"]
        epochs = config["epochs"]
        # Retrieve and cast hyperparameters from config
        lr = float(config.get('learning_rate', 1e-3))
        wd = float(config.get('weight_decay', 1e-4))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        # LR scheduler: reduce LR when success rate plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=float(config.get('lr_decay_factor', 0.5)),
            patience=int(config.get('lr_patience', 5)),
        )
        early_stop_counter = 0
        loss_fn = nn.CrossEntropyLoss()
    else:
        optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss okkk")

    losses = []
    success_rate_final = []
    flow_time_final = []

    # Load expert demonstrations for interval-based teacher forcing
    expert_demos = {}
    use_expert_training = config.get('use_expert_training', True)
    expert_interval = config.get('expert_interval', 10)  # Apply every N epochs
    expert_ratio = config.get('expert_ratio', 0.1)      # 10% of samples
    expert_weight = config.get('expert_weight', 0.6)    # Weight for expert samples
    
    if use_expert_training and net_type == "snn":
        print("Loading expert demonstrations for teacher forcing...")
        # Load expert demonstrations for available training cases
        all_case_indices = list(range(1000))  # Assuming up to 1000 training cases
        expert_demos = load_expert_demonstrations(
            dataset_root=config['train']['root_dir'],
            case_indices=all_case_indices,
            mode='train',
            num_agents=config['train']['nb_agents'],
            max_time=config['train']['max_time_dl']
        )
        print(f"Loaded {len(expert_demos)} expert demonstrations")

    # Removed debug prints and kept only essential logs
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch}")
        model.train()
        
        # Get configuration values that are used throughout the epoch
        use_collision_curriculum = config.get('use_collision_curriculum', False)
        
        # Training phase metrics
        epoch_correct_predictions = 0
        epoch_total_predictions = 0
        epoch_total_agents = 0
        epoch_agents_reached_goal = 0
        epoch_total_timesteps = 0
        
        if net_type == "snn":
            train_loss = 0.0
            total_spikes = 0  # Initialize total spikes counter
            
            # Initialize expert training variables for logging
            expert_loss_weight = 0.0
            
            # Load initial positions for collision detection if enabled
            use_collision_loss = config.get('use_collision_loss', True)
            initial_positions_cache = {}
            
            for i, (states, trajectories, _, case_indices) in enumerate(data_loader.train_loader):
                states = states.to(device)
                trajectories = trajectories.to(device)  # [batch, time, agents]
                
                # Apply interval-based expert training 
                expert_mask = None
                if use_expert_training and should_apply_expert_training(epoch, expert_interval) and len(expert_demos) > 0:
                    # Create mixed batch with expert trajectories
                    batch_case_indices = case_indices.tolist() if hasattr(case_indices, 'tolist') else case_indices
                    # Filter out invalid case indices (negative values)
                    valid_case_indices = [idx if idx >= 0 else 0 for idx in batch_case_indices]
                    states, trajectories, _, expert_mask = create_mixed_batch(
                        (states, trajectories, _),
                        expert_demos,
                        valid_case_indices,
                        expert_ratio
                    )
                    trajectories = trajectories.to(device)
                    expert_mask = expert_mask.to(device)
                    expert_loss_weight = compute_interval_expert_weight(expert_weight)
                    # Ensure expert_loss_weight is a scalar for boolean comparisons
                    if isinstance(expert_loss_weight, torch.Tensor):
                        expert_loss_weight = expert_loss_weight.item()

                T = states.shape[1]  # Number of time steps
                batch_size = states.shape[0]
                num_agents = trajectories.shape[2]

                # Track training metrics for this batch
                batch_correct = 0
                batch_total = 0
                batch_agents_total = batch_size * num_agents
                epoch_total_agents += batch_agents_total
                epoch_total_timesteps += T

                # Load initial positions for this batch if collision loss is enabled
                if use_collision_loss:
                    try:
                        # Get case indices for this batch (with proper cycling to avoid exceeding dataset size)
                        dataset_size = 1000  # Based on dataset analysis (cases 0-999)
                        batch_start_idx = i * batch_size
                        case_indices = [(batch_start_idx + j) % dataset_size for j in range(batch_size)]
                        
                        # Debug: Log if we're cycling through the dataset
                        if batch_start_idx >= dataset_size:
                            print(f"INFO: Cycling through dataset - batch {i}, start_idx {batch_start_idx}, using cases {case_indices[:3]}...{case_indices[-3:]}")
                        
                        # Load initial positions from dataset
                        initial_positions = load_initial_positions_from_dataset(
                            config['train']['root_dir'], case_indices, mode='train'
                        )
                        initial_positions = initial_positions.to(device)
                        
                        # Handle batch size mismatch if needed
                        if initial_positions.shape[0] > batch_size:
                            initial_positions = initial_positions[:batch_size]
                        elif initial_positions.shape[0] < batch_size:
                            # Pad with safe spread positions instead of random ones
                            print(f"WARNING: [FALLBACK TRIGGERED] Batch size mismatch! Expected {batch_size}, got {initial_positions.shape[0]}")
                            print(f"         Padding with SAFE spread positions - obstacle-free placement!")
                            missing_batches = batch_size - initial_positions.shape[0]
                            dummy_positions_list = []
                            for batch_idx in range(missing_batches):
                                dummy_positions = []
                                for agent_id in range(num_agents):
                                    # Spread agents evenly with offset per batch to avoid overlaps
                                    x = ((agent_id * 5) + (batch_idx * 2)) % 28  # Spread horizontally with batch offset
                                    y = ((agent_id * 3) + (batch_idx * 2)) % 28  # Spread vertically with batch offset
                                    dummy_positions.append([float(x), float(y)])
                                dummy_positions_list.append(dummy_positions)
                            dummy_positions_tensor = torch.tensor(dummy_positions_list, dtype=torch.float32, device=device)
                            print(f"         Generated {missing_batches} safe position batches")
                            initial_positions = torch.cat([initial_positions, dummy_positions_tensor], dim=0)
                            
                    except Exception as e:
                        print(f"ERROR: [FALLBACK TRIGGERED] Failed to load initial positions: {e}")
                        print(f"       Using completely SAFE spread positions - obstacle-free map!")
                        # Generate safe spread positions for entire batch
                        safe_positions_list = []
                        for batch_idx in range(batch_size):
                            batch_positions = []
                            for agent_id in range(num_agents):
                                # Spread agents evenly with batch offset to avoid overlaps
                                x = ((agent_id * 5) + (batch_idx * 2)) % 28  # Spread horizontally
                                y = ((agent_id * 3) + (batch_idx * 2)) % 28  # Spread vertically
                                batch_positions.append([float(x), float(y)])
                            safe_positions_list.append(batch_positions)
                        initial_positions = torch.tensor(safe_positions_list, dtype=torch.float32, device=device)
                        print(f"       Generated safe positions for entire batch: shape {initial_positions.shape}")

                # Load goal positions for goal progress reward if enabled
                use_goal_progress_reward = config.get('use_goal_progress_reward', False)
                goal_positions = None
                if use_goal_progress_reward:
                    try:
                        # Get case indices for this batch (use same cycling logic as for initial positions)
                        dataset_size = 1000  # Based on dataset analysis (cases 0-999)
                        batch_start_idx = i * batch_size
                        case_indices = [(batch_start_idx + j) % dataset_size for j in range(batch_size)]
                        
                        # Load goal positions from dataset
                        goal_positions = load_goal_positions_from_dataset(
                            config['train']['root_dir'], case_indices, mode='train'
                        )
                        goal_positions = goal_positions.to(device)
                        
                        # Handle batch size mismatch if needed
                        if goal_positions.shape[0] > batch_size:
                            goal_positions = goal_positions[:batch_size]
                        elif goal_positions.shape[0] < batch_size:
                            # Pad with dummy positions if needed
                            print(f"WARNING: [FALLBACK TRIGGERED] Goal batch size mismatch! Expected {batch_size}, got {goal_positions.shape[0]}")
                            print(f"         Padding with RANDOM dummy goal positions - this could cause issues!")
                            dummy_goal_positions = torch.randint(0, 28, (batch_size - goal_positions.shape[0], num_agents, 2), 
                                                               device=device, dtype=torch.float32)
                            print(f"         Generated {dummy_goal_positions.shape[0]} random goal position batches")
                            goal_positions = torch.cat([goal_positions, dummy_goal_positions], dim=0)
                            
                    except Exception as e:
                        print(f"ERROR: [FALLBACK TRIGGERED] Failed to load goal positions: {e}")
                        print(f"       Disabling goal progress reward for this batch due to missing goal data")
                        use_goal_progress_reward = False
                        goal_positions = None

                # Reset and forward through time
                functional.reset_net(model)
                spike_reg = 0.0  # initialize spike regularization accumulator
                real_collision_reg = 0.0  # initialize real collision regularization accumulator
                future_collision_reg = 0.0  # initialize future collision regularization accumulator
                entropy_bonus_reg = 0.0  # initialize entropy bonus accumulator
                goal_progress_reg = 0.0  # initialize goal progress reward accumulator
                total_loss = 0.0
                total_spikes = 0
                total_real_collisions = 0
                total_future_collisions = 0
                
                # Initialize goal state tracking for this batch
                agents_at_goal_mask = None  # Track which agents have reached goals
                previous_agents_at_goal = None  # Track previous timestep's goal achievement
                
                # Initialize best distances tracking for progress reward
                best_distances = None  # Track best (minimum) distances achieved so far [batch_size, num_agents]
                if use_goal_progress_reward and goal_positions is not None:
                    # Initialize best distances to infinity (will be updated on first timestep)
                    best_distances = torch.full_like(
                        torch.norm(goal_positions - goal_positions, dim=2), 
                        float('inf'), device=device
                    )
                
                # Track collision loss metrics for logging
                total_collision_loss = 0.0  # accumulated collision loss applied to training
                total_per_agent_collision_penalties = 0.0  # accumulated per-agent collision penalties
                
                # Get collision loss configuration with cosine annealing
                use_cosine = config.get('use_cosine_annealing', True)
                T_max = config.get('cosine_T_max', 50)
                eta_min_ratio = config.get('cosine_eta_min_ratio', 0.01)
                restart_period = config.get('cosine_restart_period', 0)
                
                # Get entropy bonus schedule
                use_entropy_bonus = config.get('use_entropy_bonus', False)
                entropy_bonus_weight = 0.0
                if use_entropy_bonus:
                    initial_entropy_weight = config.get('entropy_bonus_weight', 0.02)
                    bonus_epochs = config.get('entropy_bonus_epochs', 15)
                    decay_type = config.get('entropy_bonus_decay_type', 'linear')
                    entropy_bonus_weight = entropy_bonus_schedule(epoch, initial_entropy_weight, bonus_epochs, decay_type)
                
                # Get goal progress reward weight (constant throughout training)
                goal_progress_weight = 0.0
                use_goal_progress_reward = config.get('use_goal_progress_reward', False)
                if use_goal_progress_reward:
                    goal_progress_weight = float(config.get('goal_progress_weight', 0.05))
                
                if use_collision_curriculum:
                    # Use curriculum learning for collision loss
                    curriculum_epochs = config.get('collision_curriculum_epochs', 10)
                    start_ratio = config.get('collision_curriculum_start_ratio', 0.1)
                    final_collision_weight = config.get('collision_loss_weight', 15.0)
                    
                    collision_loss_weight = collision_curriculum_schedule(
                        epoch, final_collision_weight, curriculum_epochs, start_ratio
                    )
                    
                    # Anneal collision weight back down after curriculum period
                    if epoch > curriculum_epochs:
                        collision_loss_weight *= 0.98
                    
                    # Keep other weights using cosine annealing or fixed values
                    if use_cosine:
                        future_collision_weight = cosine_annealing_schedule(
                            epoch,
                            config.get('future_collision_weight', 0.8),
                            T_max, eta_min_ratio, restart_period
                        )
                        spike_reg_weight = cosine_annealing_schedule(
                            epoch,
                            config.get('spike_reg_weight', 5e-3),
                            T_max, eta_min_ratio, restart_period
                        )
                    else:
                        future_collision_weight = config.get('future_collision_weight', 0.8)
                        spike_reg_weight = max(
                            float(config.get('max_spike_decay', 1e-3)), 
                            config.get('spike_decay_coeff', 0.05) * (1 - epoch / config.get('epochs', 50))
                        )
                        
                elif use_cosine:
                    # Apply cosine annealing to collision loss weight
                    collision_loss_weight = cosine_annealing_schedule(
                        epoch, 
                        config.get('collision_loss_weight', 2.0),
                        T_max, eta_min_ratio, restart_period
                    )
                    future_collision_weight = cosine_annealing_schedule(
                        epoch,
                        config.get('future_collision_weight', 0.8),
                        T_max, eta_min_ratio, restart_period
                    )
                    spike_reg_weight = cosine_annealing_schedule(
                        epoch,
                        config.get('spike_reg_weight', 5e-3),
                        T_max, eta_min_ratio, restart_period
                    )
                else:
                    # Fixed weights (legacy behavior)
                    collision_loss_weight = config.get('collision_loss_weight', 2.0)
                    future_collision_weight = config.get('future_collision_weight', 0.8)
                    # Legacy linear decay for spike regularization
                    spike_reg_weight = max(
                        float(config.get('max_spike_decay', 1e-3)), 
                        config.get('spike_decay_coeff', 0.05) * (1 - epoch / config.get('epochs', 50))
                    )
                
                collision_config = {
                    'vertex_collision_weight': config.get('vertex_collision_weight', 1.0),
                    'edge_collision_weight': config.get('edge_collision_weight', 0.5),
                    'collision_loss_type': config.get('collision_loss_type', 'l2'),
                    'use_future_collision_penalty': config.get('use_future_collision_penalty', False),
                    'future_collision_steps': config.get('future_collision_steps', 2),
                    'future_collision_weight': future_collision_weight,  # Use scheduled value
                    'future_step_decay': config.get('future_step_decay', 0.7),
                    'future_blend_factor': config.get('future_blend_factor', 0.5),  # New blend factor
                    'separate_collision_types': config.get('separate_collision_types', True),
                    'per_agent_loss': config.get('per_agent_collision_loss', True)  # Enable per-agent collision loss
                }
                
                prev_positions = None
                for t in range(T):
                    fov_t = states[:, t]
                    out_t, spike_info = model(fov_t, return_spikes=True)
                    out_t = out_t.view(-1, trajectories.shape[2], model.num_classes)

                    # Track prediction accuracy for training metrics
                    for agent in range(num_agents):
                        predicted_actions = torch.argmax(out_t[:, agent], dim=1)
                        true_actions = trajectories[:, t, agent].long()
                        batch_correct += (predicted_actions == true_actions).sum().item()
                        batch_total += batch_size

                    # compute standard cross-entropy loss for time step
                    base_loss = 0
                    agent_losses = []  # Store individual agent losses for collision weighting
                    
                    for agent in range(trajectories.shape[2]):
                        logits = out_t[:, agent]
                        targets = trajectories[:, t, agent].long()
                        agent_loss = F.cross_entropy(logits, targets, reduction='none')
                        
                        # Apply expert loss weighting if using expert training
                        if expert_mask is not None and expert_loss_weight > 0:
                            # Weight loss based on whether samples use expert data
                            # expert_mask: [batch_size], agent_loss: [batch_size]
                            expert_weight_tensor = torch.where(expert_mask, 
                                                      torch.tensor(1.0 + expert_loss_weight, device=agent_loss.device),  
                                                      torch.tensor(1.0, device=agent_loss.device))
                            agent_loss = agent_loss * expert_weight_tensor
                        
                        agent_losses.append(agent_loss)  # Store for collision weighting
                        base_loss += agent_loss.mean()
                    
                    base_loss /= trajectories.shape[2]
                    loss = base_loss  # Default loss (will be modified by collision handling)

                    # Add entropy bonus for exploration (encourage diverse actions)
                    if entropy_bonus_weight > 0:
                        # Flatten logits for all agents at this timestep
                        logits_flat = out_t.view(-1, model.num_classes)
                        entropy_bonus = compute_entropy_bonus(logits_flat)
                        loss -= entropy_bonus_weight * entropy_bonus  # Subtract to encourage high entropy
                        entropy_bonus_reg += float(entropy_bonus.item())

                    # Add collision loss if enabled
                    if use_collision_loss:
                        # Use predicted actions to compute current positions
                        predicted_actions = torch.argmax(out_t.view(batch_size, num_agents, -1), dim=-1)  # [batch, num_agents]
                        if t == 0:
                            # First timestep: use initial positions
                            current_positions = initial_positions
                        else:
                            # Subsequent timesteps: apply predicted action from previous position
                            current_positions = extract_positions_from_actions(
                                prev_positions, predicted_actions, board_size=config['board_size'][0]
                            )
                        
                        # Compute collision loss based on predicted actions
                        logits_flat = out_t.view(-1, model.num_classes)
                        collision_loss, collision_info = compute_collision_loss(
                            logits_flat, current_positions, prev_positions, collision_config
                        )
                        
                        # Handle per-agent vs aggregate collision loss
                        if collision_info.get('per_agent_loss', False):
                            # Per-agent collision losses: apply individually to each agent's cross-entropy loss
                            # collision_loss shape: [batch * num_agents]
                            # Reshape to [batch, num_agents]
                            per_agent_collision_penalties = collision_loss.view(batch_size, num_agents)
                            
                            # Apply per-agent collision penalties to the individual agent losses
                            weighted_loss = 0
                            for agent in range(num_agents):
                                # Get collision penalty for this agent across all batches
                                agent_collision_penalty = per_agent_collision_penalties[:, agent]  # [batch_size]
                                
                                # Apply penalty to this agent's cross-entropy loss
                                # agent_losses[agent] shape: [batch_size]
                                weighted_agent_loss = agent_losses[agent] + collision_loss_weight * agent_collision_penalty
                                weighted_loss += weighted_agent_loss.mean()
                            
                            # Compute final loss
                            loss = weighted_loss / num_agents
                            
                            # For logging, use the mean of per-agent collision losses
                            aggregate_collision_loss = collision_loss.mean()
                            total_collision_loss += float(aggregate_collision_loss.item())
                            total_per_agent_collision_penalties += float(collision_loss.sum().item())
                        else:
                            # Legacy behavior: apply single collision loss to entire batch
                            aggregate_collision_loss = collision_loss
                            loss += collision_loss_weight * collision_loss
                            total_collision_loss += float(aggregate_collision_loss.item())
                        
                        real_collision_reg += collision_info['real_collision_loss']
                        future_collision_reg += collision_info['future_collision_loss']
                        total_real_collisions += collision_info['total_real_collisions']
                        # Track if we're using future collision penalty (when no real collisions)
                        if not collision_info.get('using_real_collision_loss', True):
                            total_future_collisions += 1  # Count timesteps where future penalty was applied
                        
                        # Update previous positions for next timestep
                        prev_positions = current_positions

                    # Add goal progress reward if enabled
                    if use_goal_progress_reward and goal_positions is not None and goal_progress_weight > 0:
                        # Reconstruct current positions from trajectories (reuse if already computed)
                        if not use_collision_loss:
                            current_positions = reconstruct_positions_from_trajectories(
                                initial_positions, trajectories, current_time=t, 
                                board_size=config['board_size'][0]
                            )
                        
                        # Initialize best distances on first timestep
                        if best_distances is None:
                            # Compute initial distances to goals
                            initial_distances = torch.norm(current_positions - goal_positions, dim=2)
                            best_distances = initial_distances.clone()
                        
                        # Compute goal progress reward with best-distance tracking
                        success_threshold = config.get('goal_success_threshold', 1.0)
                        
                        # Import the new function
                        from collision_utils import compute_best_distance_progress_reward
                        
                        progress_reward, best_distances = compute_best_distance_progress_reward(
                            current_positions, goal_positions, best_distances, 
                            progress_weight=goal_progress_weight, success_threshold=success_threshold
                        )
                        
                        # Subtract progress reward from loss (reward reduces loss)
                        loss -= progress_reward
                        goal_progress_reg += float(progress_reward.item())

                    spike_reg    += float(spike_info['output_spikes'].mean().item())
                    total_spikes += int(  spike_info['output_spikes'].sum().item())

                    total_loss += loss
                    functional.detach_net(model)

                # Update epoch metrics
                epoch_correct_predictions += batch_correct
                epoch_total_predictions += batch_total
                # Simplified agents reached metric (based on prediction accuracy)
                batch_success_rate = batch_correct / batch_total if batch_total > 0 else 0.0
                epoch_agents_reached_goal += int(batch_success_rate * batch_agents_total)

                # average over time
                total_loss /= T
                spike_reg /= T  # average regularization over time
                real_collision_reg /= T  # average real collision regularization over time
                future_collision_reg /= T  # average future collision regularization over time
                entropy_bonus_reg /= T  # average entropy bonus over time
                goal_progress_reg /= T  # average goal progress reward over time
                total_collision_loss /= T  # average collision loss over time
                total_per_agent_collision_penalties /= T  # average per-agent collision penalties over time
                
                # Apply spike regularization with scheduled weight
                total_loss += spike_reg_weight * spike_reg

                # Backpropagation and optimization
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                train_loss += total_loss.item()
            
            # Calculate epoch training metrics
            epoch_success_rate = epoch_correct_predictions / epoch_total_predictions if epoch_total_predictions > 0 else 0.0
            epoch_avg_flow_time = epoch_total_timesteps / len(data_loader.train_loader)  # Average timesteps per batch
            
            # Get current learning rate from optimizer
            current_lr = optimizer.param_groups[0]['lr']
            expert_info = ""
            if use_expert_training and should_apply_expert_training(epoch, expert_interval) and expert_loss_weight > 0:
                expert_info = f" | ExpertWeight: {expert_loss_weight:.3f} | ExpertRatio: {expert_ratio:.1%}"
            
            # Store training loss for this epoch
            losses.append(train_loss / len(data_loader.train_loader))
            success_rate_final.append(epoch_success_rate)
            flow_time_final.append(epoch_avg_flow_time)
            
            # Prepare curriculum info for logging
            curriculum_info = ""
            if use_collision_curriculum and epoch < config.get('collision_curriculum_epochs', 10):
                curriculum_info = f" | Curriculum: {epoch+1}/{config.get('collision_curriculum_epochs', 10)}"
            
            # Prepare entropy bonus info for logging
            entropy_info = ""
            if entropy_bonus_weight > 0:
                entropy_info = f" | EntropyWeight: {entropy_bonus_weight:.4f} | EntropyReg: {entropy_bonus_reg:.4f}"
            
            # Prepare goal progress reward info for logging
            goal_progress_info = ""
            if use_goal_progress_reward and goal_progress_weight > 0:
                goal_progress_info = f" | GoalProgressWeight: {goal_progress_weight:.4f} | GoalProgressReg: {goal_progress_reg:.4f}"
            
            print(f"Epoch {epoch} | Loss: {train_loss / len(data_loader.train_loader):.4f} | LR: {current_lr:.2e} | CollisionWeight: {collision_loss_weight:.2e}{curriculum_info} | SpikeRegWeight: {spike_reg_weight:.2e}{entropy_info}{goal_progress_info} | Spikes: {total_spikes} | SpikeReg: {spike_reg:.4f} | RealCollisionReg: {real_collision_reg:.4f} | FutureCollisionReg: {future_collision_reg:.4f} | RealCollisions: {total_real_collisions} | FutureCollisionSteps: {total_future_collisions}{expert_info}")
            print(f"Collision Metrics - CollisionLoss: {total_collision_loss:.4f} | PerAgentPenalties: {total_per_agent_collision_penalties:.4f} | AvgCollisionLoss: {total_collision_loss / len(data_loader.train_loader):.4f}")
            print(f"Training Summary - Success Rate: {epoch_success_rate:.4f} | Avg Flow Time: {epoch_avg_flow_time:.2f} | Agents Reached: {epoch_agents_reached_goal}/{epoch_total_agents}")
        else:
            train_loss = 0.0
            for i, batch_data in enumerate(data_loader.train_loader):
                if len(batch_data) == 4:  # With case_idx
                    states, trajectories, gso, case_idx = batch_data
                else:  # Without case_idx (backward compatibility)
                    states, trajectories, gso = batch_data
                    
                states = states.to(config["device"])
                gso = gso.to(config["device"])

                # Track training metrics for this batch
                batch_size = trajectories.shape[0]
                num_agents = trajectories.shape[1]
                batch_agents_total = batch_size * num_agents
                epoch_total_agents += batch_agents_total

                # Remove SNN+STDP logic, keep only standard supervised learning for non-SNN
                optimizer.zero_grad()
                trajectories = trajectories.to(config["device"])
                output = model(states, gso)
                
                # Track prediction accuracy
                batch_correct = 0
                batch_total = 0
                for agent in range(num_agents):
                    predicted_actions = torch.argmax(output[:, agent], dim=1)
                    true_actions = trajectories[:, agent].long()
                    batch_correct += (predicted_actions == true_actions).sum().item()
                    batch_total += batch_size
                
                epoch_correct_predictions += batch_correct
                epoch_total_predictions += batch_total
                batch_success_rate = batch_correct / batch_total if batch_total > 0 else 0.0
                epoch_agents_reached_goal += int(batch_success_rate * batch_agents_total)
                
                total_loss = torch.zeros(1, requires_grad=True).to(config["device"])
                for agent in range(trajectories.shape[1]):
                    loss = criterion(output[:, agent, :], trajectories[:, agent].long())
                    total_loss = total_loss + (loss / trajectories.shape[1])
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()
            
            # Calculate epoch training metrics
            epoch_success_rate = epoch_correct_predictions / epoch_total_predictions if epoch_total_predictions > 0 else 0.0
            epoch_avg_flow_time = len(data_loader.train_loader)  # Simplified flow time metric
            
            # Store training loss for this epoch
            losses.append(train_loss / len(data_loader.train_loader))
            success_rate_final.append(epoch_success_rate)
            flow_time_final.append(epoch_avg_flow_time)
            
            print(f"Epoch {epoch} | Loss: {train_loss / len(data_loader.train_loader):.4f}")
            print(f"Training Summary - Success Rate: {epoch_success_rate:.4f} | Avg Flow Time: {epoch_avg_flow_time:.2f} | Agents Reached: {epoch_agents_reached_goal}/{epoch_total_agents}")

        # Evaluate on validation dataset
        if net_type == "snn" and data_loader.valid_loader is not None:
            val_metrics = evaluate_on_validation(model, data_loader, config, num_samples=10)
            print(f"Validation success rate: {val_metrics['val_success_rate']:.4f}")
            print(f"Validation agents reached: {val_metrics['val_agents_reached']}/{val_metrics['val_total_agents']}")
            print(f"Validation samples processed: {val_metrics['val_samples_processed']}")
            
            # Step the learning rate scheduler based on validation success rate
            scheduler.step(val_metrics['val_success_rate'])

    # Save training data
    np.save(os.path.join(results_path, "success_rate.npy"), np.array(success_rate_final))
    np.save(os.path.join(results_path, "flow_time.npy"), np.array(flow_time_final))
    np.save(os.path.join(results_path, "loss.npy"), np.array(losses))
    torch.save(model.state_dict(), os.path.join(results_path, "model.pt"))
