"""
Enhanced Loss Computation Module
Centralizes all loss computations for cleaner code architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collision_utils import compute_collision_loss, compute_obstacle_aware_progress_reward
from collections import defaultdict, deque
from debug_utils import debug_print

# Optional import for STDP coordination
try:
    from stdp_coordination import STDPCoordinationSystem
except ImportError:
    STDPCoordinationSystem = None

# Global position history for oscillation detection
_position_history = defaultdict(lambda: deque(maxlen=4))

# Global tracking for consecutive stay actions
_stay_history = defaultdict(lambda: deque(maxlen=10))  # Track last 10 actions per agent

def compute_entropy_bonus(logits):
    """Compute entropy bonus to encourage exploration"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return torch.mean(entropy) / 5.0  # Divide by 5 to reduce entropy reward magnitude


def entropy_bonus_schedule(epoch, initial_weight, bonus_epochs, decay_type='linear', config=None):
    """Compute entropy bonus weight schedule"""
    if epoch >= bonus_epochs:
        return 0.0
    
    progress = epoch / bonus_epochs
    
    if decay_type == 'linear':
        weight = initial_weight * (1.0 - progress)
    elif decay_type == 'exponential':
        exponential_base = config.get('entropy_decay_exponential_base', 0.1) if config else 0.1
        weight = initial_weight * (exponential_base ** progress)
    elif decay_type == 'fast_exponential':
        # Faster decay - reaches near zero by 25% progress
        fast_exponential_base = config.get('entropy_decay_fast_exponential_base', 0.01) if config else 0.01
        weight = initial_weight * (fast_exponential_base ** progress)
    elif decay_type == 'cosine':
        cosine_multiplier = config.get('entropy_decay_cosine_multiplier', 0.5) if config else 0.5
        weight = initial_weight * cosine_multiplier * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    else:
        weight = initial_weight
    
    return float(weight)


def compute_stay_action_reward(model_outputs, stay_reward_weight=1.0):
    """
    Compute reward for choosing the "stay" action (action 0).
    Encourages agents to be more conservative and reduce unnecessary movement.
    
    Args:
        model_outputs: [batch_size, num_agents, action_dim] model predictions
        stay_reward_weight: float, weight for stay action reward
        
    Returns:
        stay_reward: tensor, reward for choosing stay action
        stay_stats: dict, statistics about stay action usage
    """
    batch_size, num_agents, action_dim = model_outputs.shape
    device = model_outputs.device
    
    # Get action probabilities
    action_probs = F.softmax(model_outputs, dim=-1)
    
    # Action 0 is "stay" - reward this choice
    stay_probs = action_probs[:, :, 0]  # [batch_size, num_agents]
    
    # Compute stay reward: higher probability of staying = higher reward
    # Use log probability to encourage confident stay decisions
    stay_log_probs = torch.log(stay_probs + 1e-8)  # Add epsilon for numerical stability
    
    # Mean stay reward across all agents and batches
    stay_reward = torch.mean(stay_log_probs) * stay_reward_weight
    
    # Collect statistics
    predicted_actions = torch.argmax(model_outputs, dim=-1)  # [batch_size, num_agents]
    total_stay_actions = torch.sum(predicted_actions == 0).item()
    total_actions = predicted_actions.numel()
    stay_rate = total_stay_actions / total_actions if total_actions > 0 else 0.0
    avg_stay_prob = torch.mean(stay_probs).item()
    
    stay_stats = {
        'total_stay_actions': total_stay_actions,
        'total_actions': total_actions,
        'stay_action_rate': stay_rate,
        'avg_stay_probability': avg_stay_prob
    }
    
    return stay_reward, stay_stats


def compute_communication_loss(model_outputs, agent_positions, communication_weight, device, config=None):
    """Compute loss that encourages meaningful communication between agents"""
    
    # Extract hidden states/features from model if available
    if hasattr(model_outputs, 'get_hidden_states'):
        hidden_states = model_outputs.get_hidden_states()  # [batch, num_agents, hidden_dim]
    else:
        # Fallback: use output logits as proxy for communication
        hidden_states = model_outputs.detach()  # [batch, num_agents, action_dim]
    
    batch_size, num_agents = hidden_states.shape[:2]
    communication_loss = 0.0  # Initialize as Python float to avoid tensor broadcasting issues
    
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
                is_adjacent = (distance <= 1.5).item()  # Convert to scalar
                
                # Communication strength inversely related to distance
                comm_strength = torch.exp(-distance / 5.0)  # 5.0 is communication range
                
                # Hidden state similarity (higher is better for nearby agents)
                state_similarity = torch.cosine_similarity(
                    hidden_states[b, i].unsqueeze(0),
                    hidden_states[b, j].unsqueeze(0)
                ).item()  # Convert to scalar
                
                # Base communication loss
                base_comm_loss = (comm_strength * (1.0 - state_similarity)).item()  # Convert to scalar
                
                # Apply distance-based communication weight adjustment (same as coordination)
                if is_adjacent:
                    # Scale the communication weight for adjacent agents
                    comm_weight_scale = config.get('communication_weight_scale', 2.0)
                    comm_weight = communication_weight * comm_weight_scale
                    adjacent_communication_count += 1
                else:
                    # Use base weight for non-adjacent agents
                    comm_weight = communication_weight
                
                # Apply communication weight to the loss
                weighted_comm_loss = comm_weight * base_comm_loss
                communication_loss += weighted_comm_loss
    
    # Log adjacent communication enhancement statistics
    from debug_utils import debug_print
    debug_print(f"📡 Adjacent Communication: {adjacent_communication_count}/{total_pairs_processed} pairs enhanced "
               f"(distance-based detection)")
    
    # Convert to tensor and normalize
    total_pairs = batch_size * num_agents * (num_agents - 1) / 2
    if total_pairs > 0:
        normalized_loss = communication_loss / total_pairs
    else:
        normalized_loss = 0.0
    
    # Ensure the result is a proper tensor on the correct device
    if isinstance(normalized_loss, torch.Tensor):
        return normalized_loss.to(device)
    else:
        return torch.tensor(normalized_loss, device=device, dtype=torch.float32)


def compute_coordination_reward(predicted_actions, agent_positions, coordination_weight, device, goals=None, config=None):
    """
    Compute coordination reward that encourages good coordination between agents.
    This returns a REWARD (negative loss) when agents coordinate well.
    Also penalizes bad coordination in stress environments (stepping into other agents' intentions).
    Agents within one grid unit (adjacent) have doubled coordination effects.
    
    Args:
        predicted_actions: Predicted actions [batch, num_agents, action_dim]
        agent_positions: Current agent positions [batch, num_agents, 2]
        coordination_weight: Weight for coordination effects
        device: Device for tensor operations
        goals: Goal positions [batch, num_agents, 2] for stress environment detection
    """
    
    batch_size, num_agents = predicted_actions.shape[:2]
    coordination_reward = 0.0  # Initialize as Python float - this will be a REWARD
    
    # Action deltas
    action_deltas = torch.tensor([
        [0, 0],   # 0: Stay
        [1, 0],   # 1: Right
        [0, 1],   # 2: Up  
        [-1, 0],  # 3: Left
        [0, -1],  # 4: Down
    ], device=device, dtype=torch.float32)
    
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
                is_adjacent = (current_distance <= 1.5).item()  # Convert to scalar
                
                # Predicted next positions
                action_i = torch.argmax(predicted_actions[b, i])
                action_j = torch.argmax(predicted_actions[b, j])
                
                next_pos_i = pos_i + action_deltas[action_i]
                next_pos_j = pos_j + action_deltas[action_j]
                
                # 1. Collision avoidance reward: reward agents for NOT colliding
                collision_reward = 0.0
                if torch.norm(next_pos_i - next_pos_j).item() >= 0.1:  # Different positions = good!
                    collision_reward = 1.0  # Reward for avoiding collision
                
                # 2. Traffic coordination reward: reward good coordination when close
                traffic_reward = 0.0
                if current_distance.item() < 2.0:  # Close agents need coordination
                    # Reward if at least one agent stays to avoid gridlock
                    one_staying = (action_i.item() == 0) or (action_j.item() == 0)
                    if one_staying:
                        traffic_reward = 0.5  # Reward for good coordination
                
                # 3. Intention interference penalty: penalize stepping into other agents' intentions
                intention_penalty = 0.0
                if goals is not None:
                    # Check if agents are in stress environment (both moving towards goals)
                    goal_i = goals[b, i]
                    goal_j = goals[b, j]
                    
                    # Check if both agents are actively moving towards their goals (not at goal)
                    dist_to_goal_i = torch.norm(pos_i - goal_i).item()
                    dist_to_goal_j = torch.norm(pos_j - goal_j).item()
                    
                    both_active = (dist_to_goal_i > 1.0) and (dist_to_goal_j > 1.0)  # Both not at goal
                    both_moving = (action_i.item() != 0) and (action_j.item() != 0)  # Both moving
                    
                    if both_active and both_moving:
                        # Check if agent i is stepping into agent j's intended path
                        # Agent j's intention: direction from current position to goal
                        intention_j = goal_j - pos_j
                        intention_j_norm = intention_j / (torch.norm(intention_j) + 1e-8)
                        
                        # Agent i's next position relative to agent j's current position
                        i_move_direction = next_pos_i - pos_j
                        
                        # Check if agent i is moving into agent j's intended direction
                        if torch.norm(i_move_direction) > 0.1:  # Agent i is moving relative to j
                            i_move_norm = i_move_direction / (torch.norm(i_move_direction) + 1e-8)
                            intention_similarity = torch.dot(i_move_norm, intention_j_norm).item()
                            
                            # High similarity means agent i is stepping into agent j's intention
                            if intention_similarity > 0.7:  # Strong intention interference
                                intention_penalty += 2.0  # Penalize intention interference
                        
                        # Check the reverse: agent j stepping into agent i's intention
                        intention_i = goal_i - pos_i
                        intention_i_norm = intention_i / (torch.norm(intention_i) + 1e-8)
                        j_move_direction = next_pos_j - pos_i
                        
                        if torch.norm(j_move_direction) > 0.1:  # Agent j is moving relative to i
                            j_move_norm = j_move_direction / (torch.norm(j_move_direction) + 1e-8)
                            intention_similarity = torch.dot(j_move_norm, intention_i_norm).item()
                            
                            if intention_similarity > 0.7:  # Strong intention interference
                                intention_penalty += 2.0  # Penalize intention interference
                
                # Apply distance-based coordination weight adjustment
                if is_adjacent:
                    # Scale the coordination weight for adjacent agents
                    coord_weight_scale = config.get('coordination_weight_scale', 2.0) if config else 2.0
                    coord_weight = coordination_weight * coord_weight_scale
                    adjacent_coordination_count += 1
                else:
                    # Use base weight for non-adjacent agents
                    coord_weight = coordination_weight
                
                # Apply coordination weight to rewards (ensure scalar result)
                # Note: intention_penalty is subtracted because it's a penalty (reduces reward)
                weighted_reward = coord_weight * (collision_reward + traffic_reward - intention_penalty)
                coordination_reward += weighted_reward
    # Log adjacent coordination statistics
    from debug_utils import debug_print
    debug_print(f"🤝 Adjacent Coordination: {adjacent_coordination_count}/{total_pairs_processed} pairs enhanced "
               f"(distance-based detection)")
    
    # Convert to tensor and normalize
    total_pairs = batch_size * num_agents * (num_agents - 1) / 2
    if total_pairs > 0:
        normalized_reward = coordination_reward / total_pairs
    else:
        normalized_reward = 0.0
    
    # Return NEGATIVE reward (so it reduces the loss when coordination is good)
    if isinstance(normalized_reward, torch.Tensor):
        return -normalized_reward.to(device)  # Negative because it's a reward
    else:
        return torch.tensor(-normalized_reward, device=device, dtype=torch.float32)  # Negative because it's a reward


def compute_temporal_consistency_loss(sequence_outputs):
    """Compute temporal consistency loss to encourage smooth transitions"""
    if sequence_outputs.shape[1] < 2:
        return torch.tensor(0.0, device=sequence_outputs.device)
    
    # Compute differences between consecutive timesteps
    temporal_diffs = sequence_outputs[:, 1:] - sequence_outputs[:, :-1]
    
    # L2 norm of differences (encourage small changes)
    temporal_loss = torch.mean(torch.norm(temporal_diffs, dim=-1))
    
    return temporal_loss


def compute_enhanced_collision_loss(final_outputs, agent_positions, config, device, prev_positions=None):
    """
    Compute collision loss using collision_utils for detection only.
    
    Args:
        final_outputs: Model outputs [batch, agents, action_dim]
        agent_positions: Agent positions [batch, agents, 2]
        config: Configuration dictionary
        device: Torch device
        prev_positions: Previous agent positions [batch, agents, 2] for edge collision detection
        
    Returns:
        collision_loss: Computed collision loss
    """
    collision_loss_weight = config.get('collision_loss_weight', 0.5)
    
    if collision_loss_weight <= 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    
    try:
        # Collision configuration from config
        collision_config = {
            'vertex_collision_weight': config.get('vertex_collision_weight', 1.0),
            'edge_collision_weight': config.get('edge_collision_weight', 0.5),
            'collision_loss_type': config.get('collision_loss_type', 'l2'),
            'use_future_collision_penalty': config.get('use_future_collision_penalty', False),
            'future_collision_steps': config.get('future_collision_steps', 2),
            'future_collision_weight': config.get('future_collision_weight', 0.8),
            'future_step_decay': config.get('future_step_decay', 0.7),
            'separate_collision_types': config.get('separate_collision_types', True),
            'per_agent_loss': config.get('per_agent_collision_loss', True),
            'use_continuous_collision': config.get('use_continuous_collision', True),  # Enable continuous collision
            'collision_threshold': config.get('collision_threshold', 2.0)  # Distance threshold for continuous collision
        }
        
        # Fix 3: Discretize positions to make same-cell equality robust to FP errors
        curr_pos_discrete = torch.round(agent_positions).to(torch.int)
        prev_pos_discrete = None
        if prev_positions is not None:
            prev_pos_discrete = torch.round(prev_positions).to(torch.int)
        
        # Flatten logits for collision loss computation
        logits_flat = final_outputs.reshape(-1, final_outputs.size(-1))
        collision_loss_tensor, collision_info = compute_collision_loss(
            logits_flat, curr_pos_discrete.float(), prev_positions=prev_pos_discrete.float() if prev_pos_discrete is not None else None,
            collision_config=collision_config
        )
        
        # Handle per-agent vs aggregate collision loss
        if collision_info.get('per_agent_loss', False):
            collision_loss = collision_loss_tensor.mean()  # Average over agents
        else:
            collision_loss = collision_loss_tensor
            
        return collision_loss_weight * collision_loss
        
    except Exception as e:
        print(f"Warning: Collision loss computation failed: {e}")
        return torch.tensor(0.0, device=device, dtype=torch.float32)


def compute_goal_progress_reward(current_positions, goals, config, obstacles=None, moved_away_mask=None):
    """
    Compute simple proximity reward based on current distance to goals.
    Includes reached goals as danger zones to encourage rerouting around occupied goals.
    
    Args:
        current_positions: Current agent positions [batch, agents, 2]
        goals: Goal positions [batch, agents, 2]
        config: Configuration dictionary
        obstacles: Obstacle positions (list or tensor) for obstacle-aware progress
        moved_away_mask: [batch, agents] boolean tensor marking agents moving away from goals
        
    Returns:
        progress_reward: Computed progress reward (scalar)
    """
    if not config.get('use_goal_progress_reward', False):
        return 0.0
    
    goal_progress_weight = float(config.get('goal_progress_weight', 1e-3))
    success_threshold = config.get('goal_success_threshold', 1.0)
    
    # Calculate distances to goals
    if current_positions.dim() == 2:
        # Single batch case
        current_positions_batch = current_positions.unsqueeze(0)
        goals_batch = goals.unsqueeze(0)
        if moved_away_mask is not None:
            moved_away_mask_batch = moved_away_mask.unsqueeze(0)
        else:
            moved_away_mask_batch = None
    else:
        current_positions_batch = current_positions
        goals_batch = goals
        moved_away_mask_batch = moved_away_mask
    
    # Calculate distance to goals for all agents
    distances = torch.norm(current_positions_batch - goals_batch, dim=-1)  # [batch, agents]
    
    # Simple proximity reward: higher reward for being closer to goal
    # Use inverse distance (with small epsilon to avoid division by zero)
    epsilon = 0.1
    proximity_reward = 1.0 / (distances + epsilon)  # [batch, agents]
    
    # Apply mask to prevent proximity rewards for agents moving away from goals
    if moved_away_mask_batch is not None:
        # Zero out proximity rewards for agents moving away
        proximity_reward = proximity_reward * (~moved_away_mask_batch).float()
        
        # Debug: Log how many agents had their proximity rewards blocked
        if moved_away_mask_batch.any():
            blocked_agents = moved_away_mask_batch.sum().item()
            total_agents = moved_away_mask_batch.numel()
            debug_print(f"🚫 Proximity rewards blocked for {blocked_agents}/{total_agents} agents moving away from goals")
    
    # Increase proximity reward by configurable scale for stronger goal-seeking behavior
    proximity_scale = config.get('proximity_reward_scale', 25.0)
    proximity_reward = proximity_reward * proximity_scale
    
    # Average across all agents and batches
    mean_proximity_reward = torch.mean(proximity_reward)
    
    # Apply weight and convert to scalar
    progress_reward = (mean_proximity_reward * goal_progress_weight).item()
    
    return progress_reward


def compute_all_losses(sequence_outputs, agent_positions, goals, 
                      weights, config, device, fov_observations=None,
                      stdp_system=None, collided_agents=None, prev_positions=None,
                      stress_inhibition_weight=None, selective_movement_weight=None,
                      oscillation_penalty_weight=None, batch_id=0, epoch=0, obstacles=None,
                      moved_away_mask=None):
    """
    Compute all losses in one centralized function.
    
    Args:
        sequence_outputs: Model outputs [batch, seq_len, agents, action_dim]
        agent_positions: Agent positions [batch, agents, 2]
        goals: Goal positions [batch, agents, 2]
        weights: Dictionary of loss weights
        config: Configuration dictionary
        device: Torch device
        fov_observations: FOV observations [batch, agents, 2, 7, 7] (for STDP coordination)
        stdp_system: STDPCoordinationSystem instance (optional)
        collided_agents: [batch, agents] boolean tensor marking collided agents (no rewards)
        stress_inhibition_weight: Learnable stress inhibition weight (tensor)
        selective_movement_weight: Learnable selective movement weight (tensor)
        oscillation_penalty_weight: Learnable oscillation penalty weight (tensor)
        batch_id: Batch identifier for oscillation tracking
        epoch: Current training epoch for collision weight annealing
        obstacles: Obstacle positions (list or tensor) for obstacle-aware progress
        moved_away_mask: [batch, agents] boolean tensor marking agents moving away from goals
        disable_position_rewards: If True, skip position-based rewards (applied directly to sequence)
        
    Returns:
        Dictionary containing all computed losses and metrics
    """
    final_outputs = sequence_outputs[:, -1]  # [batch, agents, action_dim]
    
    # Temporal cross-correlation loss for spike trains
    # Using final_outputs directly as spike trains (assuming they are in the correct format)
    temporal_cross_corr_loss = compute_temporal_cross_correlation_loss(
        final_outputs, agent_positions, device=device, weight=weights.get('temporal_cross_correlation_weight', 0.1)
    )
    
    # Communication loss - encourage agent communication and information sharing
    communication_loss = compute_communication_loss(
        final_outputs, agent_positions, weights.get('communication_weight', 1.0), device, config
    )
    
    # Traditional coordination reward - reward good coordination between nearby agents
    coordination_loss = compute_coordination_reward(
        final_outputs, agent_positions, weights.get('coordination_weight', 1.0), device, goals=goals, config=config
    )
    
    # Collision loss
    collision_loss_raw = compute_enhanced_collision_loss(
        final_outputs, agent_positions, config, device, prev_positions=prev_positions
    )
    
    # Apply collision normalization for all epochs (removed epoch 0-1 exception)
    # 1. Normalize collision loss to [0, 1] range
    batch_size, num_agents = agent_positions.shape[:2]
    max_steps = config.get('sequence_length', 60)  # Get max steps from config
    collision_loss_normalized = collision_loss_raw / (num_agents * max_steps)
    
    # (Optional) Saturate raw collisions - one crash hurts, 20 don't explode the loss
    collision_loss_saturated = torch.min(collision_loss_normalized, torch.tensor(1.0, device=device))
    
    debug_print(f"📊 Collision (epoch {epoch}): raw={collision_loss_raw.item():.3f}, "
               f"normalized={collision_loss_normalized.item():.3f}, saturated={collision_loss_saturated.item():.3f}")
    
    # Use collision weight from config
    collision_weight = weights.get('collision_loss_weight', 50.0)
    
    # Apply collision weight to collision loss
    collision_loss = collision_weight * collision_loss_saturated
    
    # Stress inhibition mechanism - encourage staying in high-density zones
    stress_inhibition_loss = torch.tensor(0.0, device=device)
    stress_stats = {}
    selective_movement_loss = torch.tensor(0.0, device=device)
    movement_stats = {}
    
    # Apply stress inhibition if enabled in config
    if config.get('use_stress_inhibition', True):
        # Calculate epoch-based progressive stress weight (0 to full weight over 10 epochs)
        stress_ramp_epochs = config.get('stress_ramp_epochs', 10)
        base_stress_weight = stress_inhibition_weight if stress_inhibition_weight is not None else config.get('stress_inhibition_weight', 1.0)
        
        # Progressive weight: starts at 0, reaches full weight at stress_ramp_epochs
        if epoch < stress_ramp_epochs:
            epoch_progress = epoch / stress_ramp_epochs  # 0.0 to 1.0 over 10 epochs
            stress_weight = base_stress_weight * epoch_progress
            # Convert tensors to scalars for formatting
            stress_weight_val = stress_weight.item() if torch.is_tensor(stress_weight) else stress_weight
            base_weight_val = base_stress_weight.item() if torch.is_tensor(base_stress_weight) else base_stress_weight
            debug_print(f"🧠 Stress inhibition ramping: epoch {epoch}/{stress_ramp_epochs}, weight={stress_weight_val:.4f} (base={base_weight_val:.4f})")
        else:
            stress_weight = base_stress_weight
        
        stress_radius = config.get('stress_radius', 1.5)
        min_stress_agents = config.get('min_stress_agents', 3)
        
        # Compute stress inhibition loss
        stress_inhibition_loss, stress_stats = compute_stress_inhibition_loss(
            final_outputs, agent_positions, stress_weight, stress_radius, min_stress_agents, goal_positions=goals, config=config
        )
        
        # Debug: Print stress zone detection
        if stress_stats['total_stressed_agents'] > 0:
            debug_print(f"🧠 Stress zones detected: {stress_stats['total_stressed_agents']} agents, "
                       f"avg intensity: {stress_stats['avg_stress_intensity']:.3f}")
        
        # Compute selective movement loss if stress zones detected
        if stress_stats.get('stress_zones_detected', False):
            # Calculate epoch-based progressive selective movement weight (same as stress inhibition)
            base_selective_weight = selective_movement_weight if selective_movement_weight is not None else config.get('selective_movement_weight', 0.5)
            
            # Use the same progressive scaling as stress inhibition
            if epoch < stress_ramp_epochs:
                selective_weight = base_selective_weight * epoch_progress
                # Convert tensors to scalars for formatting
                selective_weight_val = selective_weight.item() if torch.is_tensor(selective_weight) else selective_weight
                base_selective_val = base_selective_weight.item() if torch.is_tensor(base_selective_weight) else base_selective_weight
                debug_print(f"🎯 Selective movement ramping: epoch {epoch}/{stress_ramp_epochs}, weight={selective_weight_val:.4f} (base={base_selective_val:.4f})")
            else:
                selective_weight = base_selective_weight
            
            selective_movement_loss, movement_stats = compute_selective_movement_loss(
                final_outputs, agent_positions, selective_weight, stress_radius, min_stress_agents, config
            )
    
    # STDP-based coordination system
    stdp_coordination_loss = torch.tensor(0.0, device=device)
    coordination_stats = {}
    updated_stdp_system = stdp_system
    
    if config.get('use_stdp_coordination', False) and fov_observations is not None:
        stdp_coordination_loss, updated_stdp_system, coordination_stats = compute_stdp_coordination_loss(
            final_outputs, agent_positions, goals, fov_observations, config, device, stdp_system
        )
    
    # Progressive goal rewards (bigger rewards for more agents reaching goals)
    progress_reward = 0.0
    progressive_goal_reward = 0.0
    
    if config.get('use_goal_progress_reward', False):
        # Original proximity-based progress reward
        progress_reward = compute_goal_progress_reward(
            agent_positions, goals, config, obstacles=obstacles, moved_away_mask=moved_away_mask
        )
        
        # NEW: Progressive goal rewards - more goals reached = bigger bonus!
        goal_tolerance = config.get('goal_tolerance', 0.8)  # Distance considered "at goal"
        goal_distances = torch.norm(agent_positions - goals, dim=-1)  # [batch, agents]
        agents_at_goal = goal_distances <= goal_tolerance  # [batch, agents]
        goals_reached_per_batch = torch.sum(agents_at_goal, dim=1).float()  # [batch]
        
        # Progressive rewards: exponential scaling for multiple goals
        batch_size, num_agents = agent_positions.shape[:2]
        goal_rewards = torch.zeros_like(goals_reached_per_batch)
        
        for num_goals in range(1, num_agents + 1):
            # Exponential scaling: 1 goal = 0.5, 2 goals = 1.5, 3 goals = 3.5, etc.
            bonus = (0.5 * (2 ** (num_goals - 1)))  # 0.5, 1.0, 2.0, 4.0, 8.0...
            goal_rewards += (goals_reached_per_batch >= num_goals).float() * bonus
        
        progressive_goal_reward = torch.mean(goal_rewards) * config.get('progressive_goal_weight', 0.3)
        
        debug_print(f"🎯 Goals reached per batch (avg): {torch.mean(goals_reached_per_batch).item():.2f}/{num_agents}")
        debug_print(f"🎯 Progressive goal reward: {progressive_goal_reward.item():.4f}")
        
        # Note: No longer excluding rewards for collided agents - let them still learn from progress rewards
    
    # 4. Add small movement bonus to prevent "safe standing still" behavior
    movement_bonus = 0.0
    if prev_positions is not None and config.get('use_movement_bonus', True):
        # Calculate distance moved from previous position
        position_diff = torch.norm(agent_positions - prev_positions, dim=-1)  # [batch, agents]
        # Small bonus for movement (0.1 * distance_moved)
        movement_bonus_weight = config.get('movement_bonus_weight', 0.1)
        movement_bonus = torch.mean(position_diff) * movement_bonus_weight
        debug_print(f"🚶 Movement bonus: {movement_bonus.item():.3f} (avg movement: {torch.mean(position_diff).item():.3f})")
    
    # 5. Intent field attraction reward - encourage agents to move towards goals
    intent_attraction_reward = compute_intent_field_attraction_reward(
        final_outputs, agent_positions, goals, config, device
    )
    debug_print(f"🎯 Intent field attraction reward: {intent_attraction_reward.item():.3f}")
    
    # Stay action reward - encourage conservative behavior
    stay_reward = torch.tensor(0.0, device=device)
    stay_stats = {}
    if config.get('use_stay_action_reward', True):
        stay_weight = config.get('stay_action_reward_weight', 0.1)
        stay_reward, stay_stats = compute_stay_action_reward(
            final_outputs, stay_weight
        )
        # Note: No longer excluding rewards for collided agents - let them still learn from stay rewards
    
    # 5. Intent field attraction reward - encourage agents to move towards goals
    intent_attraction_reward = compute_intent_field_attraction_reward(
        final_outputs, agent_positions, goals, config, device
    )
    debug_print(f"🎯 Intent field attraction reward: {intent_attraction_reward.item():.3f}")
    
    # Combine all losses (oscillation penalty now applied per timestep in training loop)
    # Use the progressively ramped weights for stress and selective movement
    if config.get('use_stress_inhibition', True):
        effective_stress_weight = stress_weight
        effective_selective_weight = selective_weight if stress_stats.get('stress_zones_detected', False) else 0.0
        
        # Scale communication and coordination weights when stress zones are detected
        if stress_stats.get('stress_zones_detected', False):
            comm_scale = config.get('communication_weight_scale', 2.0)
            coord_scale = config.get('coordination_weight_scale', 2.0)
            effective_communication_weight = weights.get('communication_weight', 1.0) * comm_scale
            effective_coordination_weight = weights.get('coordination_weight', 1.0) * coord_scale
            debug_print(f"🚨 Stress zones detected! Scaling communication/coordination weights: comm={effective_communication_weight:.3f}, coord={effective_coordination_weight:.3f}")
        else:
            effective_communication_weight = weights.get('communication_weight', 1.0)
            effective_coordination_weight = weights.get('coordination_weight', 1.0)
    else:
        effective_stress_weight = 0.0
        effective_selective_weight = 0.0
        effective_communication_weight = weights.get('communication_weight', 1.0)
        effective_coordination_weight = weights.get('coordination_weight', 1.0)
    
    total_additional_loss = (
        effective_communication_weight * communication_loss +  # Agent communication loss (doubled in stress)
        effective_coordination_weight * coordination_loss +  # Agent coordination loss (doubled in stress)
        weights.get('temporal_cross_correlation_weight', 0.1) * temporal_cross_corr_loss +  # Temporal cross-correlation loss (using original)
        collision_loss +  # Already weighted collision loss (no annealing)
        effective_stress_weight * stress_inhibition_loss +  # Progressive stress inhibition loss
        effective_selective_weight * selective_movement_loss +  # Progressive selective movement loss
        stdp_coordination_loss -  # STDP-based coordination enhancement
        progress_reward -  # Goal progress reward (negative because it's a reward)
        progressive_goal_reward -  # Progressive goal reward (bigger bonus for more goals reached)
        stay_reward -  # Stay action reward (negative because it's a reward)
        movement_bonus -  # Movement bonus to prevent standing still (negative because it's a reward)
        intent_attraction_reward  # Intent field attraction reward (negative because it's a reward)
    )

    return {
        'total_additional_loss': total_additional_loss,
        'communication_loss': communication_loss.detach().clone().to(device=device, dtype=torch.float32),  # Actual communication loss
        'coordination_loss': coordination_loss.detach().clone().to(device=device, dtype=torch.float32),  # Actual coordination loss
        'temporal_loss': temporal_cross_corr_loss,  # Map temporal cross-correlation to temporal_loss
        'temporal_corr_loss': temporal_cross_corr_loss,
        'collision_loss': collision_loss,
        'stress_inhibition_loss': stress_inhibition_loss,
        'selective_movement_loss': selective_movement_loss,
        'stdp_coordination_loss': stdp_coordination_loss,
        'progress_reward': progress_reward.detach().clone().to(device) if torch.is_tensor(progress_reward) else torch.tensor(progress_reward, device=device),
        'progressive_goal_reward': progressive_goal_reward.detach().clone().to(device) if torch.is_tensor(progressive_goal_reward) else torch.tensor(progressive_goal_reward, device=device),
        'stay_reward': stay_reward.detach().clone().to(device) if torch.is_tensor(stay_reward) else torch.tensor(stay_reward, device=device),
        'movement_bonus': movement_bonus.detach().clone().to(device) if torch.is_tensor(movement_bonus) else torch.tensor(movement_bonus, device=device),
        'intent_attraction_reward': intent_attraction_reward.detach().clone().to(device) if torch.is_tensor(intent_attraction_reward) else torch.tensor(intent_attraction_reward, device=device),
        'collision_weight': collision_weight.detach().clone().to(device) if torch.is_tensor(collision_weight) else torch.tensor(collision_weight, device=device),  # Return current collision weight for logging
        'effective_stress_weight': effective_stress_weight.detach().clone().to(device) if torch.is_tensor(effective_stress_weight) else torch.tensor(effective_stress_weight, device=device),  # Progressive stress inhibition weight
        'effective_selective_weight': effective_selective_weight.detach().clone().to(device) if torch.is_tensor(effective_selective_weight) else torch.tensor(effective_selective_weight, device=device),  # Progressive selective movement weight
        'effective_communication_weight': effective_communication_weight.detach().clone().to(device) if torch.is_tensor(effective_communication_weight) else torch.tensor(effective_communication_weight, device=device),  # Communication weight (doubled in stress)
        'effective_coordination_weight': effective_coordination_weight.detach().clone().to(device) if torch.is_tensor(effective_coordination_weight) else torch.tensor(effective_coordination_weight, device=device),  # Coordination weight (doubled in stress)
        'updated_stdp_system': updated_stdp_system,
        'stress_stats': stress_stats,
        'movement_stats': movement_stats,
        'coordination_stats': coordination_stats,
        'stay_stats': stay_stats
    }


def compute_temporal_cross_correlation_loss(spike_trains, agent_positions, mask=None, device=None, weight=1.0):
    """
    Compute temporal cross-correlation loss for spike trains between agents.
    Args:
        spike_trains: [batch, num_agents, T] spike trains (binary or float)
        agent_positions: [batch, num_agents, 2] positions (for proximity mask)
        mask: Optional [batch, num_agents, num_agents] mask for which pairs to include
        device: torch.device
        weight: scaling factor for the loss
    Returns:
        loss: scalar tensor
    """
    import torch.nn.functional as F
    batch_size, num_agents, T = spike_trains.shape
    if device is None:
        device = spike_trains.device
    
    # Smoothing kernel for predictive correlation
    kernel = torch.tensor([[0.25, 0.5, 0.25]], device=device).unsqueeze(0)  # [1,1,3] for conv1d
    smoothed = F.conv1d(spike_trains.reshape(-1, 1, T), kernel, padding=1).reshape(batch_size, num_agents, T)
    
    # Compute pairwise cross-correlation for all agent pairs
    loss = 0.0
    count = 0
    for b in range(batch_size):
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    continue
                # Optionally mask by proximity
                if mask is not None and not mask[b, i, j]:
                    continue
                # Optionally mask by distance (e.g., only nearby agents)
                if agent_positions is not None:
                    dist = torch.norm(agent_positions[b, i] - agent_positions[b, j])
                    if dist > 2.0:
                        continue
                # Temporal cross-correlation (predictive):
                corr = torch.sum(smoothed[b, i] * spike_trains[b, j]) / (T + 1e-8)
                loss += (1.0 - corr)  # Encourage high correlation
                count += 1
    if count > 0:
        loss = loss / count
    return weight * loss


def detect_stress_zones(agent_positions, stress_radius=1.5, min_agents=2, config=None):
    """
    Detect high-stress zones where multiple agents are within close proximity.
    Enhanced for better detection in small grids.
    
    Args:
        agent_positions: [batch_size, num_agents, 2] agent positions
        stress_radius: float, radius to consider for stress detection
        min_agents: int, minimum number of agents needed to create stress
        
    Returns:
        stress_agents: [batch_size, num_agents] bool tensor indicating which agents are in stress zones
        stress_intensity: [batch_size, num_agents] float tensor indicating stress level (0.0 to 1.0)
    """
    batch_size, num_agents, _ = agent_positions.shape
    device = agent_positions.device
    
    # Initialize stress indicators
    stress_agents = torch.zeros(batch_size, num_agents, dtype=torch.bool, device=device)
    stress_intensity = torch.zeros(batch_size, num_agents, dtype=torch.float32, device=device)
    
    for b in range(batch_size):
        for i in range(num_agents):
            # Count agents within stress radius
            distances = torch.norm(agent_positions[b] - agent_positions[b, i], dim=1)
            nearby_agents = (distances <= stress_radius) & (distances > 0.1)  # Exclude self
            nearby_count = torch.sum(nearby_agents).item()
            
            # Count very close agents (adjacent cells) for higher stress
            very_close_agents = (distances <= 1.5) & (distances > 0.1)  # Adjacent or very close
            very_close_count = torch.sum(very_close_agents).item()
            
            if nearby_count >= (min_agents - 1):  # -1 because we exclude self
                stress_agents[b, i] = True
                # Base stress from nearby agents
                base_stress = min(1.0, (nearby_count - (min_agents - 2)) / 3.0)
                # Bonus stress for very close agents
                close_stress_bonus_per_agent = config.get('close_stress_bonus_per_agent', 0.3) if config else 0.3
                close_stress_bonus = very_close_count * close_stress_bonus_per_agent  # Configurable per very close agent
                # Combine stresses (capped at 1.0)
                stress_intensity[b, i] = min(1.0, base_stress + close_stress_bonus)
            elif very_close_count >= 1:  # Even with fewer agents, very close proximity creates stress
                stress_agents[b, i] = True
                # Strong stress for very close agents
                stress_multiplier = config.get('stress_intensity_multiplier', 0.4) if config else 0.4
                stress_max = config.get('stress_intensity_max', 0.7) if config else 0.7
                stress_intensity[b, i] = min(stress_max, very_close_count * stress_multiplier)  # Strong stress for close agents
    
    return stress_agents, stress_intensity


def compute_stress_inhibition_loss(model_outputs, agent_positions, stress_weight=1.0, 
                                 stress_radius=1.5, min_agents=2, goal_positions=None, config=None):
    """
    Compute inhibition loss that encourages agents to stay in place during high-stress situations.
    
    Args:
        model_outputs: [batch_size, num_agents, action_dim] model predictions
        agent_positions: [batch_size, num_agents, 2] agent positions  
        stress_weight: float, weight for stress inhibition loss
        stress_radius: float, radius to consider for stress detection
        min_agents: int, minimum agents needed to trigger stress
        goal_positions: [batch_size, num_agents, 2] goal positions (optional, to exclude agents at goal)
        
    Returns:
        inhibition_loss: tensor, loss encouraging staying behavior in stress zones
        stress_stats: dict, statistics about stress detection
    """
    batch_size, num_agents, action_dim = model_outputs.shape
    device = model_outputs.device
    
    # Detect stress zones
    stress_agents, stress_intensity = detect_stress_zones(
        agent_positions, stress_radius, min_agents, config
    )
    
    # Filter out agents that have reached their goals
    if goal_positions is not None:
        agents_at_goal = get_agents_at_goal(agent_positions, goal_positions)
        stress_agents = stress_agents & (~agents_at_goal)  # Remove agents at goal from stress calculation
    
    # Get action probabilities
    action_probs = F.softmax(model_outputs, dim=-1)
    
    # Action 0 is "stay" - we want to encourage this in stress zones
    stay_probs = action_probs[:, :, 0]  # [batch_size, num_agents]
    
    # Compute inhibition loss: encourage staying for stressed agents
    # Higher stress intensity = stronger preference for staying
    # Use a stronger incentive for high-stress situations
    stress_exponent = config.get('stress_bonus_exponent', 1.5) if config else 1.5
    stay_bonus = (stress_intensity ** stress_exponent) * torch.log(stay_probs + 1e-8)  # Configurable stress weighting
    
    # Only apply inhibition to agents actually in stress zones
    masked_stay_bonus = stress_agents.float() * stay_bonus
    
    # Loss is negative because we want to maximize stay probability (minimize negative log prob)
    # Add extra penalty for moving in very high-stress situations
    movement_penalty = stress_agents.float() * stress_intensity * torch.sum(action_probs[:, :, 1:], dim=-1)  # Penalize all movement actions
    movement_penalty_multiplier = config.get('movement_penalty_multiplier', 2.0) if config else 2.0
    inhibition_loss = -torch.mean(masked_stay_bonus) * stress_weight + torch.mean(movement_penalty) * stress_weight * movement_penalty_multiplier  # Configurable penalty multiplier
    
    # Collect statistics
    total_stressed = torch.sum(stress_agents).item()
    avg_stress_intensity = torch.mean(stress_intensity[stress_agents]).item() if total_stressed > 0 else 0.0
    
    stress_stats = {
        'total_stressed_agents': total_stressed,
        'avg_stress_intensity': avg_stress_intensity,
        'stress_zones_detected': total_stressed > 0
    }
    
    return inhibition_loss, stress_stats


def compute_selective_movement_loss(model_outputs, agent_positions, stress_weight=0.5,
                                  stress_radius=1.5, min_agents=3, config=None):
    """
    Compute loss that allows selective movement in stress zones - only one agent moves while others stay.
    
    Args:
        model_outputs: [batch_size, num_agents, action_dim] model predictions
        agent_positions: [batch_size, num_agents, 2] agent positions
        stress_weight: float, weight for selective movement loss
        stress_radius: float, radius to consider for stress detection  
        min_agents: int, minimum agents needed to trigger stress
        
    Returns:
        selective_loss: tensor, loss encouraging coordinated movement
        movement_stats: dict, statistics about movement coordination
    """
    batch_size, num_agents, action_dim = model_outputs.shape
    device = model_outputs.device
    
    # Detect stress zones
    stress_agents, stress_intensity = detect_stress_zones(
        agent_positions, stress_radius, min_agents, config
    )
    
    # Get action probabilities
    action_probs = F.softmax(model_outputs, dim=-1)
    
    # Movement probability (1 - stay_probability)
    movement_probs = 1.0 - action_probs[:, :, 0]  # [batch_size, num_agents]
    
    selective_loss = torch.tensor(0.0, device=device)
    total_stress_groups = 0
    
    for b in range(batch_size):
        if not torch.any(stress_agents[b]):
            continue
            
        # Find groups of stressed agents within stress radius
        stressed_indices = torch.where(stress_agents[b])[0]
        
        if len(stressed_indices) < min_agents:
            continue
            
        # For each group, encourage only one agent to have high movement probability
        # while others have low movement probability
        for i in range(len(stressed_indices)):
            center_agent = stressed_indices[i]
            
            # Find other stressed agents within radius of this agent
            distances = torch.norm(
                agent_positions[b, stressed_indices] - agent_positions[b, center_agent], 
                dim=1
            )
            nearby_mask = distances <= stress_radius
            nearby_agents = stressed_indices[nearby_mask]
            
            if len(nearby_agents) >= min_agents:
                total_stress_groups += 1
                
                # Get movement probabilities for this group
                group_movement_probs = movement_probs[b, nearby_agents]
                
                # Encourage sparse movement: ideally only one agent moves
                # We want to minimize entropy (encourage one agent to dominate) 
                # and maximize the difference between the most active agent and others
                
                # Method 1: Entropy penalty (lower entropy = more coordination)
                # Add small epsilon to prevent log(0)
                safe_probs = group_movement_probs + 1e-8
                movement_entropy = -torch.sum(safe_probs * torch.log(safe_probs))
                
                # Method 2: Coordination bonus - reward when one agent is clearly more active
                max_movement = torch.max(group_movement_probs)
                avg_others = torch.mean(group_movement_probs)
                coordination_bonus = max_movement - avg_others
                
                # Loss should be positive when coordination is poor (high entropy, low coordination)
                # We want to minimize this loss, so higher values = worse coordination
                sparsity_loss = movement_entropy - coordination_bonus  # High entropy = bad, high coordination = good
                selective_loss += sparsity_loss * stress_weight
    
    # Normalize by number of stress groups
    if total_stress_groups > 0:
        selective_loss = selective_loss / total_stress_groups
    
    movement_stats = {
        'stress_groups_processed': total_stress_groups,
        'selective_coordination_applied': total_stress_groups > 0
    }
    
    return selective_loss, movement_stats


def compute_stdp_coordination_loss(model_outputs, agent_positions, goals, fov_observations,
                                 config, device, stdp_system=None):
    """
    Compute STDP-based coordination loss using facilitation traces and intention flow fields.
    
    Args:
        model_outputs: [batch_size, num_agents, action_dim] model predictions
        agent_positions: [batch_size, num_agents, 2] agent positions
        goals: [batch_size, num_agents, 2] goal positions
        fov_observations: [batch_size, num_agents, 2, 7, 7] FOV observations
        config: Configuration dictionary
        device: Torch device
        stdp_system: STDPCoordinationSystem instance (optional, will create if None)
        
    Returns:
        coordination_loss: tensor, STDP coordination loss
        stdp_system: STDPCoordinationSystem instance (for reuse)
        coordination_stats: dict, statistics about coordination
    """
    if not config.get('use_stdp_coordination', False):
        return torch.tensor(0.0, device=device), None, {}
    
    batch_size, num_agents, action_dim = model_outputs.shape
    
    # Create STDP system if not provided
    if stdp_system is None:
        stdp_system = STDPCoordinationSystem(config).to(device)
    
    # Convert model outputs to spike activity (using softmax probabilities as spike rates)
    action_probs = F.softmax(model_outputs, dim=-1)  # [batch, agents, actions]
    
    # Detect collisions (simplified - could use more sophisticated detection)
    collisions = detect_simple_collisions(agent_positions)  # [batch, agents]
    
    # Run STDP coordination system
    coordination_output = stdp_system(
        positions=agent_positions,
        goals=goals,
        fov_observations=fov_observations,
        spike_outputs=action_probs,
        collisions=collisions
    )
    
    # Extract flow field scores and compute coordination loss
    flow_field_scores = coordination_output['flow_field_scores']  # [batch, agents, directions]
    
    # Coordination loss: encourage actions that align with flow field
    # Higher flow field score = should be more likely to take that action
    action_log_probs = F.log_softmax(model_outputs, dim=-1)  # [batch, agents, actions]
    
    # Weight log probabilities by flow field scores
    weighted_log_probs = action_log_probs * flow_field_scores
    
    # Loss encourages taking actions with positive flow field scores
    coordination_loss = -torch.mean(weighted_log_probs)
    
    # Apply loss weight
    coordination_weight = config.get('stdp_coordination_weight', 0.2)
    coordination_loss = coordination_loss * coordination_weight
    
    return coordination_loss, stdp_system, coordination_output['coordination_stats']


def detect_simple_collisions(agent_positions):
    """
    Simple collision detection based on position overlap.
    
    Args:
        agent_positions: [batch_size, num_agents, 2] agent positions
        
    Returns:
        collisions: [batch_size, num_agents] boolean collision indicators
    """
    batch_size, num_agents = agent_positions.shape[:2]
    device = agent_positions.device
    
    collisions = torch.zeros(batch_size, num_agents, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        positions = agent_positions[b]  # [num_agents, 2]
        
        # Check for position overlaps
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                pos_i = positions[i]
                pos_j = positions[j]
                distance = torch.norm(pos_i - pos_j)
                
                if distance < 0.5:  # Same grid cell
                    collisions[b, i] = True
                    collisions[b, j] = True
    
    return collisions


def compute_simple_oscillation_penalty(current_positions, batch_id=0, oscillation_weight=10.0, goal_positions=None):
    """
    Simple oscillation detection: Check if first 2 positions match last 2 positions
    DEPRECATED: Use compute_enhanced_oscillation_penalty for better detection
    Args:
        current_positions: [batch_size, num_agents, 2] - current agent positions
        batch_id: batch identifier for tracking
        oscillation_weight: penalty weight for oscillations
        goal_positions: [batch_size, num_agents, 2] goal positions (optional, to exclude agents at goal)
    Returns:
        oscillation_penalty: scalar tensor with penalty
    """
    global _position_history
    
    batch_size, num_agents = current_positions.shape[:2]
    total_penalty = torch.tensor(0.0, device=current_positions.device, dtype=torch.float32)
    
    # Get agents that have reached their goals (if goal_positions provided)
    agents_at_goal = None
    if goal_positions is not None:
        agents_at_goal = get_agents_at_goal(current_positions, goal_positions)
    
    for b in range(batch_size):
        for a in range(num_agents):
            # Skip agents that have reached their goals
            if agents_at_goal is not None and agents_at_goal[b, a]:
                continue  # Agent has reached goal, no oscillation penalty
                
            agent_key = f"batch_{batch_id}_agent_{a}"
            current_pos = current_positions[b, a].clone().detach()
            
            # Convert to integers for exact comparison (since agents move +-1 on grid)
            current_pos_int = current_pos.round().int()
            
            # Add current position to history
            _position_history[agent_key].append(current_pos_int)
            
            # Check for oscillation if we have 4 positions
            if len(_position_history[agent_key]) == 4:
                positions = list(_position_history[agent_key])
                
                # Skip oscillation penalty if agent is at rest (not moving)
                # Check if the last two positions are the same (agent is staying put)
                if torch.equal(positions[-1], positions[-2]):
                    continue  # Agent is at rest, no oscillation penalty
                
                # Check if first two positions match last two positions
                # positions[0] == positions[2] AND positions[1] == positions[3]
                pos_0_2_match = torch.equal(positions[0], positions[2])
                pos_1_3_match = torch.equal(positions[1], positions[3])
                
                if pos_0_2_match and pos_1_3_match:
                    # HUGE penalty for A->B->A->B oscillation
                    penalty = torch.tensor(oscillation_weight, device=current_positions.device, dtype=torch.float32)
                    total_penalty = total_penalty + penalty
    
    return total_penalty


def compute_enhanced_oscillation_penalty(current_positions, batch_id=0, oscillation_weight=1000.0, goal_positions=None):
    """
    Enhanced oscillation detection using OscillationDetector class.
    Detects various types of repetitive/cyclic patterns, not just A<->B oscillations.
    
    Args:
        current_positions: [batch_size, num_agents, 2] - current agent positions
        batch_id: batch identifier for tracking
        oscillation_weight: penalty weight for oscillations (100x increase from simple version)
        goal_positions: [batch_size, num_agents, 2] goal positions (optional, to exclude agents at goal)
    Returns:
        oscillation_penalty: scalar tensor with penalty
    """
    try:
        from oscillation_detector import compute_enhanced_oscillation_penalty
        return compute_enhanced_oscillation_penalty(current_positions, batch_id, oscillation_weight, goal_positions)
    except ImportError:
        # Fallback to simple oscillation detection if enhanced detector is not available
        return compute_simple_oscillation_penalty(current_positions, batch_id, oscillation_weight, goal_positions)


def clear_oscillation_history():
    """Clear oscillation history to prevent memory buildup"""
    global _position_history
    _position_history.clear()
    
    # Also clear enhanced oscillation detector history
    try:
        from oscillation_detector import clear_enhanced_oscillation_history
        clear_enhanced_oscillation_history()
    except ImportError:
        pass  # Enhanced detector not available


def clear_enhanced_oscillation_history():
    """Clear enhanced oscillation history (direct import)"""
    try:
        from oscillation_detector import clear_enhanced_oscillation_history
        clear_enhanced_oscillation_history()
    except ImportError:
        pass  # Enhanced detector not available


def get_agents_at_goal(agent_positions, goal_positions, tolerance=0.5):
    """
    Determine which agents have reached their goals.
    
    Args:
        agent_positions: [batch_size, num_agents, 2] current agent positions
        goal_positions: [batch_size, num_agents, 2] goal positions
        tolerance: float, distance tolerance for goal reaching
        
    Returns:
        agents_at_goal: [batch_size, num_agents] bool tensor indicating which agents reached goals
    """
    # Calculate distances to goals
    distances = torch.norm(agent_positions - goal_positions, dim=-1)  # [batch_size, num_agents]
    
    # Agents are at goal if distance is within tolerance
    agents_at_goal = distances < tolerance
    
    return agents_at_goal

# Global tracking for consecutive stay actions
from collections import defaultdict, deque
_stay_history = defaultdict(lambda: deque(maxlen=10))  # Track last 10 actions per agent

def compute_consecutive_stay_penalty(predicted_actions, batch_id=0, stay_penalty_weight=5.0, max_consecutive_stays=3):
    """
    Penalize agents for staying in place for more than max_consecutive_stays consecutive actions.
    
    Args:
        predicted_actions: [batch, agents, action_dim] - predicted actions (or action indices)
        batch_id: batch identifier for tracking
        stay_penalty_weight: penalty weight for excessive staying
        max_consecutive_stays: maximum allowed consecutive stay actions
    
    Returns:
        stay_penalty: scalar tensor with penalty
    """
    global _stay_history
    
    batch_size, num_agents = predicted_actions.shape[:2]
    total_penalty = torch.tensor(0.0, device=predicted_actions.device, dtype=torch.float32)
    
    for b in range(batch_size):
        for a in range(num_agents):
            agent_key = f"batch_{batch_id}_agent_{a}"
            
            # Get predicted action (convert to index if needed)
            if predicted_actions.dim() == 3 and predicted_actions.shape[2] > 1:
                # Action probabilities - get argmax
                action_idx = torch.argmax(predicted_actions[b, a]).item()
            else:
                # Already action indices
                action_idx = predicted_actions[b, a].item()
            
            # Add current action to history
            _stay_history[agent_key].append(action_idx)
            
            # Check for consecutive stays (action 0 = stay)
            if len(_stay_history[agent_key]) > max_consecutive_stays:
                recent_actions = list(_stay_history[agent_key])[-max_consecutive_stays-1:]
                
                # Check if all recent actions are "stay" (action 0)
                if all(action == 0 for action in recent_actions):
                    # Penalty for excessive staying
                    penalty = torch.tensor(stay_penalty_weight, device=predicted_actions.device, dtype=torch.float32)
                    total_penalty = total_penalty + penalty
    
    return total_penalty

def clear_stay_history():
    """Clear stay action history to prevent memory buildup"""
    global _stay_history
    _stay_history.clear()


def compute_intent_field_attraction_reward(model_outputs, agent_positions, goals, config, device):
    """
    Compute intent field attraction reward - encourage agents to move towards their goals.
    
    Args:
        model_outputs: [batch, agents, actions] - model action predictions
        agent_positions: [batch, agents, 2] - current agent positions  
        goals: [batch, agents, 2] - goal positions
        config: configuration dictionary
        device: torch device
        
    Returns:
        intent_attraction_reward: scalar tensor with reward value
    """
    intent_attraction_reward = torch.tensor(0.0, device=device)
    
    if not config.get('use_intent_field_attraction', True):
        return intent_attraction_reward
        
    intent_attraction_weight = config.get('intent_field_attraction_weight', 0.1)
    batch_size, num_agents = agent_positions.shape[:2]
    
    # Action deltas for movement calculation
    action_deltas = torch.tensor([
        [0, 0],   # 0: Stay
        [1, 0],   # 1: Right
        [0, 1],   # 2: Up  
        [-1, 0],  # 3: Left
        [0, -1],  # 4: Down
    ], device=device, dtype=torch.float32)
    
    for b in range(batch_size):
        for a in range(num_agents):
            agent_pos = agent_positions[b, a]
            goal_pos = goals[b, a]
            
            # Skip if agent is already at goal
            if torch.norm(agent_pos - goal_pos) <= 1.0:
                continue
            
            # Get predicted action
            if model_outputs.dim() == 3 and model_outputs.shape[2] > 1:
                predicted_action = torch.argmax(model_outputs[b, a])
            else:
                predicted_action = model_outputs[b, a].int()
            
            # Calculate movement direction
            action_delta = action_deltas[predicted_action]
            
            # Calculate goal direction
            goal_direction = goal_pos - agent_pos
            goal_direction_norm = goal_direction / (torch.norm(goal_direction) + 1e-8)
            
            # Reward if movement is towards goal
            if predicted_action > 0:  # Not staying
                action_direction_norm = action_delta / (torch.norm(action_delta) + 1e-8)
                
                # Dot product to measure alignment with goal direction
                alignment = torch.dot(action_direction_norm, goal_direction_norm)
                
                # Reward positive alignment (moving towards goal)
                if alignment > 0.1:  # Some tolerance
                    reward = intent_attraction_weight * alignment
                    intent_attraction_reward += reward
    
    return intent_attraction_reward


def compute_incremental_rewards(current_positions, previous_positions, goals, collision_masks=None, device=None):
    """
    Compute incremental rewards for useful movement dynamics.
    
    Args:
        current_positions: [batch, agents, 2] current positions
        previous_positions: [batch, agents, 2] previous positions 
        goals: [batch, agents, 2] goal positions
        collision_masks: [batch, agents] collision indicators (optional)
        device: torch device
        
    Returns:
        reward_dict: dict containing individual reward components
    """
    if device is None:
        device = current_positions.device
    
    batch_size, num_agents, _ = current_positions.shape
    
    # 1. Goal Approach Reward - reward for getting closer to goal
    prev_distances = torch.norm(previous_positions - goals, dim=-1)  # [batch, agents]
    curr_distances = torch.norm(current_positions - goals, dim=-1)  # [batch, agents]
    distance_improvement = prev_distances - curr_distances  # Positive = closer to goal
    approach_reward = torch.mean(torch.clamp(distance_improvement, min=0.0))  # Only reward improvements
    
    # 2. Collision Avoidance Reward - reward for not colliding
    if collision_masks is not None:
        # Reward agents that didn't collide
        no_collision_mask = ~collision_masks.bool()  # [batch, agents]
        collision_avoidance_reward = torch.mean(no_collision_mask.float())
    else:
        collision_avoidance_reward = torch.tensor(1.0, device=device)  # Assume no collisions if not provided
    
    # 3. Grid Position Change Reward - reward for actual movement
    position_change = torch.norm(current_positions - previous_positions, dim=-1)  # [batch, agents]
    movement_reward = torch.mean(torch.clamp(position_change, min=0.0, max=1.0))  # Reward meaningful movement
    
    # 4. Moving Away From Previous Spot Reward - prevent getting stuck
    # This is similar to position change but specifically penalizes staying in exact same spot
    stayed_in_place = (position_change < 0.1).float()  # [batch, agents] - 1 if stayed, 0 if moved
    movement_away_reward = torch.mean(1.0 - stayed_in_place)  # Higher reward for moving away
    
    reward_dict = {
        'goal_approach': approach_reward,
        'collision_avoidance': collision_avoidance_reward, 
        'grid_movement': movement_reward,
        'movement_away': movement_away_reward,
        'total_incremental': approach_reward + collision_avoidance_reward + movement_reward + movement_away_reward
    }
    
    return reward_dict


def compute_spike_sparsity_loss(spike_outputs, sparsity_weight=5.0, config=None):
    """
    Compute spike sparsity regularization to maintain healthy SNN dynamics.
    
    Args:
        spike_outputs: [batch, agents, features] or [batch, features] spike activity
        sparsity_weight: float, weight for sparsity regularization
        config: configuration dictionary with sparsity parameters
        
    Returns:
        sparsity_loss: tensor, regularization loss
        sparsity_stats: dict, statistics about spike activity
    """
    if spike_outputs is None:
        return torch.tensor(0.0), {'avg_spike_rate': 0.0, 'active_neurons': 0, 'within_tolerance': False}
    
    # Get sparsity configuration - use new parameter names
    target_spike_rate = config.get('target_spike_rate', 0.1) if config else 0.1  # Target 10% firing rate
    sparsity_tolerance = config.get('spike_rate_tolerance', 0.05) if config else 0.05  # ±5% tolerance
    max_spike_rate = config.get('max_spike_rate', 0.3) if config else 0.3  # Maximum allowed spike rate
    min_spike_rate = config.get('min_spike_rate', 0.01) if config else 0.01  # Minimum to prevent dead neurons
    
    # Flatten spike outputs to compute overall activity
    if spike_outputs.dim() > 2:
        spike_flat = spike_outputs.view(-1)
    else:
        spike_flat = spike_outputs.flatten()
    
    # Compute mean spike activity (should be between 0 and 1)
    mean_spike_activity = torch.mean(torch.abs(spike_flat))
    
    # Enhanced target-based sparsity loss with boundaries
    sparsity_loss = torch.tensor(0.0, device=spike_flat.device)
    
    if mean_spike_activity > max_spike_rate:
        # Heavy penalty for exceeding maximum rate
        excess = mean_spike_activity - max_spike_rate
        sparsity_loss += sparsity_weight * 2.0 * excess  # Double penalty for exceeding max
    elif mean_spike_activity < min_spike_rate:
        # Penalty for neurons being too inactive (dead neuron syndrome)
        deficit = min_spike_rate - mean_spike_activity
        sparsity_loss += sparsity_weight * 0.5 * deficit  # Lighter penalty for being too quiet
    else:
        # Within acceptable range - only penalize if outside tolerance of target
        deviation = torch.abs(mean_spike_activity - target_spike_rate)
        if deviation > sparsity_tolerance:
            sparsity_loss += sparsity_weight * (deviation - sparsity_tolerance)
    
    # Statistics with health indicators
    active_threshold = 0.1  # Consider neurons active if spike > 10%
    active_neurons = torch.sum(torch.abs(spike_flat) > active_threshold).item()
    total_neurons = spike_flat.numel()
    
    # Health indicators
    within_tolerance = abs(mean_spike_activity.item() - target_spike_rate) <= sparsity_tolerance
    within_bounds = min_spike_rate <= mean_spike_activity.item() <= max_spike_rate
    health_score = 1.0 if (within_tolerance and within_bounds) else 0.0
    
    sparsity_stats = {
        'avg_spike_rate': mean_spike_activity.item(),
        'active_neurons': active_neurons,
        'total_neurons': total_neurons,
        'sparsity_ratio': 1.0 - (active_neurons / total_neurons) if total_neurons > 0 else 1.0,
        'target_spike_rate': target_spike_rate,
        'deviation_from_target': abs(mean_spike_activity.item() - target_spike_rate),
        'within_tolerance': within_tolerance,
        'within_bounds': within_bounds,
        'health_score': health_score,
        'max_spike_rate': max_spike_rate,
        'min_spike_rate': min_spike_rate
    }
    
    return sparsity_loss, sparsity_stats

def compute_progress_shaping_reward(current_positions: torch.Tensor,
                                   previous_positions: torch.Tensor,
                                   goal_positions: torch.Tensor,
                                   progress_weight: float = 1.0) -> torch.Tensor:
    """
    Compute progress shaping reward based on change in distance to goal.
    Reward = -Δ(distance-to-goal) per step for continuous learning signal.
    
    Args:
        current_positions: Current agent positions [batch, num_agents, 2]
        previous_positions: Previous agent positions [batch, num_agents, 2]
        goal_positions: Goal positions [batch, num_agents, 2]
        progress_weight: Weight for progress reward
        
    Returns:
        progress_reward: Reward per agent [batch, num_agents]
    """
    # Calculate distances to goals
    current_distances = torch.norm(current_positions - goal_positions, dim=-1)  # [batch, num_agents]
    previous_distances = torch.norm(previous_positions - goal_positions, dim=-1)  # [batch, num_agents]
    
    # Progress reward = negative change in distance (reward for getting closer)
    distance_change = current_distances - previous_distances
    progress_reward = -distance_change * progress_weight
    
    return progress_reward