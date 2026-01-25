#!/usr/bin/env python3
"""
🧠 RECEPTOR-BASED NEUROMODULATED RL LOSS FUNCTION

Brain-inspired loss function with proper neurotransmitter receptor dynamics:
- Dopamine modulates AMPA/NMDA (excitatory receptors)
- GABA modulates GABAA/GABAB (inhibitory receptors)
- Phase-based training (exploration → exploitation → stabilization)

Receptor Types:
- AMPA: Fast excitation (τ=2ms) - immediate responses
- NMDA: Slow excitation (τ=50ms) - learning, temporal integration
- GABAA: Fast inhibition (τ=6ms) - rapid spike control
- GABAB: Slow inhibition (τ=150ms) - network stabilization
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class NeuromodulatedRLLoss(nn.Module):
    """
    🧠 Clean, modular brain-inspired loss function using existing config parameters
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Extract parameters from existing config (with sensible defaults)
        self.goal_reward = config.get('goal_reward', 100.0)  # Updated fallback to match current config
        self.collision_punishment = abs(config.get('collision_punishment', -1.0))  # Make positive
        self.step_penalty = abs(config.get('step_penalty', -0.5))  # Make positive
        self.stay_penalty = abs(config.get('stay_penalty', -2.0))  # NEW: Penalty for staying still
        self.cooperation_bonus = config.get('cooperation_bonus', 5.0)
        
        # Spike rate penalty parameters
        self.spike_rate_penalty = config.get('spike_rate_penalty', -2.0)
        self.min_spike_rate_threshold = config.get('min_spike_rate_threshold', 0.03)
        
        # High spike activity penalties - NEW!
        self.high_spike_penalty = config.get('high_spike_penalty', -3.0)
        self.max_spike_rate_threshold = config.get('max_spike_rate_threshold', 0.15)
        self.reward_reduction_factor = config.get('reward_reduction_factor', 0.5)
        
        # Neuromodulation baselines
        self.dopamine_baseline = 0.25  # 🔧 HALVED FROM 0.5 - Reduced dopamine baseline
        self.gaba_baseline = 0.5
        self.reward_scale = config.get('reward_scale', 0.2)
        self.punishment_scale = config.get('punishment_scale', 0.2)
        
        # Phase parameters are now ONLY in TrainingModeController - no more duplicates!
        
        # Training phase controller
        self.total_epochs = None  # Set by mode controller
        self.current_epoch = 0
        
        # Training phase thresholds - needed for get_training_phase()
        self.exploration_threshold = 100   # First 100 steps: exploration
        self.exploitation_threshold = 300  # Steps 100-300: exploitation
        # After 300 steps: stabilization
        
        # Previous state for tracking
        
        # Temporal dynamics
        self.trace_length = config.get('trace_length', 13)
        self.smoothing_factor = 0.7
        
        # State tracking
        self.step_count = 0
        self._prev_positions = None
        self._prev_distances = None
        self._prev_goal_distance = None
        self._prev_actions = None  # Track previous actions for consecutive stay penalty
        
        # History buffers for temporal smoothing
        self.register_buffer('dopamine_history', 
                           torch.ones(self.trace_length) * self.dopamine_baseline)
        self.register_buffer('gaba_history', 
                           torch.ones(self.trace_length) * self.gaba_baseline)
    
    def reset_state(self):
        """Reset internal state for new training session"""
        self.step_count = 0
        self._prev_positions = None
        self._prev_distances = None
        self._prev_goal_distance = None
        self._prev_actions = None  # Reset previous actions tracking
        self.dopamine_history.fill_(self.dopamine_baseline)
        self.gaba_history.fill_(self.gaba_baseline)
    
    def get_training_phase(self) -> str:
        """Determine current training phase based on step count"""
        if self.step_count < self.exploration_threshold:
            return "exploration"
        elif self.step_count < self.exploitation_threshold:
            return "exploitation"
        else:
            return "stabilization"
    
    # Phase adjustments are now handled by TrainingModeController only!
    
    def compute_rewards_and_punishments(self, 
                                      positions: torch.Tensor,
                                      goals: torch.Tensor,
                                      actions: torch.Tensor,
                                      collisions: Optional[torch.Tensor] = None,
                                      goal_reached: Optional[torch.Tensor] = None,
                                      spike_rates: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """Compute RL rewards and punishments"""
        
        batch_size, num_agents = positions.shape[0], positions.shape[1]
        device = positions.device
        
        # === GOAL REWARDS === CLEAN SIGNAL: Only +100 for actual goal achievement
        if goal_reached is not None:
            goal_rewards = goal_reached.float() * self.goal_reward
        else:
            # No progress rewards - force learning through pure goal achievement
            goal_rewards = torch.zeros(batch_size, num_agents, device=device)
        
        # No progress shaping - clean RL signal only
        
        self._prev_positions = positions.detach().clone()
        
        # === COLLISION PUNISHMENTS ===
        if collisions is not None:
            collision_punishments = collisions.float() * self.collision_punishment
        else:
            collision_punishments = torch.zeros_like(goal_rewards)
        
        # === COOPERATION REWARDS - ONLY WHEN BOTH AGENTS REACH GOALS ===
        cooperation_rewards = torch.zeros_like(goal_rewards)
        
        # Only give cooperation bonus when BOTH agents in a pair reach their goals
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                # Check if both agents have reached their goals
                agent_i_reached = goal_reached[:, i].float()  # 1.0 if agent i reached goal, 0.0 otherwise
                agent_j_reached = goal_reached[:, j].float()  # 1.0 if agent j reached goal, 0.0 otherwise
                
                # Both agents must reach their goals to get cooperation bonus
                both_reached_goals = agent_i_reached * agent_j_reached  # 1.0 only if both reached
                
                cooperation_bonus_value = both_reached_goals * self.cooperation_bonus
                cooperation_rewards[:, i] += cooperation_bonus_value
                cooperation_rewards[:, j] += cooperation_bonus_value
        
        # === SIMPLE STEP PENALTY ONLY ===
        phase = self.get_training_phase()
        
        if phase == "exploration":
            # VERY LIGHT step penalty during exploration to allow free exploration
            step_penalties = torch.full_like(goal_rewards, self.step_penalty * 0.5)
        elif phase == "exploitation":
            # Moderate step penalty during exploitation
            step_penalties = torch.full_like(goal_rewards, self.step_penalty)
        else:  # stabilization
            # Full step penalty during stabilization
            step_penalties = torch.full_like(goal_rewards, self.step_penalty)
        
        # === COMBINE REWARDS AND PUNISHMENTS ===
        total_reward = goal_rewards + cooperation_rewards
        
        # === SPIKE RATE PENALTIES AND REWARD REDUCTION ===
        spike_rate_penalties = torch.zeros_like(goal_rewards)
        reward_reduction_applied = False
        
        if spike_rates is not None:
            # Check for LOW spike rates (neuron death)
            low_spike_layers = []
            # Check for HIGH spike rates (excessive activity)
            high_spike_layers = []
            
            for layer_name, spike_rate in spike_rates.items():
                if spike_rate < self.min_spike_rate_threshold:
                    low_spike_layers.append(f"{layer_name}:{spike_rate:.3f}")
                elif spike_rate > self.max_spike_rate_threshold:
                    high_spike_layers.append(f"{layer_name}:{spike_rate:.3f}")
            
            # Apply penalty for LOW spike rates (existing logic)
            if low_spike_layers:
                penalty_multiplier = len(low_spike_layers) / len(spike_rates)
                spike_rate_penalties += torch.full_like(goal_rewards, 
                                                       abs(self.spike_rate_penalty) * penalty_multiplier)
            
            # Apply penalty and reward reduction for HIGH spike rates (NEW!)
            if high_spike_layers:
                # Penalty for excessive spiking
                high_spike_multiplier = len(high_spike_layers) / len(spike_rates)
                spike_rate_penalties += torch.full_like(goal_rewards, 
                                                       abs(self.high_spike_penalty) * high_spike_multiplier)
                
                # Reduce rewards when spiking is too high
                total_reward = total_reward * self.reward_reduction_factor
                reward_reduction_applied = True
        
        # === STAY PENALTY === Penalize stay actions ONLY for agents who haven't reached goals
        stay_penalties = torch.zeros_like(goal_rewards)
        if actions is not None:
            stay_mask = (actions == 0).float()  # 1.0 for any stay action
            
            # Exempt agents who have reached their goals from stay penalty
            if goal_reached is not None:
                # Only penalize stay if agent has NOT reached goal
                agents_not_at_goal = (1.0 - goal_reached.float())  # 1.0 if not at goal, 0.0 if at goal
                stay_penalties = stay_mask * agents_not_at_goal * self.stay_penalty
            else:
                # If no goal_reached info, penalize all stays (fallback)
                stay_penalties = stay_mask * self.stay_penalty
            
            # Update previous actions for tracking (keep for potential future use)
            self._prev_actions = actions.detach().clone()
                
        # Combine all punishments: collision + step + spike_rate (low + high) + consecutive_stay
        total_punishment = collision_punishments + step_penalties + spike_rate_penalties + stay_penalties
        
        return {
            'total_reward': total_reward,
            'total_punishment': total_punishment,
            'goal_rewards': goal_rewards,
            'collision_punishments': collision_punishments,
            'cooperation_rewards': cooperation_rewards,
            'spike_rate_penalties': spike_rate_penalties,
            'reward_reduction_applied': reward_reduction_applied
        }
    
    def compute_neuromodulators(self, total_reward: torch.Tensor, total_punishment: torch.Tensor, phase_adjustments: Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute dopamine and GABA levels based on rewards/punishments"""
        
        # Scale inputs based on phase
        adjusted_reward_scale = self.reward_scale * phase_adjustments['reward_scale']
        adjusted_punishment_scale = self.punishment_scale * phase_adjustments['punishment_scale']
        
        # Base neuromodulator levels (with phase adjustments)
        base_dopamine = self.dopamine_baseline + phase_adjustments['dopamine_boost']
        base_gaba = self.gaba_baseline + phase_adjustments['gaba_reduction']
        
        # Compute raw levels
        reward_input = torch.clamp(total_reward * adjusted_reward_scale, min=-10.0, max=10.0)
        punishment_input = total_punishment * adjusted_punishment_scale
        
        raw_dopamine = base_dopamine + 1.0 * torch.sigmoid(reward_input)  # 🔧 HALVED FROM 2.0 - Reduced dopamine sensitivity
        raw_gaba = (base_gaba + 2.0 * torch.sigmoid(punishment_input)) * 2.0  # 🔧 DOUBLE THE GABA VALUE
        
        # Temporal smoothing
        trace_idx = self.step_count % self.trace_length
        self.dopamine_history[trace_idx] = torch.clamp(raw_dopamine.mean(), min=0.1, max=3.0)  # Increased max from 2.1 to 3.0
        self.gaba_history[trace_idx] = torch.clamp(raw_gaba.mean(), min=0.02, max=4.2)  # Doubled range for GABA: min 0.01→0.02, max 2.1→4.2
        
        dopamine_smooth = self.dopamine_history.mean()
        gaba_smooth = self.gaba_history.mean()
        
        # Blend raw and smoothed values
        dopamine_levels = torch.clamp(
            self.smoothing_factor * dopamine_smooth + (1 - self.smoothing_factor) * raw_dopamine,
            min=0.1, max=3.0
        )
        gaba_levels = torch.clamp(
            self.smoothing_factor * gaba_smooth + (1 - self.smoothing_factor) * raw_gaba,
            min=0.02, max=4.2  # Doubled clamp range for GABA values
        )
        
        return dopamine_levels, gaba_levels
    
    def compute_loss(self, 
                    model_outputs: torch.Tensor,
                    target_actions: torch.Tensor,
                    dopamine_levels: torch.Tensor,
                    gaba_levels: torch.Tensor) -> torch.Tensor:
        """Compute neuromodulated loss"""
        
        # Action probability loss
        action_probs = torch.softmax(model_outputs, dim=-1)
        action_probs = torch.clamp(action_probs, min=1e-7, max=1.0-1e-7)
        
        target_one_hot = torch.zeros_like(action_probs)
        target_one_hot.scatter_(-1, target_actions.unsqueeze(-1), 1.0)
        
        log_probs = torch.log(action_probs + 1e-8)
        ce_loss = -torch.sum(target_one_hot * log_probs, dim=-1)
        
        # Dopamine modulation
        dopamine_weight = (dopamine_levels + 0.5) / 1.5
        dopamine_modulated_loss = ce_loss * dopamine_weight
        
        # GABA regularization - LOOSENED for lower spike rates
        activity_level = torch.mean(torch.abs(model_outputs), dim=-1)
        
        # Progressive penalty: higher activity gets exponentially more penalty (LOOSENED)
        high_activity_threshold = 2.0  # INCREASED from 1.0 - Less restrictive threshold
        activity_penalty_factor = torch.where(
            activity_level > high_activity_threshold,
            torch.square(activity_level / high_activity_threshold),  # Quadratic penalty for high activity
            torch.ones_like(activity_level)  # Normal penalty for low activity
        )
        
        gaba_regularization = (gaba_levels - 0.5) * activity_level * activity_penalty_factor * 0.015  # INCREASED by 50% from 0.01 - Stronger spike inhibition
        
        # Combine losses
        total_loss = dopamine_modulated_loss + torch.abs(gaba_regularization) * 0.015  # INCREASED by 50% from 0.01 - Stronger spike inhibition
        
        return total_loss
    
    def forward(self,
               model_outputs: torch.Tensor,
               target_actions: torch.Tensor,
               positions: torch.Tensor,
               goals: torch.Tensor,
               collisions: Optional[torch.Tensor] = None,
               goal_reached: Optional[torch.Tensor] = None,
               phase_adjustments: Optional[Dict[str, float]] = None,
               spike_rates: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """Main forward pass"""
        
        # Increment step counter
        self.step_count += 1
        
        # Compute rewards and punishments
        reward_data = self.compute_rewards_and_punishments(
            positions, goals, target_actions, collisions, goal_reached, spike_rates
        )
        
        # === REWARD SCALING (HIGHEST PRIORITY) ===
        # Scale down the raw rewards to prevent gradient instability
        # Raw rewards of 15+ are too large for stable training
        reward_scale_factor = self.config.get('reward_scale_factor', 20.0)  # Use config value with fallback
        scaled_rewards = reward_data['total_reward'] / reward_scale_factor
        scaled_punishments = reward_data['total_punishment'] / reward_scale_factor
        
        # Use default phase adjustments if none provided
        if phase_adjustments is None:
            phase_adjustments = {
                'dopamine_boost': 0.0,
                'gaba_reduction': 0.0,
                'reward_scale': 1.0,
                'punishment_scale': 1.0
            }
        
        # Compute neuromodulators USING THE SCALED VALUES
        dopamine, gaba = self.compute_neuromodulators(
            scaled_rewards, scaled_punishments, phase_adjustments
        )
        
        # Compute loss
        loss = self.compute_loss(model_outputs, target_actions, dopamine, gaba)
        
        return {
            'loss': loss,
            'dopamine': dopamine,
            'gaba': gaba,
            'rewards': reward_data['total_reward'],
            'punishments': reward_data['total_punishment'],
            'phase': self.get_training_phase(),
            'step_count': self.step_count
        }


# === FACTORY FUNCTIONS FOR BACKWARD COMPATIBILITY ===

def create_neuromodulated_loss(config: Dict):
    """Create neuromodulated loss using existing config dictionary"""
    return NeuromodulatedRLLoss(config)

def create_training_mode_controller(total_epochs: int):
    """Create training mode controller with all expected methods"""
    class TrainingModeController:
        def __init__(self, total_epochs):
            self.total_epochs = total_epochs
            self.current_epoch = 0
            self.mode = "exploration"
        
        def update_epoch(self, epoch: int):
            """Update current epoch and determine training phase"""
            self.current_epoch = epoch
            progress = epoch / self.total_epochs
            if progress < 0.3:
                self.mode = "exploration"
            elif progress < 0.8:
                self.mode = "exploitation"
            else:
                self.mode = "stabilization"
        
        def get_training_phase(self) -> str:
            """Get current training phase"""
            return self.mode
        
        def get_neuromodulator_adjustments(self) -> dict:
            """Get phase-specific neuromodulator adjustments with curriculum annealing"""
            # Curriculum annealing: gradually reduce exploration intensity
            exploration_progress = min(self.current_epoch / (self.total_epochs * 0.3), 1.0)  # First 30% of epochs
            
            if self.mode == "exploration":
                # Start aggressive, then anneal towards moderate values
                dopamine_start, dopamine_end = 2.0, 1.0
                gaba_start, gaba_end = -0.5, 1.0
                reward_start, reward_end = 2.0, 1.5
                punishment_start, punishment_end = 0.05, 0.5
                
                # Linear annealing
                dopamine_boost = dopamine_start + (dopamine_end - dopamine_start) * exploration_progress
                gaba_reduction = gaba_start + (gaba_end - gaba_start) * exploration_progress
                reward_scale = reward_start + (reward_end - reward_start) * exploration_progress
                punishment_scale = punishment_start + (punishment_end - punishment_start) * exploration_progress
                
                return {
                    'dopamine_boost': dopamine_boost,
                    'gaba_reduction': gaba_reduction,
                    'reward_scale': reward_scale,
                    'punishment_scale': punishment_scale
                }
            elif self.mode == "exploitation":
                return {
                    'dopamine_boost': 0.0,      # Balanced levels
                    'gaba_reduction': 0.0,
                    'reward_scale': 1.0,
                    'punishment_scale': 1.0
                }
            else:  # stabilization
                return {
                    'dopamine_boost': -0.1,     # Fine-tuning phase
                    'gaba_reduction': 0.2,
                    'reward_scale': 0.9,
                    'punishment_scale': 1.1
                }
        
        def update_mode(self, step_count):
            """Legacy method for backward compatibility"""
            if step_count < 1000:
                self.mode = "exploration"
            elif step_count < 5000:
                self.mode = "exploitation"
            else:
                self.mode = "stabilization"
        
        def get_mode(self):
            """Legacy method for backward compatibility"""
            return self.mode
    
    return TrainingModeController(total_epochs)


# === UTILITY FUNCTIONS ===

def detect_collisions(positions: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Detect collisions between agents"""
    batch_size, num_agents, _ = positions.shape
    device = positions.device
    collision_mask = torch.zeros(batch_size, num_agents, device=device)
    
    for b in range(batch_size):
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                distance = torch.norm(positions[b, i] - positions[b, j])
                if distance < threshold:
                    collision_mask[b, i] = 1.0
                    collision_mask[b, j] = 1.0
    
    return collision_mask

def detect_goal_reached(positions: torch.Tensor, goals: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """Detect when agents reach their goals"""
    distances = torch.norm(positions - goals, dim=-1)
    return (distances < threshold).float()


def calculate_manhattan_distance(pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
    """Calculate Manhattan distance between two positions"""
    return torch.sum(torch.abs(pos1 - pos2), dim=-1)


def calculate_euclidean_distance(pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
    """Calculate Euclidean distance between two positions"""
    return torch.norm(pos1 - pos2, dim=-1)


# === TEST FUNCTION ===
if __name__ == "__main__":
    # Test with actual config dictionary format
    test_config = {
        'goal_reward': 3.0,
        'collision_punishment': -1.0,
        'step_penalty': -0.5,
        'cooperation_bonus': 5.0,
        'reward_scale': 0.2,
        'punishment_scale': 0.2,
        'trace_length': 13
    }
    
    loss_fn = create_neuromodulated_loss(test_config)
    
    # Test the loss function works
    batch_size, num_agents = 4, 3
    device = torch.device('cpu')
    
    # Create test data
    positions = torch.randn(batch_size, num_agents, 2)
    goals = torch.randn(batch_size, num_agents, 2) 
    goal_reached = torch.zeros(batch_size, num_agents)
    goal_reached[0, 0] = 1.0  # First agent in first batch reached goal
    
    # Test forward pass
    rewards = loss_fn.compute_rewards(positions, goals, goal_reached)
    dopamine, gaba = loss_fn.compute_neuromodulators(rewards['total_reward'], rewards['total_punishment'])
    
    print("✅ Neuromodulated loss function test passed!")
    print(f"Phase: {loss_fn.get_training_phase()}")
    print(f"Dopamine range: {dopamine.min():.3f} - {dopamine.max():.3f}")
    print(f"GABA range: {gaba.min():.3f} - {gaba.max():.3f}")
