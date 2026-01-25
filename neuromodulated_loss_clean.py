#!/usr/bin/env python3
"""
🧠 RECEPTOR-BASED NEUROMODULATED RL LOSS FUNCTION

Brain-inspired loss function with proper neurotransmitter receptor dynamics:
- Dopamine modulates AMPA/NMDA (excitatory receptors)
- GABA modulates GABAA/GABAB (inhibitory receptors)
- Phase-based training (exploration → exploitation → stabilization)
- Easily configurable parameters

Receptor Types:
- AMPA: Fast excitation (τ=2ms) - immediate responses
- NMDA: Slow excitation (τ=50ms) - learning, temporal integration
- GABAA: Fast inhibition (τ=6ms) - rapid spike control
- GABAB: Slow inhibition (τ=150ms) - network stabilization

Neuromodulation:
- High dopamine → Stronger AMPA/NMDA → More excitation → Exploration
- High GABA → Stronger GABAA/GABAB → More inhibition → Stability
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NeuromodulationConfig:
    """Configuration for neuromodulation parameters - easily adjustable"""
    
    # === REWARD/PUNISHMENT PARAMETERS ===
    goal_reward: float = 10.0
    collision_punishment: float = 8.0  # Will be made negative internally
    step_penalty: float = 0.5          # Will be made negative internally
    cooperation_bonus: float = 3.0
    
    # === NEUROMODULATION BASELINES ===
    dopamine_baseline: float = 0.5
    gaba_baseline: float = 0.5
    reward_scale: float = 0.2
    punishment_scale: float = 0.2
    
    # === PHASE-BASED ADJUSTMENTS ===
    # Exploration phase (first 1000 steps)
    exploration_dopamine_boost: float = 5.0     # INCREASED from 3.0 to prevent neuron death
    exploration_gaba_reduction: float = -3.0    # INCREASED from -2.5 to prevent neuron death
    exploration_reward_scale: float = 3.0       # INCREASED from 2.5
    exploration_punishment_scale: float = 0.05  # DECREASED from 0.1
    
    # Exploitation phase (1000-5000 steps)
    exploitation_dopamine_boost: float = 0.0
    exploitation_gaba_reduction: float = 0.0
    exploitation_reward_scale: float = 1.0
    exploitation_punishment_scale: float = 1.0
    
    # Stabilization phase (5000+ steps)
    stabilization_dopamine_boost: float = -0.1
    stabilization_gaba_reduction: float = 0.2
    stabilization_reward_scale: float = 0.9
    stabilization_punishment_scale: float = 1.1
    
    # === TRAINING PHASE THRESHOLDS ===
    exploration_threshold: int = 1000
    exploitation_threshold: int = 5000
    
    # === TEMPORAL DYNAMICS ===
    trace_length: int = 10
    smoothing_factor: float = 0.7
    progress_shaping_weight: float = 5.0


class NeuromodulatedRLLoss(nn.Module):
    """
    🧠 Clean, modular brain-inspired loss function
    """
    
    def __init__(self, config: Optional[NeuromodulationConfig] = None):
        super().__init__()
        self.config = config or NeuromodulationConfig()
        
        # State tracking
        self.step_count = 0
        self._prev_positions = None
        self._prev_distances = None
        self._prev_goal_distance = None
        
        # History buffers for temporal smoothing
        self.register_buffer('dopamine_history', 
                           torch.ones(self.config.trace_length) * self.config.dopamine_baseline)
        self.register_buffer('gaba_history', 
                           torch.ones(self.config.trace_length) * self.config.gaba_baseline)
    
    def reset_state(self):
        """Reset internal state for new training session"""
        self.step_count = 0
        self._prev_positions = None
        self._prev_distances = None
        self._prev_goal_distance = None
        self.dopamine_history.fill_(self.config.dopamine_baseline)
        self.gaba_history.fill_(self.config.gaba_baseline)
    
    def get_training_phase(self) -> str:
        """Determine current training phase based on step count"""
        if self.step_count < self.config.exploration_threshold:
            return "exploration"
        elif self.step_count < self.config.exploitation_threshold:
            return "exploitation"
        else:
            return "stabilization"
    
    def get_phase_adjustments(self) -> Dict[str, float]:
        """Get phase-specific neuromodulator adjustments"""
        phase = self.get_training_phase()
        
        if phase == "exploration":
            return {
                'dopamine_boost': self.config.exploration_dopamine_boost,
                'gaba_reduction': self.config.exploration_gaba_reduction,
                'reward_scale': self.config.exploration_reward_scale,
                'punishment_scale': self.config.exploration_punishment_scale
            }
        elif phase == "exploitation":
            return {
                'dopamine_boost': self.config.exploitation_dopamine_boost,
                'gaba_reduction': self.config.exploitation_gaba_reduction,
                'reward_scale': self.config.exploitation_reward_scale,
                'punishment_scale': self.config.exploitation_punishment_scale
            }
        else:  # stabilization
            return {
                'dopamine_boost': self.config.stabilization_dopamine_boost,
                'gaba_reduction': self.config.stabilization_gaba_reduction,
                'reward_scale': self.config.stabilization_reward_scale,
                'punishment_scale': self.config.stabilization_punishment_scale
            }
    
    def compute_rewards_and_punishments(self, 
                                      positions: torch.Tensor,
                                      goals: torch.Tensor,
                                      actions: torch.Tensor,
                                      collisions: Optional[torch.Tensor] = None,
                                      goal_reached: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute RL rewards and punishments"""
        
        batch_size, num_agents = positions.shape[0], positions.shape[1]
        device = positions.device
        
        # === GOAL PROGRESS REWARDS ===
        if goal_reached is not None:
            goal_rewards = goal_reached.float() * self.config.goal_reward
        else:
            distances_to_goal = torch.norm(positions - goals, dim=-1)
            max_distance = torch.sqrt(torch.tensor(2.0, device=device))
            progress_reward = (max_distance - distances_to_goal) / max_distance * self.config.goal_reward * 0.1
            goal_rewards = progress_reward
        
        # === PROGRESS SHAPING (dense reward signal) ===
        if self._prev_positions is not None:
            prev_dist = torch.norm(self._prev_positions - goals, dim=-1)
            curr_dist = torch.norm(positions - goals, dim=-1)
            progress_shaping = (prev_dist - curr_dist) * self.config.progress_shaping_weight
            goal_rewards += progress_shaping
        
        self._prev_positions = positions.detach().clone()
        
        # === COLLISION PUNISHMENTS ===
        if collisions is not None:
            collision_punishments = collisions.float() * self.config.collision_punishment
        else:
            collision_punishments = torch.zeros_like(goal_rewards)
        
        # === COOPERATION REWARDS ===
        cooperation_rewards = torch.zeros_like(goal_rewards)
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                goal_similarity = torch.cosine_similarity(
                    goals[:, i] - positions[:, i],
                    goals[:, j] - positions[:, j],
                    dim=-1
                )
                cooperation = (goal_similarity > 0.5).float() * self.config.cooperation_bonus * 0.1
                cooperation_rewards[:, i] += cooperation
                cooperation_rewards[:, j] += cooperation
        
        # === PHASE-BASED PENALTIES ===
        phase = self.get_training_phase()
        
        if phase == "exploration":
            # Gentle opportunity cost penalty during exploration
            current_distances = torch.norm(positions - goals, dim=-1)
            if self._prev_distances is not None:
                optimal_progress = 1.0
                actual_progress = torch.clamp(self._prev_distances - current_distances, min=-2.0, max=2.0)
                missed_opportunity = torch.clamp(optimal_progress - actual_progress, min=0.0, max=1.5)
                step_penalties = -missed_opportunity * 0.3
            else:
                step_penalties = torch.zeros_like(goal_rewards)
            self._prev_distances = current_distances.detach()
            
            stay_penalties = torch.zeros_like(goal_rewards)
            inefficiency_penalty = torch.zeros_like(goal_rewards)
            
        else:
            # Harsh penalties during exploitation/stabilization
            distances_to_goal = torch.norm(positions - goals, dim=-1)
            max_distance = torch.sqrt(torch.tensor(8.0, device=device))
            step_penalties = -(distances_to_goal / max_distance * self.config.step_penalty)
            
            stay_actions = (actions == 0).float() if actions is not None else torch.zeros_like(goal_rewards)
            stay_penalties = stay_actions * (-2.0)
            
            goal_direction = torch.norm(goals - positions, dim=-1)
            if self._prev_goal_distance is not None:
                distance_change = goal_direction - self._prev_goal_distance
                inefficiency_penalty = torch.clamp(distance_change * 1.0, min=0.0, max=2.0)
            else:
                inefficiency_penalty = torch.zeros_like(goal_direction)
            self._prev_goal_distance = goal_direction.detach()
        
        # === COMBINE REWARDS AND PUNISHMENTS ===
        total_reward = goal_rewards + cooperation_rewards
        
        # Convert all penalties to positive punishment values for clarity
        total_punishment = (
            collision_punishments +
            torch.clamp(-step_penalties, min=0.0) +
            torch.clamp(inefficiency_penalty, min=0.0) +
            torch.clamp(-stay_penalties, min=0.0)
        )
        
        return {
            'total_reward': total_reward,
            'total_punishment': total_punishment,
            'goal_rewards': goal_rewards,
            'collision_punishments': collision_punishments,
            'cooperation_rewards': cooperation_rewards
        }
    
    def compute_neuromodulators(self, total_reward: torch.Tensor, total_punishment: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute dopamine and GABA levels based on rewards/punishments"""
        
        # Get phase-specific adjustments
        phase_adjustments = self.get_phase_adjustments()
        
        # Scale inputs based on phase
        adjusted_reward_scale = self.config.reward_scale * phase_adjustments['reward_scale']
        adjusted_punishment_scale = self.config.punishment_scale * phase_adjustments['punishment_scale']
        
        # Base neuromodulator levels (with phase adjustments)
        base_dopamine = self.config.dopamine_baseline + phase_adjustments['dopamine_boost']
        base_gaba = self.config.gaba_baseline + phase_adjustments['gaba_reduction']
        
        # Compute raw levels
        reward_input = torch.clamp(total_reward * adjusted_reward_scale, min=-10.0, max=10.0)
        punishment_input = total_punishment * adjusted_punishment_scale
        
        raw_dopamine = base_dopamine + 2.0 * torch.sigmoid(reward_input)
        raw_gaba = base_gaba + 2.0 * torch.sigmoid(punishment_input)
        
        # Temporal smoothing
        trace_idx = self.step_count % self.config.trace_length
        self.dopamine_history[trace_idx] = torch.clamp(raw_dopamine.mean(), min=0.1, max=3.0)  # Increased max from 2.1 to 3.0
        self.gaba_history[trace_idx] = torch.clamp(raw_gaba.mean(), min=0.01, max=2.1)  # Decreased min from 0.1 to 0.01
        
        dopamine_smooth = self.dopamine_history.mean()
        gaba_smooth = self.gaba_history.mean()
        
        # Blend raw and smoothed values
        dopamine_levels = torch.clamp(
            self.config.smoothing_factor * dopamine_smooth + (1 - self.config.smoothing_factor) * raw_dopamine,
            min=0.1, max=3.0
        )
        gaba_levels = torch.clamp(
            self.config.smoothing_factor * gaba_smooth + (1 - self.config.smoothing_factor) * raw_gaba,
            min=0.01, max=2.1
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
        
        # GABA regularization
        activity_level = torch.mean(torch.abs(model_outputs), dim=-1)
        gaba_regularization = (gaba_levels - 0.5) * activity_level * 0.1
        
        # Combine losses
        total_loss = dopamine_modulated_loss + torch.abs(gaba_regularization) * 0.1
        
        return total_loss
    
    def forward(self,
               model_outputs: torch.Tensor,
               target_actions: torch.Tensor,
               positions: torch.Tensor,
               goals: torch.Tensor,
               collisions: Optional[torch.Tensor] = None,
               goal_reached: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Main forward pass"""
        
        # Increment step counter
        self.step_count += 1
        
        # Compute rewards and punishments
        reward_data = self.compute_rewards_and_punishments(
            positions, goals, target_actions, collisions, goal_reached
        )
        
        # Compute neuromodulators
        dopamine, gaba = self.compute_neuromodulators(
            reward_data['total_reward'], reward_data['total_punishment']
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


# === FACTORY FUNCTIONS ===

def create_aggressive_exploration_config() -> NeuromodulationConfig:
    """Create config optimized for keeping neurons alive during exploration"""
    return NeuromodulationConfig(
        # Stronger neuromodulation for exploration
        exploration_dopamine_boost=6.0,        # Very high dopamine
        exploration_gaba_reduction=-3.5,       # Very low GABA
        exploration_reward_scale=4.0,          # Strong reward sensitivity
        exploration_punishment_scale=0.02,     # Minimal punishment sensitivity
        
        # Standard values for other phases
        goal_reward=15.0,                      # Higher goal rewards
        collision_punishment=5.0,              # Lower collision punishment
        progress_shaping_weight=8.0,           # Stronger progress shaping
        
        # Longer exploration phase
        exploration_threshold=1500,            # Extended exploration
        exploitation_threshold=6000,
    )

def create_balanced_config() -> NeuromodulationConfig:
    """Create balanced config for stable training"""
    return NeuromodulationConfig()  # Use defaults

def create_fast_convergence_config() -> NeuromodulationConfig:
    """Create config for faster convergence"""
    return NeuromodulationConfig(
        exploration_threshold=500,             # Shorter exploration
        exploitation_threshold=3000,          # Shorter exploitation
        exploration_dopamine_boost=4.0,       # Moderate dopamine boost
        exploration_punishment_scale=0.1,     # Higher punishment sensitivity
    )


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
