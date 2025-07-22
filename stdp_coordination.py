"""
STDP-Based Coordination System
Implements facilitation-based coordination using STDP traces, momentum vectors, and intention flow fields.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class STDPCoordinationSystem(nn.Module):
    """
    Advanced coordination system using STDP facilitation traces and momentum-based intention flow.
    """
    
    def __init__(self, config: Dict):
        super(STDPCoordinationSystem, self).__init__()
        
        self.config = config
        self.num_agents = config.get('num_agents', 5)
        self.board_size = config.get('board_size', [9, 9])[0]
        self.fov_size = 7  # 7x7 FOV
        self.num_directions = 5  # Stay, Right, Up, Left, Down
        
        # STDP parameters
        self.stdp_tau_facilitation = config.get('stdp_tau_facilitation', 50.0)
        self.stdp_tau_inhibition = config.get('stdp_tau_inhibition', 20.0)
        self.stdp_lr_facilitation = config.get('stdp_lr_facilitation', 2e-4)
        self.stdp_lr_inhibition = config.get('stdp_lr_inhibition', 1e-4)
        
        # Momentum parameters
        self.momentum_decay = config.get('momentum_decay', 0.8)
        self.momentum_alignment_threshold = config.get('momentum_alignment_threshold', 0.6)
        
        # Safety parameters
        self.collision_history_window = config.get('collision_history_window', 10)
        self.safe_region_threshold = config.get('safe_region_threshold', 0.7)
        
        # Coordination parameters
        self.coordination_radius = config.get('coordination_radius', 3.0)
        self.flow_field_strength = config.get('flow_field_strength', 0.5)
        
        # Action deltas for movement directions
        self.action_deltas = torch.tensor([
            [0, 0],   # 0: Stay
            [1, 0],   # 1: Right
            [0, 1],   # 2: Up  
            [-1, 0],  # 3: Left
            [0, -1],  # 4: Down
        ], dtype=torch.float32)
        
        # Initialize state tracking
        self.reset_state()
    
    def reset_state(self):
        """Reset all internal state tracking"""
        # STDP facilitation traces [batch, agents, board_y, board_x, directions]
        self.stdp_facilitation_traces = None
        self.stdp_inhibition_traces = None
        
        # Agent momentum vectors [batch, agents, 2] (x, y components)
        self.agent_momentum = None
        
        # Collision history [batch, agents, board_y, board_x]
        self.collision_history = None
        
        # Previous positions for momentum calculation [batch, agents, 2]
        self.prev_positions = None
        
        # Time step counter
        self.timestep = 0
    
    def update_learning_rate(self, new_base_lr: float):
        """
        Update STDP learning rates based on the main training learning rate.
        
        Args:
            new_base_lr: New base learning rate from the main optimizer
        """
        # Update facilitation learning rate (typically higher than base LR)
        facilitation_multiplier = self.config.get('stdp_facilitation_lr_multiplier', 2.0)
        self.stdp_lr_facilitation = new_base_lr * facilitation_multiplier
        
        # Update inhibition learning rate (typically half of facilitation)
        inhibition_multiplier = self.config.get('stdp_inhibition_lr_multiplier', 1.0)
        self.stdp_lr_inhibition = new_base_lr * inhibition_multiplier
        
        print(f"🧠 STDP learning rates updated - Facilitation: {self.stdp_lr_facilitation:.2e}, Inhibition: {self.stdp_lr_inhibition:.2e}")
    
    def initialize_traces(self, batch_size: int, device: torch.device):
        """Initialize all traces and state tensors"""
        # STDP traces: [batch, agents, board_y, board_x, directions]
        self.stdp_facilitation_traces = torch.zeros(
            batch_size, self.num_agents, self.board_size, self.board_size, self.num_directions,
            device=device, dtype=torch.float32
        )
        self.stdp_inhibition_traces = torch.zeros(
            batch_size, self.num_agents, self.board_size, self.board_size, self.num_directions,
            device=device, dtype=torch.float32
        )
        
        # Agent momentum: [batch, agents, 2]
        self.agent_momentum = torch.zeros(
            batch_size, self.num_agents, 2, device=device, dtype=torch.float32
        )
        
        # Collision history: [batch, agents, board_y, board_x]
        self.collision_history = torch.zeros(
            batch_size, self.num_agents, self.board_size, self.board_size,
            device=device, dtype=torch.float32
        )
        
        # Previous positions: [batch, agents, 2]
        self.prev_positions = torch.zeros(
            batch_size, self.num_agents, 2, device=device, dtype=torch.float32
        )
    
    def update_momentum(self, current_positions: torch.Tensor):
        """
        Update agent momentum vectors based on position changes.
        
        Args:
            current_positions: [batch, agents, 2] current agent positions
        """
        if self.prev_positions is None:
            self.prev_positions = current_positions.clone()
            return
        
        # Calculate position delta
        position_delta = current_positions - self.prev_positions  # [batch, agents, 2]
        
        # Update momentum with decay
        self.agent_momentum = (self.momentum_decay * self.agent_momentum + 
                              (1.0 - self.momentum_decay) * position_delta)
        
        # Update previous positions
        self.prev_positions = current_positions.clone()
    
    def compute_goal_alignment(self, positions: torch.Tensor, goals: torch.Tensor, 
                              direction_idx: int) -> torch.Tensor:
        """
        Compute how well a movement direction aligns with the goal vector.
        
        Args:
            positions: [batch, agents, 2] current positions
            goals: [batch, agents, 2] goal positions  
            direction_idx: int, movement direction index
            
        Returns:
            alignment: [batch, agents] alignment score (0-1)
        """
        device = positions.device
        action_delta = self.action_deltas[direction_idx].to(device)  # [2]
        
        # Goal vector (normalized)
        goal_vector = goals - positions  # [batch, agents, 2]
        goal_distance = torch.norm(goal_vector, dim=-1, keepdim=True) + 1e-8  # [batch, agents, 1]
        goal_vector_norm = goal_vector / goal_distance  # [batch, agents, 2]
        
        # Action vector (already unit length for grid actions)
        action_vector = action_delta.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
        
        # Compute dot product (cosine similarity)
        alignment = torch.sum(goal_vector_norm * action_vector, dim=-1)  # [batch, agents]
        
        # Convert to 0-1 range (from -1 to 1)
        alignment = (alignment + 1.0) / 2.0
        
        return alignment
    
    def compute_region_safety(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute safety of current regions based on collision history.
        
        Args:
            positions: [batch, agents, 2] current positions
            
        Returns:
            safety: [batch, agents] safety score (0-1, 1=safe)
        """
        batch_size, num_agents = positions.shape[:2]
        device = positions.device
        
        safety_scores = torch.ones(batch_size, num_agents, device=device)
        
        # Get collision history at current positions
        for b in range(batch_size):
            for a in range(num_agents):
                pos = positions[b, a].long()
                # Clamp positions to board bounds
                x = torch.clamp(pos[0], 0, self.board_size - 1)
                y = torch.clamp(pos[1], 0, self.board_size - 1)
                
                # Safety is inverse of collision history (more collisions = less safe)
                collision_rate = self.collision_history[b, a, y, x]
                safety_scores[b, a] = 1.0 - torch.clamp(collision_rate, 0.0, 1.0)
        
        return safety_scores
    
    def update_collision_history(self, positions: torch.Tensor, collisions: torch.Tensor):
        """
        Update collision history based on current collisions.
        
        Args:
            positions: [batch, agents, 2] current positions
            collisions: [batch, agents] boolean tensor indicating collisions
        """
        batch_size, num_agents = positions.shape[:2]
        
        # Decay collision history
        self.collision_history *= (1.0 - 1.0/self.collision_history_window)
        
        # Add new collision events
        for b in range(batch_size):
            for a in range(num_agents):
                if collisions[b, a]:
                    pos = positions[b, a].long()
                    # Clamp positions to board bounds
                    x = torch.clamp(pos[0], 0, self.board_size - 1)
                    y = torch.clamp(pos[1], 0, self.board_size - 1)
                    
                    # Increment collision count at this position
                    self.collision_history[b, a, y, x] += 1.0/self.collision_history_window
    
    def compute_nearby_momentum_alignment(self, positions: torch.Tensor, 
                                        agent_momentum: torch.Tensor) -> torch.Tensor:
        """
        Compute consensus momentum direction from nearby agents.
        
        Args:
            positions: [batch, agents, 2] current positions
            agent_momentum: [batch, agents, 2] agent momentum vectors
            
        Returns:
            consensus_momentum: [batch, agents, 2] weighted consensus momentum
        """
        batch_size, num_agents = positions.shape[:2]
        device = positions.device
        
        consensus_momentum = torch.zeros_like(agent_momentum)
        
        for b in range(batch_size):
            # Compute pairwise distances
            pos_b = positions[b]  # [agents, 2]
            distances = torch.cdist(pos_b, pos_b, p=2)  # [agents, agents]
            
            # Create coordination mask (agents within coordination radius)
            coord_mask = (distances <= self.coordination_radius) & (distances > 0.1)  # Exclude self
            
            for a in range(num_agents):
                nearby_agents = coord_mask[a]  # [agents]
                
                if torch.any(nearby_agents):
                    # Get momentum of nearby agents
                    nearby_momentum = agent_momentum[b, nearby_agents]  # [nearby_count, 2]
                    
                    # Weight by inverse distance
                    nearby_distances = distances[a, nearby_agents]  # [nearby_count]
                    weights = 1.0 / (nearby_distances + 1e-8)  # [nearby_count]
                    weights = weights / (torch.sum(weights) + 1e-8)  # Normalize
                    
                    # Compute weighted average momentum
                    consensus_momentum[b, a] = torch.sum(
                        nearby_momentum * weights.unsqueeze(-1), dim=0
                    )
        
        return consensus_momentum
    
    def update_stdp_traces(self, positions: torch.Tensor, goals: torch.Tensor, 
                          spike_outputs: torch.Tensor, collisions: torch.Tensor):
        """
        Update STDP facilitation and inhibition traces.
        
        Args:
            positions: [batch, agents, 2] current positions
            goals: [batch, agents, 2] goal positions
            spike_outputs: [batch, agents, directions] spike activity
            collisions: [batch, agents] collision indicators
        """
        batch_size, num_agents = positions.shape[:2]
        device = positions.device
        
        # Initialize traces if needed
        if self.stdp_facilitation_traces is None:
            self.initialize_traces(batch_size, device)
        
        # Decay existing traces
        facilitation_decay = torch.exp(-1.0 / self.stdp_tau_facilitation)
        inhibition_decay = torch.exp(-1.0 / self.stdp_tau_inhibition)
        
        self.stdp_facilitation_traces *= facilitation_decay
        self.stdp_inhibition_traces *= inhibition_decay
        
        # Update collision history
        self.update_collision_history(positions, collisions)
        
        # Compute region safety
        region_safety = self.compute_region_safety(positions)  # [batch, agents]
        
        # Update traces for each direction
        for direction_idx in range(self.num_directions):
            # Compute goal alignment for this direction
            goal_alignment = self.compute_goal_alignment(positions, goals, direction_idx)  # [batch, agents]
            
            # Extract spike activity for this direction
            direction_spikes = spike_outputs[:, :, direction_idx]  # [batch, agents]
            
            # Facilitation conditions: spike + safe region + goal aligned
            safe_condition = region_safety > self.safe_region_threshold
            aligned_condition = goal_alignment > self.momentum_alignment_threshold
            spike_condition = direction_spikes > 0.5  # Threshold for spike detection
            
            facilitation_condition = spike_condition & safe_condition & aligned_condition
            
            # Inhibition conditions: spike + unsafe region or misaligned
            unsafe_condition = region_safety <= self.safe_region_threshold
            misaligned_condition = goal_alignment <= (1.0 - self.momentum_alignment_threshold)
            
            inhibition_condition = spike_condition & (unsafe_condition | misaligned_condition)
            
            # Update traces at current positions
            for b in range(batch_size):
                for a in range(num_agents):
                    pos = positions[b, a].long()
                    # Clamp positions to board bounds
                    x = torch.clamp(pos[0], 0, self.board_size - 1)
                    y = torch.clamp(pos[1], 0, self.board_size - 1)
                    
                    if facilitation_condition[b, a]:
                        # Increase facilitation trace
                        self.stdp_facilitation_traces[b, a, y, x, direction_idx] += self.stdp_lr_facilitation
                    
                    if inhibition_condition[b, a]:
                        # Increase inhibition trace
                        self.stdp_inhibition_traces[b, a, y, x, direction_idx] += self.stdp_lr_inhibition
        
        # Clamp traces to reasonable bounds
        self.stdp_facilitation_traces = torch.clamp(self.stdp_facilitation_traces, 0.0, 2.0)
        self.stdp_inhibition_traces = torch.clamp(self.stdp_inhibition_traces, 0.0, 2.0)
    
    def add_intention_danger_zones(self, fov_observations: torch.Tensor, 
                                 positions: torch.Tensor, 
                                 agent_momentum: torch.Tensor,
                                 goals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add soft danger zones to FOV observations based on other agents' intended positions
        and goals that have been reached by other agents.
        This helps the model learn to avoid getting directly in front of other agents' paths
        and avoid goals that are already occupied.
        
        Args:
            fov_observations: [batch, agents, 2, 7, 7] original FOV
            positions: [batch, agents, 2] current agent positions
            agent_momentum: [batch, agents, 2] agent momentum vectors
            goals: [batch, agents, 2] goal positions (optional)
            
        Returns:
            enhanced_fov: [batch, agents, 3, 7, 7] FOV with added intention danger channel
        """
        batch_size, num_agents, channels, fov_h, fov_w = fov_observations.shape
        device = fov_observations.device
        
        # Create enhanced FOV with 3 channels: [obstacles, agents, intention_dangers]
        enhanced_fov = torch.zeros(batch_size, num_agents, 3, fov_h, fov_w, device=device)
        
        # Copy original channels
        enhanced_fov[:, :, :2, :, :] = fov_observations
        
        # FOV center coordinates
        center_x, center_y = fov_w // 2, fov_h // 2
        
        # For each agent, compute intention danger zones in its FOV
        for b in range(batch_size):
            for agent_id in range(num_agents):
                agent_pos = positions[b, agent_id]  # [2]
                
                # === INTENTION DANGER ZONES FROM OTHER AGENTS' MOMENTUM ===
                # For each other agent, compute their intended positions
                for other_id in range(num_agents):
                    if other_id == agent_id:
                        continue
                        
                    other_pos = positions[b, other_id]  # [2]
                    other_momentum = agent_momentum[b, other_id]  # [2]
                    
                    # Skip if other agent has no significant momentum
                    if torch.norm(other_momentum) < 0.1:
                        continue
                    
                    # Predict where the other agent intends to move (next few steps)
                    momentum_norm = other_momentum / (torch.norm(other_momentum) + 1e-8)
                    
                    # Check multiple future positions (1-3 steps ahead)
                    for step in range(1, 4):
                        intended_pos = other_pos + step * momentum_norm
                        
                        # Convert intended position to FOV coordinates relative to current agent
                        rel_x = intended_pos[0] - agent_pos[0] + center_x
                        rel_y = intended_pos[1] - agent_pos[1] + center_y
                        
                        # Check if intended position is within this agent's FOV
                        if 0 <= rel_x < fov_w and 0 <= rel_y < fov_h:
                            fov_x, fov_y = int(rel_x), int(rel_y)
                            
                            # Add soft danger zone (decreasing with distance)
                            danger_intensity = 0.8 / step  # Closer steps are more dangerous
                            
                            # Apply Gaussian-like spread to make it a soft zone
                            for dx in range(-1, 2):
                                for dy in range(-1, 2):
                                    nx, ny = fov_x + dx, fov_y + dy
                                    if 0 <= nx < fov_w and 0 <= ny < fov_h:
                                        # Distance-based decay
                                        dist = max(abs(dx), abs(dy))
                                        spread_intensity = danger_intensity * (0.5 ** dist)
                                        
                                        # Add to intention danger channel (channel 2)
                                        enhanced_fov[b, agent_id, 2, ny, nx] += spread_intensity
                
                # === OCCUPIED GOAL DANGER ZONES ===
                # Add danger zones for goals that are occupied by other agents
                if goals is not None:
                    goal_reach_threshold = 1.0  # Distance threshold to consider goal reached
                    
                    for other_id in range(num_agents):
                        if other_id == agent_id:
                            continue
                            
                        other_pos = positions[b, other_id]  # [2]
                        other_goal = goals[b, other_id]  # [2]
                        
                        # Check if other agent has reached their goal
                        dist_to_goal = torch.norm(other_pos - other_goal)
                        if dist_to_goal <= goal_reach_threshold:
                            # This goal is occupied - mark it as a danger zone
                            occupied_goal_pos = other_goal
                            
                            # Convert occupied goal position to FOV coordinates relative to current agent
                            rel_x = occupied_goal_pos[0] - agent_pos[0] + center_x
                            rel_y = occupied_goal_pos[1] - agent_pos[1] + center_y
                            
                            # Check if occupied goal is within this agent's FOV
                            if 0 <= rel_x < fov_w and 0 <= rel_y < fov_h:
                                fov_x, fov_y = int(rel_x), int(rel_y)
                                
                                # Add strong danger zone for occupied goal
                                occupied_goal_danger = 0.9  # High danger intensity for occupied goals
                                
                                # Apply spread around the occupied goal
                                for dx in range(-1, 2):
                                    for dy in range(-1, 2):
                                        nx, ny = fov_x + dx, fov_y + dy
                                        if 0 <= nx < fov_w and 0 <= ny < fov_h:
                                            # Distance-based decay
                                            dist = max(abs(dx), abs(dy))
                                            spread_intensity = occupied_goal_danger * (0.7 ** dist)
                                            
                                            # Add to intention danger channel (channel 2)
                                            enhanced_fov[b, agent_id, 2, ny, nx] += spread_intensity
        
        # Clamp danger values to reasonable range
        enhanced_fov[:, :, 2, :, :] = torch.clamp(enhanced_fov[:, :, 2, :, :], 0.0, 1.0)
        
        return enhanced_fov
    
    def compute_intention_flow_field(self, positions: torch.Tensor, 
                                   fov_observations: torch.Tensor,
                                   goals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute dynamic intention flow field based on STDP traces, momentum consensus, and intention dangers.
        
        Args:
            positions: [batch, agents, 2] current positions
            fov_observations: [batch, agents, 2, 7, 7] FOV observations
            goals: [batch, agents, 2] goal positions (optional)
            
        Returns:
            flow_field_scores: [batch, agents, directions] intention flow scores
        """
        batch_size, num_agents = positions.shape[:2]
        device = positions.device
        
        # Update momentum vectors
        self.update_momentum(positions)
        
        # Add intention danger zones to FOV (including occupied goals)
        enhanced_fov = self.add_intention_danger_zones(fov_observations, positions, self.agent_momentum, goals)
        
        # Compute nearby momentum consensus
        consensus_momentum = self.compute_nearby_momentum_alignment(positions, self.agent_momentum)
        
        # Initialize flow field scores
        flow_field_scores = torch.zeros(batch_size, num_agents, self.num_directions, device=device)
        
        # Compute scores for each direction
        for direction_idx in range(self.num_directions):
            action_delta = self.action_deltas[direction_idx].to(device)  # [2]
            
            for b in range(batch_size):
                for a in range(num_agents):
                    pos = positions[b, a].long()
                    # Clamp positions to board bounds
                    x = torch.clamp(pos[0], 0, self.board_size - 1)
                    y = torch.clamp(pos[1], 0, self.board_size - 1)
                    
                    # Get STDP traces at current position
                    facilitation = self.stdp_facilitation_traces[b, a, y, x, direction_idx]
                    inhibition = self.stdp_inhibition_traces[b, a, y, x, direction_idx]
                    
                    # Compute momentum alignment with this direction
                    agent_mom = self.agent_momentum[b, a]  # [2]
                    consensus_mom = consensus_momentum[b, a]  # [2]
                    
                    # Alignment with agent's own momentum
                    if torch.norm(agent_mom) > 1e-6:
                        agent_mom_norm = agent_mom / (torch.norm(agent_mom) + 1e-8)
                        agent_alignment = torch.dot(agent_mom_norm, action_delta)
                    else:
                        agent_alignment = 0.0
                    
                    # Alignment with consensus momentum
                    if torch.norm(consensus_mom) > 1e-6:
                        consensus_mom_norm = consensus_mom / (torch.norm(consensus_mom) + 1e-8)
                        consensus_alignment = torch.dot(consensus_mom_norm, action_delta)
                    else:
                        consensus_alignment = 0.0
                    
                    # Compute intention danger penalty from enhanced FOV
                    # Check where this direction would lead and if it has intention dangers
                    next_pos = positions[b, a] + action_delta
                    fov_center = 3  # Center of 7x7 FOV
                    
                    # Convert next position to FOV coordinates
                    rel_x = int(next_pos[0] - positions[b, a][0] + fov_center)
                    rel_y = int(next_pos[1] - positions[b, a][1] + fov_center)
                    
                    intention_danger = 0.0
                    if 0 <= rel_x < 7 and 0 <= rel_y < 7:
                        # Check intention danger at the position this action would lead to
                        intention_danger = enhanced_fov[b, a, 2, rel_y, rel_x].item()
                    
                    # Combine all factors
                    stdp_score = facilitation - inhibition
                    momentum_score = (agent_alignment + consensus_alignment) / 2.0
                    danger_penalty = -2.0 * intention_danger  # Negative penalty for dangers
                    
                    # Final flow field score (includes soft danger avoidance)
                    flow_field_scores[b, a, direction_idx] = (
                        stdp_score + self.flow_field_strength * momentum_score + danger_penalty
                    )
        
        return flow_field_scores
    
    def forward(self, positions: torch.Tensor, goals: torch.Tensor, 
               fov_observations: torch.Tensor, spike_outputs: torch.Tensor,
               collisions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing STDP coordination effects.
        
        Args:
            positions: [batch, agents, 2] current positions
            goals: [batch, agents, 2] goal positions
            fov_observations: [batch, agents, 2, 7, 7] FOV observations
            spike_outputs: [batch, agents, directions] spike activity
            collisions: [batch, agents] collision indicators (optional)
            
        Returns:
            Dictionary with coordination outputs and statistics
        """
        batch_size, num_agents = positions.shape[:2]
        device = positions.device
        
        # Initialize traces if needed
        if self.stdp_facilitation_traces is None:
            self.initialize_traces(batch_size, device)
        
        # Default collision detection if not provided
        if collisions is None:
            collisions = torch.zeros(batch_size, num_agents, device=device, dtype=torch.bool)
        
        # Update STDP traces
        self.update_stdp_traces(positions, goals, spike_outputs, collisions)
        
        # Compute intention flow field (this also creates enhanced FOV with occupied goal dangers)
        flow_field_scores = self.compute_intention_flow_field(positions, fov_observations, goals)
        
        # Store enhanced FOV for potential debugging/visualization
        enhanced_fov = self.add_intention_danger_zones(fov_observations, positions, self.agent_momentum)
        
        # Compute statistics
        stats = self.compute_coordination_stats(positions, goals)
        
        self.timestep += 1
        
        return {
            'flow_field_scores': flow_field_scores,
            'facilitation_traces': self.stdp_facilitation_traces,
            'inhibition_traces': self.stdp_inhibition_traces,
            'agent_momentum': self.agent_momentum,
            'enhanced_fov': enhanced_fov,  # Include enhanced FOV with intention dangers
            'coordination_stats': stats
        }
    
    def compute_coordination_stats(self, positions: torch.Tensor, 
                                 goals: torch.Tensor) -> Dict[str, float]:
        """Compute coordination statistics for monitoring"""
        if self.stdp_facilitation_traces is None:
            return {}
        
        # Average trace strengths
        avg_facilitation = torch.mean(self.stdp_facilitation_traces).item()
        avg_inhibition = torch.mean(self.stdp_inhibition_traces).item()
        
        # Average momentum magnitude
        avg_momentum = torch.mean(torch.norm(self.agent_momentum, dim=-1)).item()
        
        # Goal progress (average distance to goals)
        goal_distances = torch.norm(positions - goals, dim=-1)  # [batch, agents]
        avg_goal_distance = torch.mean(goal_distances).item()
        
        return {
            'avg_facilitation_trace': avg_facilitation,
            'avg_inhibition_trace': avg_inhibition,
            'avg_momentum_magnitude': avg_momentum,
            'avg_goal_distance': avg_goal_distance,
            'coordination_timestep': self.timestep
        }
    
    def visualize_intention_dangers(self, enhanced_fov: torch.Tensor, agent_id: int = 0, batch_id: int = 0):
        """
        Visualize the intention danger zones for debugging.
        
        Args:
            enhanced_fov: [batch, agents, 3, 7, 7] enhanced FOV with intention dangers
            agent_id: Agent to visualize
            batch_id: Batch to visualize
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Channel 0: Obstacles
        axes[0].imshow(enhanced_fov[batch_id, agent_id, 0].cpu().numpy(), cmap='Greys')
        axes[0].set_title(f'Agent {agent_id}: Obstacles')
        axes[0].grid(True, alpha=0.3)
        
        # Channel 1: Other agents
        axes[1].imshow(enhanced_fov[batch_id, agent_id, 1].cpu().numpy(), cmap='Blues')
        axes[1].set_title(f'Agent {agent_id}: Other Agents')
        axes[1].grid(True, alpha=0.3)
        
        # Channel 2: Intention dangers (soft danger zones)
        danger_map = enhanced_fov[batch_id, agent_id, 2].cpu().numpy()
        im = axes[2].imshow(danger_map, cmap='Reds', vmin=0, vmax=1)
        axes[2].set_title(f'Agent {agent_id}: Intention Dangers')
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(im, ax=axes[2])
        
        # Mark center position
        for ax in axes:
            ax.plot(3, 3, 'g*', markersize=10, label='Agent Position')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig
