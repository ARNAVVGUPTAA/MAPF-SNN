"""
Enhanced oscillation detection for MAPF-GNN.
Detects various types of repetitive/cyclic patterns, not just A<->B oscillations.
"""

import torch
from collections import defaultdict, deque
import numpy as np


class OscillationDetector:
    """
    Enhanced oscillation detector that can identify various types of repetitive patterns:
    1. Simple A<->B oscillations (A->B->A->B)
    2. Triangular cycles (A->B->C->A)
    3. Longer cycles (A->B->C->D->A)
    4. Partial cycles with backtracking
    5. Stagnation detection (staying in small area)
    """
    
    def __init__(self, history_length=8, min_cycle_length=2, max_cycle_length=6, 
                 stagnation_radius=1.5, stagnation_threshold=0.7, enable_diagnostics=False):
        """
        Initialize oscillation detector.
        
        Args:
            history_length: Maximum number of positions to track per agent
            min_cycle_length: Minimum cycle length to detect (default: 2 for A<->B)
            max_cycle_length: Maximum cycle length to detect 
            stagnation_radius: Maximum distance from starting position to consider stagnation
            stagnation_threshold: Fraction of moves that must be within radius for stagnation
            enable_diagnostics: Enable diagnostic logging
        """
        self.history_length = history_length
        self.min_cycle_length = min_cycle_length
        self.max_cycle_length = max_cycle_length
        self.stagnation_radius = stagnation_radius
        self.stagnation_threshold = stagnation_threshold
        self.enable_diagnostics = enable_diagnostics
        
        # Position history for each agent (batch_id, agent_id -> deque of positions)
        self.position_history = defaultdict(lambda: deque(maxlen=history_length))
        
        # Cycle detection cache to avoid redundant computations
        self.cycle_cache = {}
        
    def clear_history(self):
        """Clear all position history"""
        self.position_history.clear()
        self.cycle_cache.clear()
        
    def clear_batch_history(self, batch_id):
        """Clear history for a specific batch"""
        keys_to_remove = [k for k in self.position_history.keys() if k.startswith(f"batch_{batch_id}_")]
        for key in keys_to_remove:
            del self.position_history[key]
            
    def configure(self, enable_diagnostics=False):
        """Configure oscillation detector settings"""
        self.enable_diagnostics = enable_diagnostics
            
    def update_position(self, batch_id, agent_id, position):
        """
        Update position history for an agent.
        
        Args:
            batch_id: Batch identifier
            agent_id: Agent identifier
            position: Current position tensor [2] (x, y)
        """
        agent_key = f"batch_{batch_id}_agent_{agent_id}"
        
        # Convert to integers for exact comparison (grid-based movement)
        pos_int = position.round().int()
        
        # Add to history
        self.position_history[agent_key].append(pos_int)
        
        # Clear cache for this agent when position updates
        cache_key = f"{agent_key}_cycles"
        if cache_key in self.cycle_cache:
            del self.cycle_cache[cache_key]
            
    def detect_cycles(self, agent_key):
        """
        Detect cyclical patterns in an agent's movement history.
        
        Args:
            agent_key: Key for the agent's position history
            
        Returns:
            List of detected cycles, each as (cycle_length, repetitions, cycle_positions)
        """
        if agent_key not in self.position_history:
            return []
            
        history = list(self.position_history[agent_key])
        if len(history) < self.min_cycle_length * 2:
            return []
            
        # Check cache first
        cache_key = f"{agent_key}_cycles"
        if cache_key in self.cycle_cache:
            return self.cycle_cache[cache_key]
            
        detected_cycles = []
        
        # Check for cycles of different lengths
        for cycle_len in range(self.min_cycle_length, min(self.max_cycle_length + 1, len(history) // 2 + 1)):
            cycles = self._find_cycles_of_length(history, cycle_len)
            detected_cycles.extend(cycles)
            
        # Cache the result
        self.cycle_cache[cache_key] = detected_cycles
        
        return detected_cycles
        
    def _find_cycles_of_length(self, history, cycle_length):
        """Find cycles of a specific length in the position history"""
        if len(history) < cycle_length * 2:
            return []
            
        detected_cycles = []
        
        # Look for repeating patterns of the given length
        for start_idx in range(len(history) - cycle_length * 2 + 1):
            cycle_pattern = history[start_idx:start_idx + cycle_length]
            
            # Check how many times this pattern repeats
            repetitions = 1
            check_idx = start_idx + cycle_length
            
            while check_idx + cycle_length <= len(history):
                next_pattern = history[check_idx:check_idx + cycle_length]
                if self._positions_equal(cycle_pattern, next_pattern):
                    repetitions += 1
                    check_idx += cycle_length
                else:
                    break
                    
            # If we found at least 2 repetitions, it's a cycle
            if repetitions >= 2:
                detected_cycles.append((cycle_length, repetitions, cycle_pattern))
                
        return detected_cycles
        
    def _positions_equal(self, pos_list1, pos_list2):
        """Check if two lists of positions are equal"""
        if len(pos_list1) != len(pos_list2):
            return False
            
        for p1, p2 in zip(pos_list1, pos_list2):
            if not torch.equal(p1, p2):
                return False
                
        return True
        
    def detect_stagnation(self, agent_key):
        """
        Detect if an agent is stagnating (staying in a small area without progress).
        
        Args:
            agent_key: Key for the agent's position history
            
        Returns:
            (is_stagnating, stagnation_score) where score is 0-1
        """
        if agent_key not in self.position_history:
            return False, 0.0
            
        history = list(self.position_history[agent_key])
        if len(history) < 4:
            return False, 0.0
            
        # Get starting position for reference
        start_pos = history[0].float()
        
        # Count how many positions are within stagnation radius
        within_radius = 0
        for pos in history:
            distance = torch.norm(pos.float() - start_pos)
            if distance <= self.stagnation_radius:
                within_radius += 1
                
        stagnation_score = within_radius / len(history)
        is_stagnating = stagnation_score >= self.stagnation_threshold
        
        return is_stagnating, stagnation_score
        
    def compute_oscillation_penalty(self, current_positions, batch_id, oscillation_weight=1000.0, 
                                  goal_positions=None):
        """
        Compute enhanced oscillation penalty for all agents.
        
        Args:
            current_positions: [batch_size, num_agents, 2] current agent positions
            batch_id: Batch identifier
            oscillation_weight: Base penalty weight (will be multiplied by pattern severity)
            goal_positions: [batch_size, num_agents, 2] or [num_agents, 2] goal positions (optional)
            
        Returns:
            total_penalty: Scalar tensor with total oscillation penalty
        """
        # Handle different input shapes
        if current_positions.dim() == 2:
            # [num_agents, 2] -> [1, num_agents, 2]
            current_positions = current_positions.unsqueeze(0)
        
        batch_size, num_agents = current_positions.shape[:2]
        total_penalty = torch.tensor(0.0, device=current_positions.device, dtype=torch.float32)
        
        # Get agents that have reached their goals (no penalty for agents at goal)
        # Determine which agents have reached their goals
        agents_at_goal = None
        if goal_positions is not None:
            agents_at_goal = self._get_agents_at_goal(current_positions, goal_positions)
            
        for b in range(batch_size):
            for a in range(num_agents):
                # Skip agents that have reached their goals
                if agents_at_goal is not None:
                    # Handle different tensor shapes more robustly
                    try:
                        if agents_at_goal.dim() == 2:
                            # [batch_size, num_agents] - expected case
                            agent_at_goal = agents_at_goal[b, a]
                        elif agents_at_goal.dim() == 1:
                            # [num_agents] - single batch case
                            agent_at_goal = agents_at_goal[a]
                        elif agents_at_goal.dim() == 3:
                            # [batch_size, num_agents, 1] - unexpected but handle it
                            agent_at_goal = agents_at_goal[b, a, 0]
                        else:
                            # Unknown shape - try flattening
                            flat_idx = b * num_agents + a
                            agent_at_goal = agents_at_goal.flatten()[flat_idx]
                        
                        # Convert to Python boolean
                        if isinstance(agent_at_goal, torch.Tensor):
                            if agent_at_goal.numel() == 1:
                                agent_at_goal = agent_at_goal.item()
                            else:
                                # This tensor has multiple elements - take the first
                                agent_at_goal = agent_at_goal.flatten()[0].item()
                        
                        if agent_at_goal:
                            continue
                            
                    except Exception:
                        # Skip this agent if we can't determine goal status
                        continue
                    
                agent_key = f"batch_{batch_id}_agent_{a}"
                
                # Update position history
                self.update_position(batch_id, a, current_positions[b, a])
                
                # Detect cycles
                cycles = self.detect_cycles(agent_key)
                
                # Apply penalty for each detected cycle
                for cycle_length, repetitions, cycle_positions in cycles:
                    # More severe penalty for shorter cycles (more obvious oscillation)
                    cycle_severity = 1.0 / cycle_length
                    
                    # More penalty for more repetitions
                    repetition_multiplier = repetitions
                    
                    # Calculate penalty
                    cycle_penalty = oscillation_weight * cycle_severity * repetition_multiplier
                    total_penalty += cycle_penalty
                    
                # Detect stagnation
                is_stagnating, stagnation_score = self.detect_stagnation(agent_key)
                if is_stagnating:
                    # Penalty proportional to stagnation score
                    stagnation_penalty = oscillation_weight * 0.5 * stagnation_score
                    total_penalty += stagnation_penalty
                    
        return total_penalty
        
    def _get_agents_at_goal(self, agent_positions, goal_positions, tolerance=0.5):
        """Determine which agents have reached their goals with robust shape handling."""
        
        # Squeeze out any singleton dimensions that might be causing issues
        if agent_positions.dim() > 3:
            agent_positions = agent_positions.squeeze()
        if goal_positions.dim() > 3:
            goal_positions = goal_positions.squeeze()

        # Ensure inputs are at least 3D: batch x agents x coords
        while agent_positions.dim() < 3:
            agent_positions = agent_positions.unsqueeze(0)
        while goal_positions.dim() < 3:
            goal_positions = goal_positions.unsqueeze(0)

        # If agent_positions has a sequence dimension, unsqueeze goal_positions to match
        if agent_positions.dim() == 4 and goal_positions.dim() == 3:
            # Reshape goal_positions from [batch, agents, coords] to [batch, 1, agents, coords]
            # This allows broadcasting against agent_positions [batch, seq, agents, coords]
            goal_positions = goal_positions.unsqueeze(1)

        try:
            # Use broadcasting to handle shape mismatches gracefully
            b_agent_positions, b_goal_positions = torch.broadcast_tensors(agent_positions, goal_positions)

            # Compute distances [batch, agents]
            distances = torch.norm(b_agent_positions - b_goal_positions, dim=-1)
            agents_at_goal = distances <= tolerance
            
            # Pad back to original agent dimension if necessary
            if agents_at_goal.shape[1] < agent_positions.shape[1]:
                padding_needed = agent_positions.shape[1] - agents_at_goal.shape[1]
                padding = torch.zeros(agents_at_goal.shape[0], padding_needed, dtype=torch.bool, device=agent_positions.device)
                agents_at_goal = torch.cat([agents_at_goal, padding], dim=1)

        except Exception as e:
            # Fallback: return a 'no agents at goal' mask of the correct shape
            return torch.zeros(agent_positions.shape[:-1], dtype=torch.bool, device=agent_positions.device)

        return agents_at_goal
        
    def get_diagnostics(self, batch_id):
        """
        Get diagnostic information about detected oscillations.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            dict with diagnostic information
        """
        diagnostics = {
            'total_cycles': 0,
            'cycle_breakdown': defaultdict(int),
            'stagnating_agents': 0,
            'agent_details': {}
        }
        
        # Get all agents for this batch
        batch_agents = [k for k in self.position_history.keys() if k.startswith(f"batch_{batch_id}_")]
        
        for agent_key in batch_agents:
            agent_id = agent_key.split('_')[-1]
            
            # Detect cycles for this agent
            cycles = self.detect_cycles(agent_key)
            
            # Detect stagnation
            is_stagnating, stagnation_score = self.detect_stagnation(agent_key)
            
            # Update diagnostics
            diagnostics['total_cycles'] += len(cycles)
            
            for cycle_length, repetitions, _ in cycles:
                diagnostics['cycle_breakdown'][f'length_{cycle_length}'] += repetitions
                
            if is_stagnating:
                diagnostics['stagnating_agents'] += 1
                
            # Agent-specific details
            diagnostics['agent_details'][agent_id] = {
                'cycles': len(cycles),
                'stagnating': is_stagnating,
                'stagnation_score': stagnation_score,
                'history_length': len(self.position_history[agent_key])
            }
            
        return diagnostics


# Global oscillation detector instance
_global_oscillation_detector = OscillationDetector()


def compute_enhanced_oscillation_penalty(current_positions, batch_id=0, oscillation_weight=1000.0, 
                                       goal_positions=None):
    """
    Compute enhanced oscillation penalty using the global detector.
    
    Args:
        current_positions: [batch_size, num_agents, 2] or [num_agents, 2] current agent positions
        batch_id: Batch identifier
        oscillation_weight: Base penalty weight (increased by 100x from original)
        goal_positions: [batch_size, num_agents, 2] or [num_agents, 2] goal positions (optional)
        
    Returns:
        total_penalty: Scalar tensor with total oscillation penalty
    """
    # Configure the global detector with config settings
    try:
        from config import config
        enable_diagnostics = config.get('oscillation_enable_diagnostics', False)
        _global_oscillation_detector.configure(enable_diagnostics=enable_diagnostics)
    except ImportError:
        # If config is not available, use default (no diagnostics)
        _global_oscillation_detector.configure(enable_diagnostics=False)
    
    return _global_oscillation_detector.compute_oscillation_penalty(
        current_positions, batch_id, oscillation_weight, goal_positions
    )


def clear_enhanced_oscillation_history():
    """Clear all oscillation history"""
    _global_oscillation_detector.clear_history()


def clear_enhanced_oscillation_batch_history(batch_id):
    """Clear oscillation history for a specific batch"""
    _global_oscillation_detector.clear_batch_history(batch_id)


def get_oscillation_diagnostics(batch_id):
    """Get diagnostic information about detected oscillations"""
    return _global_oscillation_detector.get_diagnostics(batch_id)
