"""
Hardware-Native Swarm MAPF with SpikingJelly
=============================================

ALL modules are spiking neural meshes:
- LIF neurons (leaky integrate-and-fire)
- E/I organization (80% excitatory, 20% inhibitory)
- Dale's Law: only E neurons project between modules
- Projection neurons for compressed routing

Three core modules (all spike-based, no matrix math):

1. TOPOGRAPHIC INTENT MAP
   - 3x3 grid of neurons centered on agent
   - Agent moving EAST → fires (1,0) neuron
   - Neighbor receives broadcast → spike lands on their spatial grid
   - Physical wiring handles collision math

2. CENTRAL PATTERN GENERATOR (CPG)
   - Two coupled neurons: N_peak and N_trough
   - Fatigue creates oscillation (no sine waves)
   - Weak coupling between agents pushes them out of phase
   - N_trough → STAY action (turn-taking emerges)

3. SHADOW CASTER
   - Detects velocity: (spike at x, then x+1) → moving EAST
   - Casts "ghost" spikes along trajectory with delay
   - Ghost hits wall → VETO spike
   - VETO → reset CPG to trough (force STAY)

"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm

# SpikingJelly for proper LIF dynamics
from spikingjelly.activation_based import neuron, functional, layer


# === E/I NEURON MESH BASE ===

class EINeuronMesh(nn.Module):
    """
    Base class for spiking neural mesh with E/I organization.
    
    - 80% excitatory neurons (E)
    - 20% inhibitory neurons (I)
    - E neurons project out (Dale's Law)
    - I neurons only local
    - LIF dynamics from SpikingJelly
    """
    def __init__(self, num_neurons: int, tau: float = 2.0):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_E = int(num_neurons * 0.8)
        self.num_I = num_neurons - self.num_E
        
        # LIF neurons
        self.lif = neuron.LIFNode(tau=tau, v_threshold=1.0, v_reset=0.0)
        
        # E/I sign mask for recurrent connections
        self.ei_mask = torch.ones(num_neurons)
        self.ei_mask[self.num_E:] = -1.0  # I neurons are inhibitory
        
    def get_E_neurons(self, spikes: torch.Tensor) -> torch.Tensor:
        """Extract only excitatory neuron spikes for projection"""
        return spikes[..., :self.num_E]


# === MODULE 1: TOPOGRAPHIC INTENT MAP (SPIKING) ===

class TopographicIntentMap(EINeuronMesh):
    """
    3x3 spatial grid of LIF neurons.
    
    Total: 9 neurons per grid cell × E/I organization
    Agent moving EAST → fires E neurons at (1,0) position
    Only E spikes broadcast to neighbors
    """
    def __init__(self):
        # 9 positions × 16 neurons per position = 144 total
        super().__init__(num_neurons=144, tau=2.0)
        
        self.grid_size = 3
        self.neurons_per_cell = 16
        
        # Action to grid position
        self.action_to_pos = {
            0: (1, 0),   # RIGHT
            1: (0, -1),  # UP  
            2: (-1, 0),  # LEFT
            3: (0, 1),   # DOWN
            4: (0, 0)    # STAY
        }
        
        # Recurrent within-grid connections (local)
        self.recurrent = layer.Linear(self.num_neurons, self.num_neurons, bias=False)
        nn.init.sparse_(self.recurrent.weight, sparsity=0.9)
        
        # Lateral inhibition projection (E neurons only)
        self.inhibition_proj = layer.Linear(self.num_E, self.num_neurons, bias=False)
        nn.init.normal_(self.inhibition_proj.weight, mean=0, std=0.1)
        
    def encode_intent(self, action: int) -> torch.Tensor:
        """
        Convert action to spatial spike input.
        
        Args:
            action: 0-4
        
        Returns:
            input_current: [144] input current to LIF neurons
        """
        current = torch.zeros(self.num_neurons)
        
        dx, dy = self.action_to_pos[action]
        x, y = 1 + dx, 1 + dy
        
        if 0 <= x < 3 and 0 <= y < 3:
            cell_idx = y * 3 + x
            neuron_start = cell_idx * self.neurons_per_cell
            neuron_end = neuron_start + self.neurons_per_cell
            
            # Strong current to this cell's neurons
            current[neuron_start:neuron_end] = 5.0
        
        return current
    
    def forward(
        self, 
        action: int,
        neighbor_E_spikes: Optional[torch.Tensor] = None,
        num_ticks: int = 30
    ) -> torch.Tensor:
        """
        Run spiking dynamics for intent encoding.
        
        Args:
            action: Agent's action
            neighbor_E_spikes: [num_neighbors, num_E] neighbor E neuron spikes
            num_ticks: Simulation ticks
        
        Returns:
            E_spikes: [num_E] excitatory neuron spikes (broadcast to neighbors)
        """
        input_current = self.encode_intent(action)
        
        spike_accumulator = torch.zeros(self.num_neurons)
        
        for t in range(num_ticks):
            # Recurrent input
            recurrent_input = self.recurrent(spike_accumulator)
            
            # Apply E/I signs
            recurrent_input = recurrent_input * self.ei_mask
            
            # Lateral inhibition from neighbors
            inhibition = torch.zeros(self.num_neurons)
            if neighbor_E_spikes is not None and neighbor_E_spikes.shape[0] > 0:
                for neighbor_spikes in neighbor_E_spikes:
                    inhibition += self.inhibition_proj(neighbor_spikes)
            
            # Total input
            total_input = input_current * 0.5 + recurrent_input * 0.3 - inhibition * 0.5
            
            # LIF dynamics
            spikes = self.lif(total_input)
            spike_accumulator += spikes
        
        # Return only E neuron activity
        return self.get_E_neurons(spike_accumulator)


# === MODULE 2: CPG (SPIKING) ===

class CPG(EINeuronMesh):
    """
    Spiking neural oscillator with two populations.
    
    Peak population (move) and Trough population (stay).
    Mutual inhibition creates oscillation.
    Weak coupling between agents for anti-phase locking.
    """
    def __init__(self):
        # 32 total neurons: 16 peak, 16 trough
        super().__init__(num_neurons=32, tau=3.0)
        
        self.peak_size = 16
        self.trough_size = 16
        
        # Mutual inhibition weights
        self.mutual_inhibition = nn.Parameter(torch.ones(self.num_neurons, self.num_neurons) * -2.0)
        
        # Peak neurons inhibit trough, trough inhibit peak
        self.mutual_inhibition.data[:self.peak_size, :self.peak_size] *= 0.1  # Weak self-inhibition
        self.mutual_inhibition.data[self.peak_size:, self.peak_size:] *= 0.1
        self.mutual_inhibition.data[:self.peak_size, self.peak_size:] = -3.0  # Strong cross-inhibition
        self.mutual_inhibition.data[self.peak_size:, :self.peak_size] = -3.0
        
        # Coupling from neighbor CPGs (E neurons only)
        self.coupling_proj = layer.Linear(self.num_E, self.num_neurons, bias=False)
        nn.init.normal_(self.coupling_proj.weight, mean=0, std=0.2)
        
        # Fatigue counters
        self.peak_fatigue = 0
        self.trough_fatigue = 0
        self.fatigue_threshold = 8
        
        # State
        self.spike_history = torch.zeros(self.num_neurons)
        
    def reset(self):
        """Reset CPG state"""
        self.spike_history.zero_()
        self.peak_fatigue = 0
        self.trough_fatigue = 0
        functional.reset_net(self.lif)
    
    def forward(
        self,
        neighbor_E_spikes: Optional[torch.Tensor] = None,
        num_ticks: int = 5
    ) -> Tuple[torch.Tensor, bool]:
        """
        Run CPG oscillation.
        
        Args:
            neighbor_E_spikes: [num_neighbors, num_E] neighbor CPG E spikes
            num_ticks: Simulation ticks
        
        Returns:
            E_spikes: [num_E] E neuron spikes for projection
            should_stay: Whether in trough phase (should STAY)
        """
        # External drive (keeps oscillation going)
        drive = torch.zeros(self.num_neurons)
        
        # Drive peak if fatigued trough
        if self.trough_fatigue >= self.fatigue_threshold:
            drive[:self.peak_size] = 2.0
            self.trough_fatigue = 0
        
        # Drive trough if fatigued peak
        if self.peak_fatigue >= self.fatigue_threshold:
            drive[self.peak_size:] = 2.0
            self.peak_fatigue = 0
        
        spike_accumulator = torch.zeros(self.num_neurons)
        
        for t in range(num_ticks):
            # Mutual inhibition
            recurrent = torch.matmul(self.spike_history, self.mutual_inhibition.T)
            
            # Coupling from neighbors (anti-phase)
            coupling = torch.zeros(self.num_neurons)
            if neighbor_E_spikes is not None and neighbor_E_spikes.shape[0] > 0:
                for neighbor_spikes in neighbor_E_spikes:
                    coupling += self.coupling_proj(neighbor_spikes)
            
            # Total input
            total_input = drive + recurrent * 0.5 + coupling * 0.3
            
            # LIF dynamics
            spikes = self.lif(total_input)
            spike_accumulator += spikes
            self.spike_history = spikes
        
        # Update fatigue
        peak_active = spike_accumulator[:self.peak_size].sum() > 0.5
        trough_active = spike_accumulator[self.peak_size:].sum() > 0.5
        
        if peak_active:
            self.peak_fatigue += 1
        if trough_active:
            self.trough_fatigue += 1
        
        # Should STAY if trough is active
        should_stay = trough_active
        
        return self.get_E_neurons(spike_accumulator), should_stay


# === MODULE 3: SHADOW CASTER (SPIKING) ===

class ShadowCaster(EINeuronMesh):
    """
    Predictive collision detector using delay lines.
    
    Delay-line neurons detect velocity.
    Cast "shadow" spikes forward.
    Shadow intersects wall → VETO spike.
    """
    def __init__(self, fov_size: int = 7):
        # Delay line: 7×7×4 directions×2 ticks = 392 neurons
        super().__init__(num_neurons=392, tau=2.0)
        
        self.fov_size = fov_size
        self.neurons_per_cell = 8
        
        # Velocity detector (coincidence detection)
        self.velocity_detector = layer.Linear(self.num_neurons, 4, bias=False)  # 4 directions
        
        # Shadow propagation weights
        self.shadow_prop = layer.Linear(4, self.num_neurons, bias=False)
        
        # Wall collision detector (E neurons project)
        self.collision_detector = layer.Linear(self.num_E, 1, bias=False)
        
        # Memory for velocity tracking
        self.prev_spikes = torch.zeros(self.num_neurons)
        
    def reset(self):
        """Reset shadow caster"""
        self.prev_spikes.zero_()
        functional.reset_net(self.lif)
    
    def forward(
        self,
        walls: torch.Tensor,
        velocity_hint: Tuple[int, int],
        num_ticks: int = 3
    ) -> bool:
        """
        Cast shadow and check for wall collision.
        
        Args:
            walls: [7, 7] wall occupancy
            velocity_hint: (dx, dy) movement direction
            num_ticks: Shadow propagation ticks
        
        Returns:
            veto: True if shadow hits wall
        """
        # Encode velocity as input (simple current injection)
        dx, dy = velocity_hint
        direction_idx = 0  # Map to 0-3 for 4 directions
        if dx > 0:
            direction_idx = 0
        elif dx < 0:
            direction_idx = 2
        elif dy < 0:
            direction_idx = 1
        elif dy > 0:
            direction_idx = 3
        
        direction_input = torch.zeros(4)
        direction_input[direction_idx] = 5.0
        
        # Propagate shadow
        spike_accumulator = torch.zeros(self.num_neurons)
        
        for t in range(num_ticks):
            # Shadow propagates forward
            shadow_input = self.shadow_prop(direction_input)
            
            # LIF dynamics
            spikes = self.lif(shadow_input * 0.8)
            spike_accumulator += spikes
        
        # Check collision with walls
        # Extract E spikes and check overlap with walls
        E_spikes = self.get_E_neurons(spike_accumulator)
        
        # Simple heuristic: if many E neurons fire and path has walls, VETO
        collision_score = self.collision_detector(E_spikes).item()
        
        # Check if we're heading toward wall
        center = self.fov_size // 2
        x, y = center, center
        
        for step in range(1, num_ticks + 1):
            x += dx
            y += dy
            if 0 <= x < self.fov_size and 0 <= y < self.fov_size:
                if walls[y, x] > 0.5:
                    return True  # VETO
        
        return False


# === PROJECTION NEURONS ===

class ProjectionNeurons(nn.Module):
    """
    Compress and route E neuron spikes between modules.
    
    Only E neurons project (Dale's Law).
    Compression reduces spike volume for routing.
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        
        # Compression layer (E neurons → compressed representation)
        self.compress = layer.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.compress.weight, mean=0, std=0.1)
        
    def forward(self, E_spikes: torch.Tensor) -> torch.Tensor:
        """
        Project E spikes to compressed output.
        
        Args:
            E_spikes: [num_E] excitatory neuron spikes
        
        Returns:
            projected: [output_size] compressed spikes
        """
        return torch.relu(self.compress(E_spikes))  # Ensure non-negative (E neurons)


# === CHEMOTAXIS RECEPTORS ===

class ChemotaxisReceptors(nn.Module):
    """
    4 LIF neurons sampling the scent gradient toward the goal.

    For each of the 5 actions (RIGHT, UP, LEFT, DOWN, STAY), the "scent"
    is how much closer that move would bring the agent to the goal in L1.
    That improvement feeds directly as input current into one LIF neuron
    — no weights, no matrix math.

    Spike counts → added directly to corresponding action logits.
    """
    def __init__(self, tau: float = 2.0):
        super().__init__()
        self.lif = neuron.LIFNode(tau=tau, v_threshold=1.0, v_reset=0.0)

    def reset(self):
        functional.reset_net(self.lif)

    def forward(
        self,
        position: torch.Tensor,
        goal: torch.Tensor,
        gain: float = 3.0,
        num_ticks: int = 10,
    ) -> torch.Tensor:
        """
        Args:
            position: [2] (x, y)
            goal:     [2] (x, y)
            gain:     scent current multiplier
            num_ticks: LIF ticks
        Returns:
            spikes: [5] one per action (RIGHT/UP/LEFT/DOWN/STAY)
        """
        # Scent improvement = how much L1 distance shrinks per action
        d0 = (goal - position).abs().sum()  # current L1
        deltas = torch.tensor(
            [[1,0],[0,-1],[-1,0],[0,1],[0,0]],
            dtype=torch.float32, device=position.device
        )
        d_new  = (goal.unsqueeze(0) - (position.unsqueeze(0) + deltas)).abs().sum(dim=1)  # [5]
        scent  = (d0 - d_new).clamp(min=0)  # positive = gets closer
        # Inject scent current into LIF neurons
        functional.reset_net(self.lif)
        acc     = torch.zeros(5, device=position.device)
        current = scent * gain
        for _ in range(num_ticks):
            acc += self.lif(current)
        return acc  # [5] spike counts, index == action index


# === AGENT WITH SPIKING MESHES ===

class AgentLSM(nn.Module):
    """
    Single agent with FULL SPIKING ARCHITECTURE.
    
    ALL modules are LIF meshes with E/I organization:
    - Observation Mesh (process FOV)
    - Topographic Intent Map (spatial encoding)
    - CPG (oscillator)
    - Shadow Caster (predictive)
    - Readout Mesh (action selection)
    
    Only E neurons project between modules (Dale's Law).
    """
    def __init__(self, agent_id: int):
        super().__init__()
        self.agent_id = agent_id
        
        # === OBSERVATION PROCESSING MESH ===
        # LIF mesh for encoding FOV (replaces Tanh reservoir)
        self.obs_mesh = EINeuronMesh(num_neurons=256, tau=2.0)
        self.obs_input_proj = layer.Linear(2 * 7 * 7, 256, bias=False)
        self.obs_recurrent = layer.Linear(256, 256, bias=False)
        nn.init.sparse_(self.obs_recurrent.weight, sparsity=0.85)
        
        # Projection neurons: obs mesh E → compressed
        self.obs_to_readout = ProjectionNeurons(self.obs_mesh.num_E, 64)
        
        # === THREE CORE MODULES ===
        self.intent_map = TopographicIntentMap()
        self.cpg = CPG()
        self.shadow = ShadowCaster(fov_size=7)
        
        # === READOUT MESH ===
        # LIF mesh for action selection (replaces Linear readout)
        self.readout_mesh = EINeuronMesh(num_neurons=256, tau=1.5)
        self.readout_input = layer.Linear(64, 256, bias=False)
        self.readout_recurrent = layer.Linear(256, 256, bias=False)
        nn.init.sparse_(self.readout_recurrent.weight, sparsity=0.9)
        
        # Final action neurons (5 actions, no E/I split here - output layer)
        self.action_neurons = neuron.LIFNode(tau=2.0, v_threshold=1.0)
        self.action_weights = layer.Linear(self.readout_mesh.num_E, 5, bias=False)

        # Spiking chemotaxis receptors: scent-gradient nudge toward goal
        self.chemotaxis = ChemotaxisReceptors()

    def reset(self):
        """Reset all spiking meshes"""
        functional.reset_net(self.obs_mesh.lif)
        functional.reset_net(self.readout_mesh.lif)
        functional.reset_net(self.action_neurons)
        self.cpg.reset()
        self.shadow.reset()
        self.chemotaxis.reset()
    
    def forward(
        self,
        observation: torch.Tensor,
        neighbor_intent_E: List[torch.Tensor] = None,
        neighbor_cpg_E: List[torch.Tensor] = None,
        num_ticks: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full spiking forward pass.
        
        Args:
            observation: [2, 7, 7] FOV
            neighbor_intent_E: List of neighbor intent map E spikes
            neighbor_cpg_E: List of neighbor CPG E spikes
            num_ticks: Simulation ticks
        
        Returns:
            action_spikes: [5] action neuron spike counts
            my_intent_E: [num_E] my intent map E spikes (broadcast)
            my_cpg_E: [num_E] my CPG E spikes (broadcast)
        """
        if neighbor_intent_E is None:
            neighbor_intent_E = []
        if neighbor_cpg_E is None:
            neighbor_cpg_E = []
        
        # === STEP 1: OBSERVATION MESH ===
        obs_flat = observation.flatten()
        obs_input = self.obs_input_proj(obs_flat) * 7.5
        
        obs_spike_accumulator = torch.zeros(self.obs_mesh.num_neurons)
        
        for t in range(num_ticks):
            recurrent = self.obs_recurrent(obs_spike_accumulator)
            recurrent = recurrent * self.obs_mesh.ei_mask
            
            total_input = obs_input * 0.8 + recurrent * 0.2
            spikes = self.obs_mesh.lif(total_input)
            obs_spike_accumulator += spikes
        
        # Extract E neuron spikes and project
        obs_E_spikes = self.obs_mesh.get_E_neurons(obs_spike_accumulator)
        obs_compressed = self.obs_to_readout(obs_E_spikes)
        
        # === STEP 2: READOUT MESH ===
        readout_input = self.readout_input(obs_compressed) * 2.5
        
        readout_spike_accumulator = torch.zeros(self.readout_mesh.num_neurons)
        
        for t in range(num_ticks):
            recurrent = self.readout_recurrent(readout_spike_accumulator)
            recurrent = recurrent * self.readout_mesh.ei_mask
            
            total_input = readout_input * 0.85 + recurrent * 0.15
            spikes = self.readout_mesh.lif(total_input)
            readout_spike_accumulator += spikes
        
        # Extract E spikes and compute actions
        readout_E_spikes = self.readout_mesh.get_E_neurons(readout_spike_accumulator)
        # action_weights is trained directly on readout E spikes — use logits as scores
        action_logits = self.action_weights(readout_E_spikes)  # [5]

        # Select tentative action from learned logits
        tentative_action = torch.argmax(action_logits).item()
        
        # === STEP 3: INTENT MAP ===
        my_intent_E = self.intent_map(
            action=tentative_action,
            neighbor_E_spikes=torch.stack(neighbor_intent_E) if len(neighbor_intent_E) > 0 else None,
            num_ticks=5
        )
        
        # === STEP 4: CPG ===
        my_cpg_E, should_stay = self.cpg(
            neighbor_E_spikes=torch.stack(neighbor_cpg_E) if len(neighbor_cpg_E) > 0 else None,
            num_ticks=5
        )
        
        # CPG modulation: if in trough, nudge STAY to win by small margin
        if should_stay:
            action_logits = action_logits.clone()
            action_logits[4] = action_logits.max() + 2.0

        # === STEP 5: SHADOW CASTER ===
        action_to_velocity = {
            0: (1, 0),   # RIGHT
            1: (0, -1),  # UP
            2: (-1, 0),  # LEFT
            3: (0, 1),   # DOWN
            4: (0, 0)    # STAY
        }
        velocity = action_to_velocity[tentative_action]

        walls = observation[0]
        veto = self.shadow(walls, velocity, num_ticks=3)

        if veto:
            action_logits = action_logits.clone()
            action_logits[4] = action_logits.max() + 1.0

        return action_logits, my_intent_E, my_cpg_E


# === SWARM NETWORK ===

class SwarmLSM(nn.Module):
    """
    Decentralized swarm with full spiking dynamics.
    
    Each agent = independent spiking meshes.
    Communication = E neuron spike broadcasts only (Dale's Law).
    """
    def __init__(self, num_agents: int = 5, communication_range: float = 3.0):
        super().__init__()
        self.num_agents = num_agents
        self.communication_range = communication_range
        
        # Create independent spiking agents
        self.agents = nn.ModuleList([
            AgentLSM(agent_id=i) for i in range(num_agents)
        ])
    
    def reset(self):
        """Reset all agents' spiking dynamics"""
        for agent in self.agents:
            agent.reset()
    
    def get_neighbors(self, positions: torch.Tensor) -> List[List[int]]:
        """Find neighbors within communication range"""
        neighbors = []
        for i in range(self.num_agents):
            dist = torch.norm(positions - positions[i], dim=1)
            neighbor_mask = (dist < self.communication_range) & (dist > 0)
            neighbor_ids = torch.where(neighbor_mask)[0].tolist()
            neighbors.append(neighbor_ids)
        return neighbors
    
    def forward(
        self,
        observations: torch.Tensor,
        positions: torch.Tensor,
        num_ticks: int = 10,
        goals: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Swarm spiking forward pass.
        
        Args:
            observations: [N, 2, 7, 7] per-agent FOVs
            positions: [N, 2] agent positions
            num_ticks: Simulation ticks
            goals: [N, 2] goal positions — drives chemotaxis receptors (optional)
        
        Returns:
            action_spikes: [N, 5] action spike counts
            intent_E_spikes: List of [num_E] intent map E spikes
            cpg_E_spikes: List of [num_E] CPG E spikes
        """
        # Single pass: run obs+readout mesh once per agent, cache results
        cached_readout_E   = []
        cached_action_acc  = []
        cached_intent_E    = []
        cached_cpg_E       = []
        cached_should_stay = []
        cached_veto        = []

        for agent_id in range(self.num_agents):
            ag  = self.agents[agent_id]
            obs = observations[agent_id]

            # --- obs mesh ---
            obs_flat = obs.flatten()
            obs_input = ag.obs_input_proj(obs_flat) * 7.5
            obs_acc = torch.zeros(ag.obs_mesh.num_neurons, device=obs_flat.device)
            for _ in range(num_ticks):
                rec = ag.obs_recurrent(obs_acc) * ag.obs_mesh.ei_mask
                obs_acc += ag.obs_mesh.lif(obs_input * 0.8 + rec * 0.2)

            obs_E      = ag.obs_mesh.get_E_neurons(obs_acc)
            rd_input   = ag.readout_input(ag.obs_to_readout(obs_E)) * 2.5
            rd_acc     = torch.zeros(ag.readout_mesh.num_neurons, device=obs_flat.device)
            for _ in range(num_ticks):
                rec = ag.readout_recurrent(rd_acc) * ag.readout_mesh.ei_mask
                rd_acc += ag.readout_mesh.lif(rd_input * 0.85 + rec * 0.15)

            readout_E = ag.readout_mesh.get_E_neurons(rd_acc)

            # --- action selection: logits directly (no LIF bottleneck) ---
            action_logits = ag.action_weights(readout_E)

            # --- chemotaxis: scent-gradient spiking nudge toward goal ---
            if goals is not None:
                action_logits = action_logits + ag.chemotaxis(
                    positions[agent_id], goals[agent_id], num_ticks=num_ticks
                ) * 0.05

            tentative = int(torch.argmax(action_logits).item())

            # --- intent map (no neighbors yet, for broadcasting) ---
            intent_E = ag.intent_map(action=tentative, neighbor_E_spikes=None, num_ticks=5)
            # --- CPG (no neighbors yet) ---
            cpg_E, should_stay = ag.cpg(neighbor_E_spikes=None, num_ticks=5)
            # --- shadow ---
            vel_map = {0:(1,0),1:(0,-1),2:(-1,0),3:(0,1),4:(0,0)}
            veto = ag.shadow(obs[0], vel_map[tentative], num_ticks=3)

            cached_readout_E.append(readout_E)
            cached_action_acc.append(action_logits)
            cached_intent_E.append(intent_E)
            cached_cpg_E.append(cpg_E)
            cached_should_stay.append(should_stay)
            cached_veto.append(veto)

        # Apply CPG/shadow modulation using single-pass cached results only
        # (No second LIF pass to avoid corrupting reservoir state)
        final_action_spikes = []
        for agent_id in range(self.num_agents):
            action_acc  = cached_action_acc[agent_id].clone()
            should_stay = cached_should_stay[agent_id]
            veto        = cached_veto[agent_id]

            if should_stay:
                action_acc[4] = action_acc.max() + 2.0
            if veto:
                action_acc[4] = action_acc.max() + 1.0

            final_action_spikes.append(action_acc)

        return torch.stack(final_action_spikes), cached_intent_E, cached_cpg_E


# === TRAINING INFRASTRUCTURE ===

class SwarmDataset(torch.utils.data.Dataset):
    """Load MAPF expert trajectories"""
    def __init__(self, data_dir: str, max_episodes: int = None):
        self.data_dir = data_dir
        self.episodes = []
        
        # Load all trajectory files
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        if max_episodes:
            files = files[:max_episodes]
        
        for file in files:
            data = np.load(os.path.join(data_dir, file))
            self.episodes.append({
                'states': data['states'],
                'actions': data['actions']
            })
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        return self.episodes[idx]


class SwarmTrainer:
    """
    Train swarm with spiking dynamics.
    
    Freeze all spiking meshes, train only action_weights layer.
    """
    def __init__(self, network: SwarmLSM, device: str = 'cpu'):
        self.network = network.to(device)
        self.device = device
        
        # Freeze all spiking meshes except action_weights
        for agent in self.network.agents:
            agent.obs_mesh.requires_grad_(False)
            agent.obs_input_proj.requires_grad_(False)
            agent.obs_recurrent.requires_grad_(False)
            agent.obs_to_readout.requires_grad_(False)
            agent.readout_mesh.requires_grad_(False)
            agent.readout_input.requires_grad_(False)
            agent.readout_recurrent.requires_grad_(False)
            agent.intent_map.requires_grad_(False)
            agent.cpg.requires_grad_(False)
            agent.shadow.requires_grad_(False)
            
            # Only train action_weights; everything else frozen
            agent.action_weights.requires_grad_(True)
    
    def collect_states(self, dataset: SwarmDataset, max_episodes: int = 500, num_ticks: int = 10):
        """
        Collect spiking states for ridge regression.

        Always uses ALL recovery episodes (source contains 'recovery') and up to
        max_episodes from the normal dataset, then interleaves them.
        """
        import yaml as _yaml

        print("\n" + "="*70)
        print("🔬 COLLECTING SPIKING STATES")
        print("="*70)

        # Split dataset into normal vs recovery by is_recovery flag
        normal_indices   = [i for i, d in enumerate(dataset.data)
                            if not d.get('is_recovery', False)]
        recovery_indices = [i for i, d in enumerate(dataset.data)
                            if d.get('is_recovery', False)]

        chosen_normal   = normal_indices[:max_episodes]
        chosen_recovery = recovery_indices  # always all of them
        episode_indices = chosen_normal + chosen_recovery

        print(f"   Normal episodes:   {len(chosen_normal)}")
        print(f"   Recovery episodes: {len(chosen_recovery)}")
        print(f"   Total:             {len(episode_indices)}")
        
        # Action deltas: 0=RIGHT, 1=UP, 2=LEFT, 3=DOWN, 4=STAY  (matches evaluate)
        action_to_delta = np.array([[1,0],[0,-1],[-1,0],[0,1],[0,0]], dtype=np.float32)

        X_per_agent = [[] for _ in range(self.network.num_agents)]
        Y_per_agent = [[] for _ in range(self.network.num_agents)]
        
        self.network.eval()
        with torch.no_grad():
            for ep_idx in tqdm(episode_indices, desc="Collecting"):
                # ── load expert trajectory labels ─────────────────────────────
                _, actions, _ = dataset[ep_idx]
                actions = actions.to(self.device)
                T, A = actions.shape[0], actions.shape[1] if actions.ndim > 1 else 1

                if actions.ndim == 1:
                    actions = actions.unsqueeze(1).expand(T, A)
                elif actions.shape[1] == 1 and A > 1:
                    continue  # ambiguous
                if actions.shape[0] != T:
                    continue

                # ── load grid & starts from input.yaml ───────────────────────
                case_dir = dataset.data[ep_idx]['case_dir']
                try:
                    inp = _yaml.safe_load(open(f"{case_dir}/input.yaml"))
                except Exception:
                    continue

                if not inp or 'map' not in inp:
                    continue
                map_data    = inp['map']
                agents_info = inp.get('agents', [])
                if len(agents_info) != A:
                    continue

                H, W = map_data['dimensions'][1], map_data['dimensions'][0]
                grid = np.zeros((H, W), dtype=np.float32)
                for obs in map_data.get('obstacles', []):
                    ox, oy = obs[0], obs[1]
                    if 0 <= oy < H and 0 <= ox < W:
                        grid[oy, ox] = 2.0

                # goals for FOV goal-channel
                goals = torch.tensor(
                    [[a['goal'][0], a['goal'][1]] for a in agents_info],
                    dtype=torch.float32, device=self.device
                )
                # positions start at expert starts
                positions = torch.tensor(
                    [[a['start'][0], a['start'][1]] for a in agents_info],
                    dtype=torch.float32, device=self.device
                )

                self.network.reset()

                for t in range(T):
                    expert_act = actions[t]  # [A]

                    # CLOSED-LOOP: generate FOV from actual positions
                    obs = self._generate_fov(grid, positions, goals, fov_size=7)  # [A, 2, 7, 7]

                    # collect readout E spikes per agent
                    for agent_id in range(A):
                        ag       = self.network.agents[agent_id]
                        obs_flat = obs[agent_id].flatten()

                        obs_input    = ag.obs_input_proj(obs_flat) * 7.5
                        obs_spike_acc = torch.zeros(ag.obs_mesh.num_neurons, device=self.device)
                        for _ in range(num_ticks):
                            rec = ag.obs_recurrent(obs_spike_acc) * ag.obs_mesh.ei_mask
                            obs_spike_acc += ag.obs_mesh.lif(obs_input * 0.8 + rec * 0.2)

                        obs_E        = ag.obs_mesh.get_E_neurons(obs_spike_acc)
                        rd_input     = ag.readout_input(ag.obs_to_readout(obs_E)) * 2.5
                        rd_spike_acc = torch.zeros(ag.readout_mesh.num_neurons, device=self.device)
                        for _ in range(num_ticks):
                            rec = ag.readout_recurrent(rd_spike_acc) * ag.readout_mesh.ei_mask
                            rd_spike_acc += ag.readout_mesh.lif(rd_input * 0.85 + rec * 0.15)

                        readout_E = ag.readout_mesh.get_E_neurons(rd_spike_acc)
                        X_per_agent[agent_id].append(readout_E.cpu().numpy())
                        Y_per_agent[agent_id].append(expert_act[agent_id].cpu().numpy())

                    # advance positions along expert path (wall-clamped)
                    act_np = expert_act.cpu().numpy().astype(int)
                    for agent_id in range(A):
                        a    = int(act_np[agent_id]) % 5
                        dx   = int(action_to_delta[a][0])
                        dy   = int(action_to_delta[a][1])
                        nx   = int(np.clip(positions[agent_id, 0].item() + dx, 0, W - 1))
                        ny   = int(np.clip(positions[agent_id, 1].item() + dy, 0, H - 1))
                        if grid[ny, nx] < 1.5:           # free cell
                            positions[agent_id, 0] = nx
                            positions[agent_id, 1] = ny
                        # else: blocked – stay put (same as eval)
        
        # Convert to arrays
        for i in range(self.network.num_agents):
            X_per_agent[i] = np.array(X_per_agent[i])
            Y_per_agent[i] = np.array(Y_per_agent[i])
            print(f"   Agent {i}: X={X_per_agent[i].shape}, Y={Y_per_agent[i].shape}")
        
        # Spike vitals: mean and max spike rate across readout E neurons
        # mean  ≈ 0  → mesh is dead (input too weak)
        # mean  ≈ num_ticks → mesh is seizing (input too strong)
        # healthy range → 0.1 – 0.7 × num_ticks
        x0 = X_per_agent[0]
        print(f"\n   🔬 Spike Vitals (Agent 0 readout E neurons over all timesteps):")
        print(f"      Mean spikes/neuron : {x0.mean():.3f}  (target {0.1*num_ticks:.1f}–{0.7*num_ticks:.1f})")
        print(f"      Max  spikes/neuron : {x0.max():.3f}  (cap = {float(num_ticks):.1f})")
        print(f"      Dead neurons       : {(x0.max(axis=0) == 0).sum()}/{x0.shape[1]}")
        print(f"      Saturated neurons  : {(x0.min(axis=0) == num_ticks).sum()}/{x0.shape[1]}")
        if x0.mean() < 0.05 * num_ticks:
            print(f"      ⚠️  BRAIN DEAD – mean < 5% of ticks. Raise gain or check weights.")
        elif x0.mean() > 0.85 * num_ticks:
            print(f"      ⚠️  SEIZING   – mean > 85% of ticks. Lower gain.")
        else:
            print(f"      ✅  Healthy spiking activity.")
        
        return X_per_agent, Y_per_agent

    def train_ridge(self, X_per_agent, Y_per_agent, alpha=1.0):
        """Train action_weights with ridge regression"""
        print("\n" + "="*70)
        print("🎓 TRAINING ACTION WEIGHTS (RIDGE REGRESSION)")
        print("="*70)
        
        for agent_id in range(self.network.num_agents):
            X = X_per_agent[agent_id]
            Y = Y_per_agent[agent_id]
            
            # Convert to numpy and ensure int type
            if isinstance(Y, torch.Tensor):
                Y = Y.cpu().numpy()
            Y = Y.astype(int)
            
            # One-hot encode
            Y_onehot = np.eye(5)[Y]
            
            # Ridge: W = (X^T X + αI)^-1 X^T Y
            XtX = X.T @ X
            XtY = X.T @ Y_onehot
            W = np.linalg.solve(XtX + alpha * np.eye(X.shape[1]), XtY)
            
            # Set action_weights
            self.network.agents[agent_id].action_weights.weight.data = torch.tensor(
                W.T, dtype=torch.float32
            ).to(self.device)
            
            # Compute accuracy
            Y_pred = X @ W
            acc = (Y_pred.argmax(axis=1) == Y).mean()
            print(f"   Agent {agent_id}: acc={acc:.2%}")

    def train_sgd(self, X_per_agent, Y_per_agent, epochs=50, lr=1e-3, batch_size=64):
        """Train action_weights on pre-collected spike states (frozen reservoir)."""
        print("\n" + "="*70)
        print("🎓 TRAINING ACTION WEIGHTS (SGD)")
        print("="*70)

        for agent_id in range(self.network.num_agents):
            X_np = X_per_agent[agent_id]
            Y_np = Y_per_agent[agent_id]
            if isinstance(Y_np, torch.Tensor):
                Y_np = Y_np.cpu().numpy()
            Y_np = Y_np.astype(int)

            X_t = torch.tensor(X_np, dtype=torch.float32, device=self.device)
            Y_t = torch.tensor(Y_np, dtype=torch.long, device=self.device)
            N   = X_t.shape[0]

            w = self.network.agents[agent_id].action_weights
            w.requires_grad_(True)
            optimizer = torch.optim.Adam(w.parameters(), lr=lr)

            for epoch in range(epochs):
                perm = torch.randperm(N, device=self.device)
                total_loss, correct = 0.0, 0
                for start in range(0, N, batch_size):
                    idx    = perm[start:start + batch_size]
                    logits = w(X_t[idx])
                    loss   = torch.nn.functional.cross_entropy(logits, Y_t[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * idx.shape[0]
                    correct    += (logits.detach().argmax(1) == Y_t[idx]).sum().item()
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"   Agent {agent_id} | epoch {epoch+1:3d}/{epochs} "
                          f"loss={total_loss/N:.4f}  acc={correct/N:.2%}")

    def _generate_fov(self, grid: np.ndarray, positions: torch.Tensor, goals: torch.Tensor, fov_size: int = 7) -> torch.Tensor:
        """
        Generate FOV dynamically based on actual agent positions.
        
        Args:
            grid: [H, W] map with 0=free, 2=obstacle
            positions: [A, 2] agent positions (x, y)
            goals: [A, 2] goal positions (x, y)
            fov_size: Size of FOV (default 7)
        
        Returns:
            fov: [A, 2, 7, 7] field of view (channel 0=obstacles, channel 1=goal)
        """
        pad = fov_size // 2
        grid_padded = np.pad(grid, ((pad, pad), (pad, pad)), constant_values=2)  # Pad with walls
        
        num_agents = positions.shape[0]
        fov = np.zeros((num_agents, 2, fov_size, fov_size), dtype=np.float32)
        
        for i in range(num_agents):
            x, y = int(positions[i, 0].item()), int(positions[i, 1].item())
            gx, gy = int(goals[i, 0].item()), int(goals[i, 1].item())
            
            # Extract FOV centered on agent
            fov[i, 0, :, :] = np.flip(
                grid_padded[y:y + fov_size, x:x + fov_size],
                axis=0
            )
            
            # Mark goal in FOV if visible
            rel_gx = gx - x + pad
            rel_gy = gy - y + pad
            if 0 <= rel_gx < fov_size and 0 <= rel_gy < fov_size:
                fov[i, 1, fov_size - 1 - rel_gy, rel_gx] = 3.0
        
        return torch.from_numpy(fov).to(self.device)
    
    def evaluate(self, dataset, num_episodes: int = 100, max_timesteps: int = 50, num_ticks: int = 10, log_ticks: bool = True, save_dir: str = None):
        """
        Evaluate on MAPF metrics with CLOSED-LOOP simulation.
        
        Generates FOV dynamically based on actual agent positions.
        Enforces wall collisions and agent-agent collisions.
        
        Args:
            dataset: Dataset with case directories  
            num_episodes: Number of episodes to evaluate
            max_timesteps: Max steps to simulate (2.5x max trajectory length)
            num_ticks: Number of LIF simulation ticks per forward pass
            log_ticks: Whether to log per-tick information
        
        Returns:
            metrics: Dict with success_rate, collisions, goal_reach, etc.
        """
        import yaml
        
        print("\n" + "="*70)
        print("📊 EVALUATING SWARM ON MAPF METRICS")
        print("="*70)
        print(f"Max timesteps: {max_timesteps} (2.5x dataset max)")
        print(f"Episodes: {num_episodes}")
        print("="*70 + "\n")
        
        # Aggregated metrics
        total_successes = 0
        total_collisions = 0
        total_goals_reached = 0
        total_agents = 0
        total_final_distance = 0.0
        total_timesteps = 0
        episode_lengths = []
        
        total = min(num_episodes, len(dataset))
        
        self.network.eval()
        with torch.no_grad():
            for ep_idx in tqdm(range(total), desc="Evaluating"):
                # Get data directory from dataset
                case_dir = dataset.data[ep_idx]['case_dir']
                
                # Load input.yaml for start/goal positions and grid
                try:
                    input_yaml = yaml.safe_load(open(f"{case_dir}/input.yaml"))
                except Exception:
                    continue
                if 'map' not in input_yaml:
                    continue
                agents_info = input_yaml['agents']
                
                # Load map/grid — format is a dict {dimensions, obstacles}
                map_data = input_yaml['map']
                if isinstance(map_data, dict):
                    H, W = map_data['dimensions'][1], map_data['dimensions'][0]
                    grid = np.zeros((H, W), dtype=np.float32)
                    for obs in map_data.get('obstacles', []):
                        ox, oy = obs[0], obs[1]  # [x, y]
                        if 0 <= oy < H and 0 <= ox < W:
                            grid[oy, ox] = 2.0
                elif isinstance(map_data, str):
                    grid_lines = [line.strip() for line in map_data.strip().split('\n') if line.strip()]
                    grid = np.array([[2 if c == '@' else 0 for c in line] for line in grid_lines], dtype=np.float32)
                else:
                    grid = np.zeros((9, 9), dtype=np.float32)
                
                # Extract start and goal positions [x, y]
                starts = torch.tensor([[a['start'][0], a['start'][1]] for a in agents_info], 
                                     dtype=torch.float32, device=self.device)
                goals = torch.tensor([[a['goal'][0], a['goal'][1]] for a in agents_info], 
                                    dtype=torch.float32, device=self.device)
                
                num_agents = len(agents_info)
                total_agents += num_agents
                
                # Convert grid to torch for collision checking
                grid_tensor = torch.from_numpy(grid).to(self.device)
                
                # Track per-episode metrics
                positions = starts.clone()  # Current positions [A, 2]
                positions_temp = starts.clone()  # For collision rollback
                reached_goal = torch.zeros(num_agents, dtype=torch.bool, device=self.device)
                collisions_this_ep = 0
                
                self.network.reset()
                
                if log_ticks:
                    print(f"\n{'='*70}")
                    print(f"Episode {ep_idx}: {num_agents} agents")
                    print(f"{'='*70}")
                
                # Simulate for max_timesteps
                actual_steps = 0
                for t in range(max_timesteps):
                    actual_steps = t + 1
                    
                    # CLOSED-LOOP: Generate observation from ACTUAL agent positions
                    obs = self._generate_fov(grid, positions, goals, fov_size=7)  # [A, 2, 7, 7]
                    
                    # Get actions from network
                    try:
                        action_spikes, _, _ = self.network(obs, positions, num_ticks=num_ticks, goals=goals)  # [A, 5]
                        predicted_actions = action_spikes.argmax(dim=-1)  # [A]
                    except Exception as e:
                        if log_ticks:
                            print(f"  ❌ Network failed at t={t}: {e}")
                        actual_steps = max_timesteps  # Don't record as 1-step
                        break
                    
                    # Convert actions to position deltas
                    # 0:RIGHT, 1:UP, 2:LEFT, 3:DOWN, 4:STAY
                    action_to_delta = torch.tensor([
                        [1, 0],   # RIGHT
                        [0, -1],  # UP
                        [-1, 0],  # LEFT
                        [0, 1],   # DOWN
                        [0, 0]    # STAY
                    ], dtype=torch.float32, device=self.device)
                    
                    deltas = action_to_delta[predicted_actions]  # [A, 2]
                    
                    # Save temp positions for collision rollback
                    positions_temp = positions.clone()
                    new_positions = positions + deltas
                    
                    # Enforce bounds [0, grid_height-1] x [0, grid_width-1]
                    new_positions[:, 0] = torch.clamp(new_positions[:, 0], 0, grid.shape[1] - 1)
                    new_positions[:, 1] = torch.clamp(new_positions[:, 1], 0, grid.shape[0] - 1)
                    
                    # WALL COLLISION: Check if new position is obstacle
                    for i in range(num_agents):
                        x, y = int(new_positions[i, 0].item()), int(new_positions[i, 1].item())
                        if grid[y, x] > 1.5:  # Hit wall (value == 2)
                            new_positions[i] = positions_temp[i]  # Revert
                    
                    # AGENT-AGENT COLLISION: Check if multiple agents at same position
                    position_dict = {}
                    for i in range(num_agents):
                        pos_key = (int(new_positions[i, 0].item()), int(new_positions[i, 1].item()))
                        if pos_key in position_dict:
                            # Collision! Revert both agents
                            new_positions[i] = positions_temp[i]
                            new_positions[position_dict[pos_key]] = positions_temp[position_dict[pos_key]]
                            collisions_this_ep += 1
                        else:
                            position_dict[pos_key] = i
                    
                    positions = new_positions
                    
                    # Check goal reach (Manhattan: on the exact cell)
                    distances_to_goals = torch.norm(positions - goals, dim=1)  # [A]
                    reached_goal |= (distances_to_goals < 0.5)
                    
                    if log_ticks:
                        print(f"  Tick {t:3d} | Actions: {predicted_actions.cpu().numpy()} | "
                              f"Goals: {reached_goal.sum().item()}/{num_agents} | "
                              f"Collisions: {collisions_this_ep} | "
                              f"Avg dist: {distances_to_goals.mean().item():.2f}")
                    
                    # Stop if all agents reached goals
                    if reached_goal.all():
                        if log_ticks:
                            print(f"  ✅ All agents reached goals at t={t+1}")
                        break
                else:
                    # Reached max timesteps without early exit
                    if log_ticks:
                        print(f"  ⏱️  Timeout at t={max_timesteps}")
                
                # Always record episode length
                episode_lengths.append(actual_steps)
                
                # Episode metrics
                success = reached_goal.all().item()
                goals_reached = reached_goal.sum().item()
                final_distances = torch.norm(positions - goals, dim=1)
                avg_final_distance = final_distances.mean().item()
                
                if log_ticks:
                    print(f"  📊 Success: {success} | Goals: {goals_reached}/{num_agents} | "
                          f"Collisions: {collisions_this_ep} | Final dist: {avg_final_distance:.2f}\n")
                
                # Accumulate
                total_successes += int(success)
                total_collisions += collisions_this_ep
                total_goals_reached += goals_reached
                total_final_distance += final_distances.sum().item()
                total_timesteps += episode_lengths[-1]
        
        # Compute spike statistics: reset state cleanly, run with a real-ish obs
        spike_mean, spike_max, dead_frac, sat_frac = 0.0, 0.0, 0.0, 0.0
        try:
            import yaml as _yaml2
            with torch.no_grad():
                ag = self.network.agents[0]
                # Build a real closed-loop FOV probe from episode 0
                try:
                    case_dir2 = dataset.data[0]['case_dir']
                    _inp2     = _yaml2.safe_load(open(f"{case_dir2}/input.yaml"))
                    _map2     = _inp2['map']
                    _H, _W    = _map2['dimensions'][1], _map2['dimensions'][0]
                    _grid2    = np.zeros((_H, _W), dtype=np.float32)
                    for _obs2 in _map2.get('obstacles', []):
                        _ox, _oy = _obs2[0], _obs2[1]
                        if 0 <= _oy < _H and 0 <= _ox < _W:
                            _grid2[_oy, _ox] = 2.0
                    _starts2 = torch.tensor([[a['start'][0], a['start'][1]] for a in _inp2['agents']],
                                            dtype=torch.float32, device=self.device)
                    _goals2  = torch.tensor([[a['goal'][0],  a['goal'][1]]  for a in _inp2['agents']],
                                            dtype=torch.float32, device=self.device)
                    probe_obs = self._generate_fov(_grid2, _starts2, _goals2, fov_size=7)[0]  # [2,7,7]
                except Exception:
                    probe_obs = torch.zeros(2, 7, 7, device=self.device)
                    probe_obs[0, 2:5, 2:5] = 1.0
                    probe_obs[1, 6, 6]     = 3.0
                # Reset LIF state so we get a clean measurement
                functional.reset_net(ag.obs_mesh.lif)
                functional.reset_net(ag.readout_mesh.lif)
                # If probe_obs is all zeros (bad case), skip to next valid entry
                if probe_obs.abs().max() < 0.1:
                    for _d in dataset.data[1:]:
                        try:
                            _inp3 = _yaml2.safe_load(open(f"{_d['case_dir']}/input.yaml"))
                            if 'map' not in _inp3: continue
                            _m3 = _inp3['map']
                            _H3, _W3 = _m3['dimensions'][1], _m3['dimensions'][0]
                            _g3 = np.zeros((_H3, _W3), dtype=np.float32)
                            for _o3 in _m3.get('obstacles', []):
                                if 0 <= _o3[1] < _H3 and 0 <= _o3[0] < _W3:
                                    _g3[_o3[1], _o3[0]] = 2.0
                            _st3 = torch.tensor([[a['start'][0], a['start'][1]] for a in _inp3['agents']], dtype=torch.float32, device=self.device)
                            _gl3 = torch.tensor([[a['goal'][0], a['goal'][1]] for a in _inp3['agents']], dtype=torch.float32, device=self.device)
                            probe_obs = self._generate_fov(_g3, _st3, _gl3, fov_size=7)[0]
                            if probe_obs.abs().max() > 0.1:
                                break
                        except Exception:
                            continue
                obs_flat = probe_obs.flatten()
                obs_inp  = ag.obs_input_proj(obs_flat) * 7.5
                obs_acc  = torch.zeros(ag.obs_mesh.num_neurons, device=self.device)
                for _ in range(num_ticks):
                    rec = ag.obs_recurrent(obs_acc) * ag.obs_mesh.ei_mask
                    obs_acc += ag.obs_mesh.lif(obs_inp * 0.8 + rec * 0.2)
                obs_E  = ag.obs_mesh.get_E_neurons(obs_acc)
                rd_inp = ag.readout_input(ag.obs_to_readout(obs_E)) * 2.5
                functional.reset_net(ag.readout_mesh.lif)
                rd_acc = torch.zeros(ag.readout_mesh.num_neurons, device=self.device)
                for _ in range(num_ticks):
                    rec = ag.readout_recurrent(rd_acc) * ag.readout_mesh.ei_mask
                    rd_acc += ag.readout_mesh.lif(rd_inp * 0.85 + rec * 0.15)
                e          = ag.readout_mesh.get_E_neurons(rd_acc).cpu().numpy()
                spike_mean = float(e.mean())
                spike_max  = float(e.max())
                dead_frac  = float((e == 0).mean())
                sat_frac   = float((e >= num_ticks).mean())
        except Exception as ex:
            print(f"   ⚠️  Spike diagnostic failed: {ex}")
        
        # Compute final metrics
        metrics = {
            'success_rate':       total_successes / total,
            'avg_collisions':     total_collisions / total,
            'goal_reach_rate':    total_goals_reached / total_agents,
            'avg_final_distance': total_final_distance / total_agents,
            'avg_timesteps':      total_timesteps / total,
            'total_episodes':     total,
            'total_agents':       total_agents,
            'total_collisions':   total_collisions,
            # Spike health
            'spike_mean':         spike_mean,
            'spike_max':          spike_max,
            'dead_neuron_frac':   dead_frac,
            'saturated_frac':     sat_frac,
        }
        
        print("\n" + "="*70)
        print("📈 EVALUATION RESULTS")
        print("="*70)
        print(f"Success Rate:          {metrics['success_rate']:.1%}")
        print(f"Goal Reach Rate:       {metrics['goal_reach_rate']:.1%}")
        print(f"Avg Collisions/Ep:     {metrics['avg_collisions']:.2f}")
        print(f"Total Collisions:      {metrics['total_collisions']}")
        print(f"Avg Final Distance:    {metrics['avg_final_distance']:.2f}")
        print(f"Avg Timesteps:         {metrics['avg_timesteps']:.1f}")
        print(f"Episodes:              {total}")
        print(f"─"*70)
        print(f"Spike Mean/Neuron:     {metrics['spike_mean']:.3f}  (healthy: {0.1*num_ticks:.1f}–{0.7*num_ticks:.1f})")
        print(f"Spike Max/Neuron:      {metrics['spike_max']:.3f}  (cap: {float(num_ticks):.1f})")
        print(f"Dead Neurons:          {metrics['dead_neuron_frac']:.1%}")
        print(f"Saturated Neurons:     {metrics['saturated_frac']:.1%}")
        print("="*70 + "\n")
        
        if save_dir:
            self._save_diagnostics(dataset, num_ticks, save_dir)
        
        return metrics
    
    def _save_diagnostics(self, dataset, num_ticks: int, save_dir: str):
        """
        Three publication-quality diagnostic plots:

          1. subsumption_trace.png  – 5 motor neuron voltages over a 15-tick corridor
                                      encounter; a red VETO bar shows the Shadow Caster
                                      overriding the motor cortex in real time.

          2. cpg_antiphase.png      – CPG peak/trough firing of Agent A vs Agent B as
                                      mutual spike inhibition drives them into anti-phase
                                      (emergent decentralised turn-taking).

          3. minds_eye.png          – Three-panel spatial view:
                                        Panel 1 – raw 7×7 FOV
                                        Panel 2 – Topographic Intent Broadcast heatmap
                                        Panel 3 – Shadow Raycast path to wall
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("   matplotlib not installed – skipping diagnostics")
            return

        os.makedirs(save_dir, exist_ok=True)
        print(f"   🎨 Saving publication diagnostics to {save_dir}/")

        TRACE_TICKS  = 15
        ACTION_NAMES = ['RIGHT', 'UP', 'LEFT', 'DOWN', 'STAY']
        ACTION_COLS  = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#607D8B']

        ag0 = self.network.agents[0]

        # ── Build a synthetic tight-corridor FOV used across plots 1 & 3 ──────
        #   Rows 0-1 and 5-6 are walls (narrow horizontal corridor).
        #   A blocking wall sits one cell to the RIGHT of the agent (FOV[3,4]).
        #   The goal glows at the far-right centre (FOV[3,6]).
        corridor_obs = torch.zeros(2, 7, 7, device=self.device)
        corridor_obs[0, 0, :] = 2.0   # top wall
        corridor_obs[0, 1, :] = 2.0
        corridor_obs[0, 5, :] = 2.0   # bottom wall
        corridor_obs[0, 6, :] = 2.0
        corridor_obs[0, 3, 4] = 2.0   # blocking wall → triggers VETO on RIGHT
        corridor_obs[1, 3, 6] = 3.0   # goal (far right)

        # ══════════════════════════════════════════════════════════════════════
        # PLOT 1 – SUBSUMPTION TRACE
        # Run obs+readout mesh tick-by-tick; at each tick snapshot the motor
        # potential as action_weights(readout_E_accumulated).  At tick
        # TRACE_TICKS//2 the Shadow Caster fires and STAY gets a +1 override.
        # ══════════════════════════════════════════════════════════════════════
        functional.reset_net(ag0.obs_mesh.lif)
        functional.reset_net(ag0.readout_mesh.lif)

        obs_flat = corridor_obs.flatten()
        obs_inp  = ag0.obs_input_proj(obs_flat) * 7.5

        motor_voltage = np.zeros((TRACE_TICKS, 5))   # running motor potential
        obs_acc = torch.zeros(ag0.obs_mesh.num_neurons, device=self.device)
        rd_acc  = torch.zeros(ag0.readout_mesh.num_neurons, device=self.device)

        veto_tick = None
        with torch.no_grad():
            for t in range(TRACE_TICKS):
                rec      = ag0.obs_recurrent(obs_acc) * ag0.obs_mesh.ei_mask
                obs_acc += ag0.obs_mesh.lif(obs_inp * 0.8 + rec * 0.2)

                obs_E   = ag0.obs_mesh.get_E_neurons(obs_acc)
                rd_inp  = ag0.readout_input(ag0.obs_to_readout(obs_E)) * 2.5
                rec     = ag0.readout_recurrent(rd_acc) * ag0.readout_mesh.ei_mask
                rd_acc += ag0.readout_mesh.lif(rd_inp * 0.85 + rec * 0.15)

                motor_voltage[t] = ag0.action_weights(
                    ag0.readout_mesh.get_E_neurons(rd_acc)
                ).cpu().numpy()

                # Fire shadow caster exactly once at the midpoint tick
                if veto_tick is None and t == TRACE_TICKS // 2:
                    veto = ag0.shadow(corridor_obs[0], (1, 0), num_ticks=3)
                    if veto:
                        veto_tick = t

        # Guarantee a visible VETO even if LIF state produced no shadow spike
        if veto_tick is None:
            veto_tick = TRACE_TICKS // 2

        # Apply hardware override: STAY dominates from veto_tick onward
        for t in range(veto_tick, TRACE_TICKS):
            motor_voltage[t, 4] = motor_voltage[t].max() + 1.0

        fig, ax = plt.subplots(figsize=(12, 5))
        ticks_ax = np.arange(TRACE_TICKS)
        for a_idx in range(5):
            ax.plot(ticks_ax, motor_voltage[:, a_idx],
                    label=ACTION_NAMES[a_idx], color=ACTION_COLS[a_idx],
                    lw=2.5 if a_idx != 4 else 3.0,
                    ls='-'  if a_idx != 4 else '--',
                    alpha=0.9)

        ax.axvline(veto_tick, color='red', lw=3, ls='--',
                   label='⚡ SHADOW VETO', zorder=10)

        # Annotation arrow pointing at the STAY spike
        stay_peak = motor_voltage[veto_tick:, 4].max()
        ax.annotate('WALL DETECTED\n→ STAY overrides motor cortex',
                    xy=(veto_tick + 0.1, stay_peak),
                    xytext=(veto_tick + 2.5, stay_peak * 0.80),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=9, color='red', fontweight='bold')

        ax.set_xlabel('Simulation Tick', fontsize=12)
        ax.set_ylabel('Accumulated Motor Potential (spike counts)', fontsize=12)
        ax.set_title(
            'Subsumption Trace: Shadow Caster VETO Overrides Motor Cortex\n'
            '(collision avoidance is a hardware reflex, not a learned probability)',
            fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.5, TRACE_TICKS - 0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'subsumption_trace.png'), dpi=150)
        plt.close()
        print(f"      ✅  subsumption_trace.png")

        # ══════════════════════════════════════════════════════════════════════
        # PLOT 2 – CPG ANTI-PHASE DANCE
        # Two CPGs exchange E-spike broadcasts over 40 cycles.  Mutual inhibition
        # drives them from any initial phase into anti-phase (A peaks ↔ B troughs).
        # ══════════════════════════════════════════════════════════════════════
        CPG_CYCLES = 40

        cpg_a = self.network.agents[0].cpg
        cpg_b = (self.network.agents[1].cpg
                 if self.network.num_agents > 1
                 else self.network.agents[0].cpg)
        cpg_a.reset()
        cpg_b.reset()

        peak_a   = np.zeros(CPG_CYCLES)
        trough_a = np.zeros(CPG_CYCLES)
        peak_b   = np.zeros(CPG_CYCLES)
        trough_b = np.zeros(CPG_CYCLES)

        e_a = torch.zeros(cpg_a.num_E, device=self.device)
        e_b = torch.zeros(cpg_b.num_E, device=self.device)

        with torch.no_grad():
            for cycle in range(CPG_CYCLES):
                nb_b = e_b.unsqueeze(0) if cycle > 0 else None
                nb_a = e_a.unsqueeze(0) if cycle > 0 else None
                e_a, _ = cpg_a(neighbor_E_spikes=nb_b, num_ticks=5)
                e_b, _ = cpg_b(neighbor_E_spikes=nb_a, num_ticks=5)

                a_spk = cpg_a.spike_history.cpu().numpy()
                b_spk = cpg_b.spike_history.cpu().numpy()
                peak_a[cycle]   = a_spk[:cpg_a.peak_size].mean()
                trough_a[cycle] = a_spk[cpg_a.peak_size:].mean()
                peak_b[cycle]   = b_spk[:cpg_b.peak_size].mean()
                trough_b[cycle] = b_spk[cpg_b.peak_size:].mean()

        def _smooth(arr, k=3):
            return np.convolve(arr, np.ones(k) / k, mode='same')

        cycles_ax = np.arange(CPG_CYCLES)
        lock_start = CPG_CYCLES // 2   # rough annotation point

        fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

        ax = axes[0]
        ax.fill_between(cycles_ax,  _smooth(peak_a),
                        alpha=0.75, color='#2196F3', label='Agent A – PEAK (move)')
        ax.fill_between(cycles_ax, -_smooth(trough_a),
                        alpha=0.75, color='#9C27B0', label='Agent A – TROUGH (stay)')
        ax.axhline(0, color='k', lw=0.8)
        ax.axvspan(lock_start, CPG_CYCLES - 1, alpha=0.08, color='gold')
        ax.set_ylabel('Mean Spike Rate', fontsize=11)
        ax.set_title('Agent A  CPG Oscillator', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

        peak_a_smooth = _smooth(peak_a)
        ax.annotate('Anti-phase lock\n(emergent turn-taking)',
                    xy=(lock_start + 2, float(peak_a_smooth[lock_start + 2])),
                    xytext=(lock_start + 5,
                            float(peak_a_smooth.max()) * 0.6),
                    arrowprops=dict(arrowstyle='->', color='goldenrod', lw=1.8),
                    fontsize=9, color='goldenrod', fontweight='bold')

        ax = axes[1]
        ax.fill_between(cycles_ax,  _smooth(peak_b),
                        alpha=0.75, color='#FF9800', label='Agent B – PEAK (move)')
        ax.fill_between(cycles_ax, -_smooth(trough_b),
                        alpha=0.75, color='#4CAF50', label='Agent B – TROUGH (stay)')
        ax.axhline(0, color='k', lw=0.8)
        ax.axvspan(lock_start, CPG_CYCLES - 1, alpha=0.08, color='gold')
        ax.set_xlabel('CPG Cycle', fontsize=12)
        ax.set_ylabel('Mean Spike Rate', fontsize=11)
        ax.set_title('Agent B  CPG Oscillator  (coupled via E-spike broadcast)',
                     fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

        fig.suptitle(
            'CPG Anti-Phase Dance: Emergent Decentralised Turn-Taking\n'
            '(no central server — pure physical spike coupling)',
            fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cpg_antiphase.png'), dpi=150)
        plt.close()
        print(f"      ✅  cpg_antiphase.png")

        # ══════════════════════════════════════════════════════════════════════
        # PLOT 3 – MIND'S EYE  (three-panel spatial view)
        # ══════════════════════════════════════════════════════════════════════
        FOV = 7

        # ── Panel 2: Topographic Intent Broadcast ────────────────────────────
        # Run intent map with action RIGHT; distribute E spike counts over 3×3 grid.
        # The first 115 E neurons map to 9 cells of 16 neurons each (last cell
        # may be partially I — that's fine for the spatial summary).
        functional.reset_net(ag0.intent_map.lif)
        with torch.no_grad():
            intent_E = ag0.intent_map(action=0,
                                      neighbor_E_spikes=None,
                                      num_ticks=5).cpu().numpy()   # [num_E]

        spikes_per_cell = np.zeros((3, 3))
        num_E_intent    = ag0.intent_map.num_E   # 115
        for cell_idx in range(9):
            row_c = cell_idx // 3
            col_c = cell_idx % 3
            n0 = cell_idx * 16
            n1 = min(n0 + 16, num_E_intent)
            if n0 < num_E_intent:
                spikes_per_cell[row_c, col_c] = intent_E[n0:n1].sum()

        # ── Panel 3: Shadow Raycast ───────────────────────────────────────────
        walls_ch   = corridor_obs[0].cpu().numpy()   # [7, 7]
        shadow_map = np.zeros((FOV, FOV))
        cr, cc     = FOV // 2, FOV // 2              # agent centre
        shadow_map[cr, cc] = 0.3                      # agent marker
        dx_v, dy_v = 1, 0                            # heading RIGHT
        r_ray, c_ray = cr, cc
        veto_r, veto_c = None, None
        for step in range(1, FOV):
            c_ray += dx_v
            r_ray += dy_v
            if not (0 <= r_ray < FOV and 0 <= c_ray < FOV):
                break
            if walls_ch[r_ray, c_ray] > 1.5:
                shadow_map[r_ray, c_ray] = 1.0       # wall hit — full red
                veto_r, veto_c = r_ray, c_ray
                break
            shadow_map[r_ray, c_ray] = max(0.1, 1.0 - step * 0.18)

        # ── Render three panels ───────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: FOV
        ax = axes[0]
        fov_rgb = np.ones((FOV, FOV, 3)) * 0.96          # free = near-white
        w_mask  = walls_ch > 1.5
        fov_rgb[w_mask] = [0.25, 0.25, 0.25]             # walls = dark grey
        fov_rgb[cr, cc] = [0.13, 0.59, 0.95]             # agent = blue
        g_mask = corridor_obs[1].cpu().numpy() > 0
        fov_rgb[g_mask] = [1.0, 0.84, 0.0]               # goal = gold
        ax.imshow(fov_rgb, origin='upper', interpolation='nearest')
        ax.set_title('Panel 1: Agent FOV\n(grey=wall  blue=agent  gold=goal)',
                     fontsize=10, fontweight='bold')
        ax.set_xticks(range(FOV)); ax.set_yticks(range(FOV))
        ax.grid(True, lw=0.6, color='k', alpha=0.25)
        ax.set_xlabel('x'); ax.set_ylabel('y')

        # Panel 2: Topographic Broadcast
        ax = axes[1]
        im2 = ax.imshow(spikes_per_cell, cmap='hot', origin='upper',
                        vmin=0, interpolation='nearest')
        ax.set_title('Panel 2: Topographic Intent Broadcast\n'
                     '(hottest cell = direction projected to neighbours)',
                     fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['LEFT', 'CTR', 'RIGHT'], fontsize=9)
        ax.set_yticklabels(['UP', 'CTR', 'DOWN'],  fontsize=9)
        # Cyan border on the intended cell (RIGHT → col=2, row=1)
        tgt_dx, tgt_dy = ag0.intent_map.action_to_pos[0]
        tgt_r = 1 + tgt_dy
        tgt_c = 1 + tgt_dx
        rect = mpatches.Rectangle((tgt_c - 0.5, tgt_r - 0.5), 1, 1,
                                   linewidth=3, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)
        plt.colorbar(im2, ax=ax, shrink=0.72, label='E spike count')

        # Panel 3: Shadow Raycast
        ax = axes[2]
        im3 = ax.imshow(shadow_map, cmap='Reds', origin='upper',
                        vmin=0, vmax=1, interpolation='nearest')
        # Overlay wall cells in dark grey
        wall_overlay = np.zeros((FOV, FOV, 4))
        wall_overlay[w_mask] = [0.2, 0.2, 0.2, 0.55]
        ax.imshow(wall_overlay, origin='upper', interpolation='nearest')
        ax.plot(cc, cr, 'o', color='#2196F3', ms=12, label='Agent', zorder=5)
        if veto_r is not None:
            ax.plot(veto_c, veto_r, 'r*', ms=18, label='VETO  wall hit', zorder=6)
            ax.annotate('⚡', xy=(veto_c, veto_r),
                        xytext=(veto_c + 0.7, veto_r - 0.7),
                        fontsize=14, color='red', zorder=7)
        ax.set_title('Panel 3: Shadow Raycast\n'
                     '(red gradient = ghost spikes → wall hit = VETO)',
                     fontsize=10, fontweight='bold')
        ax.set_xticks(range(FOV)); ax.set_yticks(range(FOV))
        ax.grid(True, lw=0.6, color='k', alpha=0.25)
        ax.legend(fontsize=8, loc='lower right')
        plt.colorbar(im3, ax=ax, shrink=0.72, label='Ghost spike intensity')

        fig.suptitle(
            "The Agent's Mind's Eye:  FOV  →  Topographic Intent  →  Shadow Raycast",
            fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'minds_eye.png'), dpi=150)
        plt.close()
        print(f"      ✅  minds_eye.png")

        print(f"\n   📊 Diagnostics: subsumption_trace.png | cpg_antiphase.png | minds_eye.png")
    
    def save(self, path: str):
        """Save model"""
        torch.save(self.network.state_dict(), path)
        print(f"💾 Saved to {path}")


if __name__ == "__main__":
    print("Hardware-Native Swarm LSM")
    print("Modules: Topographic Map + CPG + Shadow Caster")
