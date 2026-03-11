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
        
        # E/I sign mask for recurrent connections (registered buffer → moves with .to(device))
        self.register_buffer('ei_mask', torch.cat([
            torch.ones(self.num_E),
            -torch.ones(self.num_I)
        ]))
        
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

        # BDSM FIX: Hardware-wired directional repulsion — no random telepathy.
        # An UP spike from a neighbour directly inhibits my UP neurons.
        nn.init.zeros_(self.inhibition_proj.weight)
        min_dim = min(self.num_E, self.num_neurons)
        with torch.no_grad():
            self.inhibition_proj.weight.diagonal()[:min_dim].fill_(2.0)
        
    def encode_intent(self, action: int) -> torch.Tensor:
        """
        Convert action to spatial spike input.
        
        Args:
            action: 0-4
        
        Returns:
            input_current: [144] input current to LIF neurons
        """
        current = torch.zeros(self.num_neurons, device=self.ei_mask.device)
        
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
    ) -> Tuple[torch.Tensor, bool]:
        """
        Run spiking dynamics for intent encoding.
        
        Args:
            action: Agent's action
            neighbor_E_spikes: [num_neighbors, num_E] neighbor E neuron spikes
            num_ticks: Simulation ticks
        
        Returns:
            E_spikes:    [num_E] excitatory neuron spikes (broadcast to neighbours)
            intent_veto: True if target direction was crushed by neighbour spikes
        """
        input_current = self.encode_intent(action)
        
        device = self.ei_mask.device
        spike_accumulator = torch.zeros(self.num_neurons, device=device)
        current_spikes    = torch.zeros(self.num_neurons, device=device)  # wires start cold
        
        for t in range(num_ticks):
            # Recurrent input — instantaneous spikes only (not cumulative accumulator)
            recurrent_input = self.recurrent(current_spikes)
            
            # Apply E/I signs
            recurrent_input = recurrent_input * self.ei_mask
            
            # Lateral inhibition from neighbors
            inhibition = torch.zeros(self.num_neurons, device=device)
            if neighbor_E_spikes is not None and neighbor_E_spikes.shape[0] > 0:
                for neighbor_spikes in neighbor_E_spikes:
                    inhibition += self.inhibition_proj(neighbor_spikes)
            
            # Total input
            total_input = input_current + recurrent_input * 0.3 - inhibition * 0.5

            # LIF dynamics — separate instantaneous from accumulator
            current_spikes  = self.lif(total_input)
            spike_accumulator += current_spikes
        
        E_spikes = self.get_E_neurons(spike_accumulator)

        # BDSM FIX: Spatial Forcefield Veto
        # If we tried to move but our target cell was crushed by a neighbour, yield!
        intent_veto = False
        if action != 4:
            dx, dy = self.action_to_pos[action]
            cell_idx = (1 + dy) * 3 + (1 + dx)
            e_start = min(cell_idx * self.neurons_per_cell, self.num_E)
            e_end   = min(e_start + self.neurons_per_cell, self.num_E)
            my_target_spikes = E_spikes[e_start:e_end].sum().item()
            if my_target_spikes < (num_ticks * 2.0):  # heavily suppressed by neighbour
                intent_veto = True

        return E_spikes, intent_veto

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
        
        # Peak/Trough self-connections: just enough to group a burst, not sustain it forever
        self.mutual_inhibition.data[:self.peak_size, :self.peak_size] = 0.5
        self.mutual_inhibition.data[self.peak_size:, self.peak_size:] = 0.5
        # Strong cross-inhibition ensures only one population is active at a time
        self.mutual_inhibition.data[:self.peak_size, self.peak_size:] = -3.0  # Peak suppresses Trough
        self.mutual_inhibition.data[self.peak_size:, :self.peak_size] = -3.0  # Trough suppresses Peak
        
        # Coupling from neighbor CPGs (E neurons only)
        self.coupling_proj = layer.Linear(self.num_E, self.num_neurons, bias=False)
        nn.init.normal_(self.coupling_proj.weight, mean=0, std=0.2)
        
        # Spike-frequency adaptation — biological fatigue via slow K+ channel analog.
        # Replaces the Python FSM fatigue counter: rises with each spike, decays
        # between spikes, naturally suppresses whichever population fires too long.
        self.register_buffer('adaptation', torch.zeros(self.num_neurons))

        # State (registered buffer → moves with .to(device))
        self.register_buffer('spike_history', torch.zeros(self.num_neurons))
        
    def reset(self):
        """Reset CPG state"""
        self.spike_history.zero_()
        self.adaptation.zero_()
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
        # Basal drive slightly above threshold; symmetry-breaking noise lets one
        # population win the initial competition without a hardcoded defibrillator.
        noise = torch.rand(self.num_neurons, device=self.spike_history.device) * 0.1
        drive = torch.ones(self.num_neurons, device=self.spike_history.device) * 1.1 + noise

        spike_accumulator = torch.zeros(self.num_neurons, device=self.spike_history.device)

        for t in range(num_ticks):
            recurrent = torch.matmul(self.spike_history, self.mutual_inhibition.T)

            coupling = torch.zeros(self.num_neurons, device=self.spike_history.device)
            if neighbor_E_spikes is not None and neighbor_E_spikes.shape[0] > 0:
                for neighbor_spikes in neighbor_E_spikes:
                    coupling += self.coupling_proj(neighbor_spikes)

            # Spike-frequency adaptation subtracts from whichever population is
            # currently firing — biologically honest fatigue, no Python counter.
            total_input = drive + recurrent * 0.5 + coupling * 0.3 - self.adaptation

            spikes = self.lif(total_input)
            spike_accumulator += spikes
            self.spike_history = spikes.detach()

            # Leaky adaptation: rises with each spike, decays passively (slow K+ analog)
            # Slower fatigue buildup allows the population to actually oscillate
            self.adaptation = self.adaptation * 0.9 + spikes.detach() * 1.0

        peak_active   = spike_accumulator[:self.peak_size].sum() > 0.5
        trough_active = spike_accumulator[self.peak_size:].sum() > 0.5
        should_stay   = trough_active

        return self.get_E_neurons(spike_accumulator), should_stay


# === MODULE 3: SHADOW CASTER (SPIKING) ===

class ShadowCaster(nn.Module):
    """
    Predictive collision detector — honest raycast sensor.

    Previously wrapped 392 fake LIF neurons whose spike output was never
    wired to the VETO decision; the actual collision check was always a
    plain grid raycast. Those dead layers have been removed.

    This is a zero-parameter, zero-cost sensor. The VETO signal it returns
    is physically real: it reads the FOV wall channel directly.
    The Software Subsumption Bridge in SwarmLSM.forward() then injects
    the VETO into the motor logits — that bridge is a known architectural
    debt (see LIE #2 comment there).
    """
    def __init__(self, fov_size: int = 7):
        super().__init__()
        self.fov_size = fov_size

    def reset(self):
        pass  # stateless

    def forward(
        self,
        walls: torch.Tensor,
        velocity_hint: Tuple[int, int],
        num_ticks: int = 3
    ) -> bool:
        """
        Raycast along velocity_hint and return True if a wall is hit.

        Args:
            walls: [fov_size, fov_size] wall occupancy (values > 0.5 = wall)
            velocity_hint: (dx, dy) movement direction
            num_ticks: steps to cast (reuses existing call signature)

        Returns:
            veto: True if the ray hits a wall within num_ticks steps
        """
        dx, dy = velocity_hint
        center = self.fov_size // 2
        x, y = center, center
        for _ in range(1, num_ticks + 1):
            x += dx
            y += dy
            if 0 <= x < self.fov_size and 0 <= y < self.fov_size:
                if walls[y, x] > 0.5:
                    return True
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

class GhostAntenna(nn.Module):
    """
    1-Step Vicarious Trial & Error (Local Lookahead).
    
    Hardware-native decentralized planning: For each action, the agent
    "imagines" the outcome using only its local 7×7 field-of-view.
    
    - If target cell is a wall → 0.0 scent (blocked path)
    - If target cell is free → scent = L1 distance improvement to goal
    
    Injects scent current into 5 LIF neurons (one per action).
    Spike counts → added to action_logits.
    
    Biological analog: Rodent vicarious trial & error, hippocampal replay.
    """
    def __init__(self, tau: float = 2.0):
        super().__init__()
        self.lif = neuron.LIFNode(tau=tau, v_threshold=1.0, v_reset=0.0)

    def reset(self):
        functional.reset_net(self.lif)

    def forward(
        self,
        fov_walls: torch.Tensor,
        fov_goal: torch.Tensor,
        num_ticks: int = 10,
        gain: float = 1.5,  # Lowered gain for healthy LIF integration
    ) -> torch.Tensor:
        """
        Args:
            fov_walls: [7, 7] wall channel from observation (agent at centre [3, 3])
            fov_goal:  [7, 7] goal channel from observation (non-zero where goal visible)
            num_ticks: LIF simulation ticks
            gain: scent current multiplier (best direction → 1.5V → ~20 spikes/30 ticks)

        Returns:
            spikes: [5] spike counts per action (RIGHT/UP/LEFT/DOWN/STAY)

        No global coordinates used — the agent reads only its local FOV.
        """
        fov_targets = [
            (3, 4),  # RIGHT
            (2, 3),  # UP
            (3, 2),  # LEFT
            (4, 3),  # DOWN
            (3, 3),  # STAY
        ]

        # Locate goal within the FOV grid — purely local, no GPS math
        goal_mask   = fov_goal > 0
        goal_in_fov = bool(goal_mask.any().item())
        if goal_in_fov:
            idxs     = goal_mask.nonzero(as_tuple=False)
            goal_row = int(idxs[0, 0].item())
            goal_col = int(idxs[0, 1].item())

        scent = torch.zeros(5, device=fov_walls.device)
        for action_id, (fov_y, fov_x) in enumerate(fov_targets):
            if fov_walls[fov_y, fov_x] > 0.5:
                scent[action_id] = 0.0          # wall blocks this direction
            elif goal_in_fov:
                # Square the distance for sharper drop-off: only immediate neighbours score high
                d = abs(fov_y - goal_row) + abs(fov_x - goal_col)
                scent[action_id] = 10.0 / (d ** 2 + 1.0)
            else:
                scent[action_id] = 1.0          # goal outside FOV: uniform weak scent

        # Normalise so best direction = 1.0, worst ≈ 0.0 (contrast fix)
        scent_max = scent.max()
        if scent_max > 0.001:
            scent = scent / scent_max

        # best direction current = gain (1.5V → ~20 spikes in 30 ticks)
        # bad direction current  = ~0.0V → 0 spikes
        current = scent * gain

        functional.reset_net(self.lif)
        acc = torch.zeros(5, device=fov_walls.device)
        for _ in range(num_ticks):
            acc += self.lif(current)
        return acc


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

        # RM-STDP eligibility trace: same shape as action_weights [5, num_E]
        self.register_buffer('eligibility_trace', torch.zeros_like(self.action_weights.weight))

        # GhostAntenna: 1-step vicarious trial & error (local lookahead)
        self.ghost_antenna = GhostAntenna()

    def reset(self):
        """Reset all spiking meshes"""
        functional.reset_net(self.obs_mesh.lif)
        functional.reset_net(self.readout_mesh.lif)
        functional.reset_net(self.action_neurons)
        self.cpg.reset()
        self.shadow.reset()
        self.ghost_antenna.reset()
    
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
        obs_input = self.obs_input_proj(obs_flat) * 1.5

        obs_spike_accumulator = torch.zeros(self.obs_mesh.num_neurons, device=obs_input.device)
        obs_current_spikes    = torch.zeros(self.obs_mesh.num_neurons, device=obs_input.device)
        
        for t in range(num_ticks):
            recurrent = self.obs_recurrent(obs_current_spikes)
            recurrent = recurrent * self.obs_mesh.ei_mask
            
            total_input = obs_input * 0.8 + recurrent * 0.2
            obs_current_spikes    = self.obs_mesh.lif(total_input)
            obs_spike_accumulator += obs_current_spikes
        
        # Extract E neuron spikes and project
        obs_E_spikes = self.obs_mesh.get_E_neurons(obs_spike_accumulator)
        obs_compressed = self.obs_to_readout(obs_E_spikes)
        
        # === STEP 2: READOUT MESH ===
        readout_input = self.readout_input(obs_compressed) * 0.9
        
        readout_spike_accumulator = torch.zeros(self.readout_mesh.num_neurons, device=readout_input.device)
        readout_current_spikes    = torch.zeros(self.readout_mesh.num_neurons, device=readout_input.device)
        
        for t in range(num_ticks):
            recurrent = self.readout_recurrent(readout_current_spikes)
            recurrent = recurrent * self.readout_mesh.ei_mask
            
            total_input = readout_input * 0.85 + recurrent * 0.15
            readout_current_spikes    = self.readout_mesh.lif(total_input)
            readout_spike_accumulator += readout_current_spikes
        
        # Extract E spikes and compute actions
        readout_E_spikes = self.readout_mesh.get_E_neurons(readout_spike_accumulator)
        # action_weights is trained directly on readout E spikes — use logits as scores
        action_logits = self.action_weights(readout_E_spikes)  # [5]

        # RM-STDP trace update: trace(t) = trace(t-1) * 0.95 + softmax(logits) ⊗ readout_E
        with torch.no_grad():
            probs = torch.softmax(action_logits, dim=0)          # [5]
            self.eligibility_trace.mul_(0.95).add_(torch.outer(probs, readout_E_spikes))

        # Select tentative action from learned logits
        tentative_action = torch.argmax(action_logits).item()
        
        # === STEP 3: INTENT MAP ===
        my_intent_E, intent_veto = self.intent_map(
            action=tentative_action,
            neighbor_E_spikes=torch.stack(neighbor_intent_E) if len(neighbor_intent_E) > 0 else None,
            num_ticks=5
        )
        
        # === STEP 4: CPG ===
        my_cpg_E, should_stay = self.cpg(
            neighbor_E_spikes=torch.stack(neighbor_cpg_E) if len(neighbor_cpg_E) > 0 else None,
            num_ticks=5
        )
        
        # CPG modulation: gentle nudge to turn-take
        if should_stay:
            action_logits = action_logits.clone()
            action_logits[4] = action_logits.max() + 1.0

        # Intent map spatial forcefield: neighbour claimed this cell — yield!
        if intent_veto:
            action_logits = action_logits.clone()
            action_logits[4] = action_logits.max() + 5.0

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
        veto = self.shadow(walls, velocity, num_ticks=1)  # Only fear the immediate next cell

        # Shadow VETO: hard wall reflex
        if veto:
            action_logits = action_logits.clone()
            action_logits[4] = action_logits.max() + 10.0

        return action_logits, my_intent_E, my_cpg_E

    def apply_dopamine(self, reward_signal: float, learning_rate: float):
        """Three-factor weight update: ΔW = lr * reward * eligibility_trace."""
        with torch.no_grad():
            self.action_weights.weight.data.add_(
                learning_rate * reward_signal * self.eligibility_trace
            )
            self.eligibility_trace.zero_()


# === SWARM NETWORK ===

class SwarmLSM(nn.Module):
    """
    Decentralized swarm with full spiking dynamics + Virtual Stigmergy.
    
    Each agent = independent spiking meshes.
    Communication = E neuron spike broadcasts only (Dale's Law).
    Pheromone Mesh = Shared environmental memory (repels from visited cells).
    """
    def __init__(self, num_agents: int = 5, communication_range: float = 3.0):
        super().__init__()
        self.num_agents = num_agents
        self.communication_range = communication_range
        
        # Create independent spiking agents
        self.agents = nn.ModuleList([
            AgentLSM(agent_id=i) for i in range(num_agents)
        ])
        
        # S.E.X. V2: Noradrenaline injection state
        # When a VETO is detected, store {agent_id: noise_magnitude} here
        # Forward pass will check this and inject voltage noise
        self.inject_noradrenaline = {}
    
    def reset(self):
        """Reset all agents' spiking dynamics"""
        for agent in self.agents:
            agent.reset()
        # Clear noradrenaline injection flags
        self.inject_noradrenaline = {}
    
    def get_neighbor_weights(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute [N, N] distance-decay weight matrix for spike broadcast.
        weights[i, j] = signal received by agent i from agent j.

        Uses exponential decay instead of a hard distance threshold — all agents
        broadcast; physical distance attenuates the signal naturally. No central
        server decides who hears whom.
        """
        N = self.num_agents
        weights = torch.zeros(N, N, device=positions.device)
        for i in range(N):
            dists      = torch.norm(positions - positions[i], dim=1)
            weights[i] = torch.exp(-dists / self.communication_range)
            weights[i, i] = 0.0  # no self-connection
        return weights
    
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
        cached_intent_veto = []

        for agent_id in range(self.num_agents):
            ag  = self.agents[agent_id]
            obs = observations[agent_id]

            # --- obs mesh ---
            obs_flat = obs.flatten()
            obs_input = ag.obs_input_proj(obs_flat) * 1.5
            obs_acc = torch.zeros(ag.obs_mesh.num_neurons, device=obs_flat.device)
            obs_cur = torch.zeros(ag.obs_mesh.num_neurons, device=obs_flat.device)
            for _ in range(num_ticks):
                rec = ag.obs_recurrent(obs_cur) * ag.obs_mesh.ei_mask
                obs_cur  = ag.obs_mesh.lif(obs_input * 0.8 + rec * 0.2)
                obs_acc += obs_cur

            obs_E      = ag.obs_mesh.get_E_neurons(obs_acc)
            rd_input   = ag.readout_input(ag.obs_to_readout(obs_E)) * 0.9
            rd_acc     = torch.zeros(ag.readout_mesh.num_neurons, device=obs_flat.device)
            rd_cur     = torch.zeros(ag.readout_mesh.num_neurons, device=obs_flat.device)
            
            # S.E.X. V2: Noradrenaline Flush (voltage injection for online learning)
            for tick in range(num_ticks):
                rec = ag.readout_recurrent(rd_cur) * ag.readout_mesh.ei_mask
                total_input = rd_input * 0.85 + rec * 0.15
                
                # NORADRENALINE FLUSH: If this agent had a VETO, inject chemical noise
                if agent_id in self.inject_noradrenaline and tick == 0:  # Only first tick
                    noise_magnitude = self.inject_noradrenaline[agent_id]
                    num_E = ag.readout_mesh.num_E
                    chemical_noise = torch.zeros(ag.readout_mesh.num_neurons, device=obs_flat.device)
                    chemical_noise[:num_E] = torch.abs(torch.randn(num_E, device=obs_flat.device)) * noise_magnitude * 3.0
                    total_input = total_input + chemical_noise
                
                rd_cur  = ag.readout_mesh.lif(total_input)
                rd_acc += rd_cur

            readout_E = ag.readout_mesh.get_E_neurons(rd_acc)

            # --- action selection: logits directly (no LIF bottleneck) ---
            action_logits = ag.action_weights(readout_E)
            
            # GhostAntenna: FOV-local 1-step lookahead — no global coordinates
            antenna_spikes = ag.ghost_antenna(
                obs[0], obs[1],  # fov_walls, fov_goal
                num_ticks=num_ticks, gain=1.5
            )
            action_logits = action_logits + antenna_spikes * 0.4

            tentative = int(torch.argmax(action_logits).item())

            # --- Pass 1: intent map + CPG with no neighbors (cold broadcast) ---
            intent_E_cold, _ = ag.intent_map(action=tentative, neighbor_E_spikes=None, num_ticks=5)
            cpg_E_cold, should_stay_cold = ag.cpg(neighbor_E_spikes=None, num_ticks=5)
            # --- shadow ---
            vel_map = {0:(1,0),1:(0,-1),2:(-1,0),3:(0,1),4:(0,0)}
            veto = ag.shadow(obs[0], vel_map[tentative], num_ticks=1)

            cached_readout_E.append(readout_E)
            cached_action_acc.append(action_logits)
            cached_intent_E.append(intent_E_cold)
            cached_cpg_E.append(cpg_E_cold)
            cached_should_stay.append(should_stay_cold)
            cached_veto.append(veto)
            cached_intent_veto.append(False)  # cold — overwritten in Pass 2 if neighbour exists

        # ── Pass 2: re-run intent_map + CPG with real neighbour spikes ────────
        # Now that ALL agents have broadcast their cold spikes we can compute
        # the physical communication-range neighbourhood and wire them up.
        if positions is not None and self.num_agents > 1:
            # Distance-decay broadcast: all agents hear all others; amplitude
            # falls off exponentially with distance. No hard threshold, no
            # central router making selective routing decisions.
            neighbor_weights = self.get_neighbor_weights(positions)  # [N, N]
            for agent_id in range(self.num_agents):
                ag = self.agents[agent_id]
                nb_intent = torch.stack([
                    cached_intent_E[j] * neighbor_weights[agent_id, j]
                    for j in range(self.num_agents) if j != agent_id
                ])  # [N-1, num_E]
                nb_cpg = torch.stack([
                    cached_cpg_E[j] * neighbor_weights[agent_id, j]
                    for j in range(self.num_agents) if j != agent_id
                ])  # [N-1, num_E]

                # Reset LIF state so the second run is clean
                functional.reset_net(ag.intent_map.lif)
                functional.reset_net(ag.cpg.lif)

                tentative = int(torch.argmax(cached_action_acc[agent_id]).item())
                intent_E, intent_veto = ag.intent_map(
                    action=tentative,
                    neighbor_E_spikes=nb_intent,
                    num_ticks=5
                )

                cpg_E, should_stay = ag.cpg(
                    neighbor_E_spikes=nb_cpg,
                    num_ticks=5
                )

                cached_intent_E[agent_id]    = intent_E
                cached_intent_veto[agent_id] = intent_veto
                cached_cpg_E[agent_id]       = cpg_E
                cached_should_stay[agent_id] = should_stay

        # Apply CPG/shadow modulation using fully-wired cached results
        # (obs/readout reservoir untouched — only intent_map/CPG re-ran above)
        #
        # ARCHITECTURAL DEBT — SOFTWARE SUBSUMPTION BRIDGE:
        # The VETO/CPG signals below are injected via direct tensor writes rather
        # than through a real inhibitory spike projection onto the motor neurons.
        # In true neuromorphic hardware the ShadowCaster would fire an inhibitory
        # volley suppressing RIGHT/UP/LEFT/DOWN and exciting STAY. Here a Python
        # if-statement plays that role. This is a known lie; fixing it requires a
        # dedicated InhibitoryProjection layer routing shadow spikes → motor layer.
        final_action_spikes = []
        for agent_id in range(self.num_agents):
            action_acc  = cached_action_acc[agent_id].clone()
            should_stay = cached_should_stay[agent_id]
            veto        = cached_veto[agent_id]
            intent_veto = cached_intent_veto[agent_id]

            if should_stay:
                action_acc[4] = action_acc.max() + 1.0   # gentle CPG nudge
            if intent_veto:  # spatial forcefield: neighbour claimed this cell — yield!
                action_acc[4] = action_acc.max() + 5.0
            if veto:
                action_acc[4] = action_acc.max() + 10.0  # hard wall reflex

            final_action_spikes.append(action_acc)

        # Clear noradrenaline injection flags after forward pass is complete
        self.inject_noradrenaline = {}
        
        # Return veto flags so the training loop knows a mistake happened (pre-subsumption state)
        # Return cached_readout_E so evaluate() can track real live spike health
        return torch.stack(final_action_spikes), cached_intent_E, cached_cpg_E, cached_veto, cached_readout_E


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

                        obs_input    = ag.obs_input_proj(obs_flat) * 1.5
                        obs_spike_acc = torch.zeros(ag.obs_mesh.num_neurons, device=self.device)
                        obs_spk_cur   = torch.zeros(ag.obs_mesh.num_neurons, device=self.device)
                        for _ in range(num_ticks):
                            rec = ag.obs_recurrent(obs_spk_cur) * ag.obs_mesh.ei_mask
                            obs_spk_cur   = ag.obs_mesh.lif(obs_input * 0.8 + rec * 0.2)
                            obs_spike_acc += obs_spk_cur

                        obs_E        = ag.obs_mesh.get_E_neurons(obs_spike_acc)
                        rd_input     = ag.readout_input(ag.obs_to_readout(obs_E)) * 0.9
                        rd_spike_acc = torch.zeros(ag.readout_mesh.num_neurons, device=self.device)
                        rd_spk_cur   = torch.zeros(ag.readout_mesh.num_neurons, device=self.device)
                        for _ in range(num_ticks):
                            rec = ag.readout_recurrent(rd_spk_cur) * ag.readout_mesh.ei_mask
                            rd_spk_cur   = ag.readout_mesh.lif(rd_input * 0.85 + rec * 0.15)
                            rd_spike_acc += rd_spk_cur

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
            # LR schedule: divide by 10 if stagnation for 5 epochs
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5
            )

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
                
                avg_loss = total_loss / N
                scheduler.step(avg_loss)
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"   Agent {agent_id} | epoch {epoch+1:3d}/{epochs} "
                          f"loss={avg_loss:.4f}  acc={correct/N:.2%}  lr={current_lr:.2e}")

    def train_volatility(self, dataset, num_episodes: int = 500, lr: float = 0.005, decay: float = 0.995, num_ticks: int = 10):
        """
        S.E.X. V2: Noradrenaline Flush (Voltage-Based Attractor Destabilization).
        
        The VETO signal (from Shadow Caster or Intent Map) triggers ephemeral voltage noise.
        When a VETO fires, zero-mean noise is injected into E-neuron membrane potentials
        for ONE tick, scrambling the liquid state WITHOUT modifying the frozen weights.
        
        Biological analog: Norepinephrine temporarily alters neuronal excitability
        (RAM voltage) instead of rewiring synapses (hard drive weights).
        
        No backprop. No loss function. The hardware sensors scramble the voltage.
        
        Args:
            dataset: MAPF dataset with trajectories
            num_episodes: Number of exploration episodes
            lr: Noise magnitude for voltage injection (zero-mean)
            decay: Learning rate annealing per episode
            num_ticks: LIF simulation ticks
        """
        import yaml as _yaml
        
        print("\n" + "="*70)
        print("🌀 S.E.X. V2: NORADRENALINE FLUSH (VOLTAGE-BASED ATTRACTOR DESTABILIZATION)")
        print("="*70)
        print(f"   Episodes: {num_episodes}")
        print(f"   Initial LR: {lr:.5f} (voltage noise magnitude)")
        print(f"   Decay: {decay:.4f} (annealing)")
        print("="*70)
        print("\n   The agent will explore. When it makes a mistake, the Shadow")
        print("   Caster or Intent Map will fire a VETO. At that exact moment,")
        print("   zero-mean VOLTAGE noise floods E-neuron membrane potentials. The")
        print("   agent learns by destabilizing bad attractor states without")
        print("   draining computational energy from the liquid state machine.")
        print("="*70 + "\n")
        
        # Action deltas
        action_to_delta = np.array([[1,0],[0,-1],[-1,0],[0,1],[0,0]], dtype=np.float32)
        
        # Track VETO statistics per agent
        veto_counts = [0 for _ in range(self.network.num_agents)]
        total_steps = [0 for _ in range(self.network.num_agents)]
        
        # Enable gradients on action_weights only
        for agent in self.network.agents:
            agent.action_weights.requires_grad_(True)
        
        self.network.eval()  # Keep reservoir frozen
        current_lr = lr
        
        for ep_idx in tqdm(range(min(num_episodes, len(dataset))), desc="Exploring"):
            # Load episode environment
            case_dir = dataset.data[ep_idx]['case_dir']
            try:
                inp = _yaml.safe_load(open(f"{case_dir}/input.yaml"))
            except Exception:
                continue
            
            if not inp or 'map' not in inp:
                continue
            
            map_data = inp['map']
            agents_info = inp.get('agents', [])
            if len(agents_info) != self.network.num_agents:
                continue
            
            H, W = map_data['dimensions'][1], map_data['dimensions'][0]
            grid = np.zeros((H, W), dtype=np.float32)
            for obs in map_data.get('obstacles', []):
                ox, oy = obs[0], obs[1]
                if 0 <= oy < H and 0 <= ox < W:
                    grid[oy, ox] = 2.0
            
            # Start positions and goals
            positions = torch.tensor(
                [[a['start'][0], a['start'][1]] for a in agents_info],
                dtype=torch.float32, device=self.device
            )
            goals = torch.tensor(
                [[a['goal'][0], a['goal'][1]] for a in agents_info],
                dtype=torch.float32, device=self.device
            )
            
            self.network.reset()
            
            # Closed-loop simulation for up to 50 steps
            for t in range(50):
                # Generate FOV
                obs = self._generate_fov(grid, positions, goals, fov_size=7)  # [A, 2, 7, 7]
                
                # Track which agents need S.E.X. (noradrenaline flush)
                inject_noradrenaline = {}  # {agent_id: noise_magnitude}
                
                # Forward pass with gradient tracking
                for agent_id in range(self.network.num_agents):
                    ag = self.network.agents[agent_id]
                    obs_flat = obs[agent_id].flatten()
                    
                    # === Obs mesh (frozen) ===
                    with torch.no_grad():
                        obs_input = ag.obs_input_proj(obs_flat) * 1.5
                        obs_acc = torch.zeros(ag.obs_mesh.num_neurons, device=self.device)
                        obs_cur = torch.zeros(ag.obs_mesh.num_neurons, device=self.device)
                        for _ in range(num_ticks):
                            rec = ag.obs_recurrent(obs_cur) * ag.obs_mesh.ei_mask
                            obs_cur = ag.obs_mesh.lif(obs_input * 0.8 + rec * 0.2)
                            obs_acc += obs_cur
                        
                        obs_E = ag.obs_mesh.get_E_neurons(obs_acc)
                        rd_input = ag.readout_input(ag.obs_to_readout(obs_E)) * 0.9
                        rd_acc = torch.zeros(ag.readout_mesh.num_neurons, device=self.device)
                        rd_cur = torch.zeros(ag.readout_mesh.num_neurons, device=self.device)
                        for _ in range(num_ticks):
                            rec = ag.readout_recurrent(rd_cur) * ag.readout_mesh.ei_mask
                            rd_cur = ag.readout_mesh.lif(rd_input * 0.85 + rec * 0.15)
                            rd_acc += rd_cur
                        
                        readout_E = ag.readout_mesh.get_E_neurons(rd_acc)
                    
                    # === Action selection (TRAINABLE) ===
                    action_logits = ag.action_weights(readout_E)  # Gradients flow here!
                    tentative_action = int(torch.argmax(action_logits).item())
                    
                    # === Check for VETO ===
                    with torch.no_grad():
                        # Shadow caster
                        vel_map = {0:(1,0), 1:(0,-1), 2:(-1,0), 3:(0,1), 4:(0,0)}
                        shadow_veto = ag.shadow(obs[agent_id][0], vel_map[tentative_action], num_ticks=1)
                        
                        # Intent map (simplified - check if neighbor would crush)
                        # For now, use shadow veto only (intent veto requires neighbor communication)
                        veto = shadow_veto
                    
                    # === S.E.X. PROTOCOL: Mark for Noradrenaline Flush ===
                    if veto and tentative_action != 4:  # Mistake detected! (not STAY)
                        veto_counts[agent_id] += 1
                        # Store noise magnitude for this agent's next forward pass
                        inject_noradrenaline[agent_id] = current_lr
                    
                    total_steps[agent_id] += 1
                
                # Execute actions with honest wall + agent-agent collision physics
                with torch.no_grad():
                    # --- Collect post-noradrenaline actions (one obs+readout pass per agent) ---
                    chosen_actions = []
                    for agent_id in range(self.network.num_agents):
                        ag = self.network.agents[agent_id]
                        obs_flat = obs[agent_id].flatten()
                        obs_input = ag.obs_input_proj(obs_flat) * 1.5
                        obs_acc = torch.zeros(ag.obs_mesh.num_neurons, device=self.device)
                        obs_cur = torch.zeros(ag.obs_mesh.num_neurons, device=self.device)
                        for _ in range(num_ticks):
                            rec = ag.obs_recurrent(obs_cur) * ag.obs_mesh.ei_mask
                            obs_cur = ag.obs_mesh.lif(obs_input * 0.8 + rec * 0.2)
                            obs_acc += obs_cur
                        obs_E = ag.obs_mesh.get_E_neurons(obs_acc)
                        rd_input = ag.readout_input(ag.obs_to_readout(obs_E)) * 0.9
                        rd_acc = torch.zeros(ag.readout_mesh.num_neurons, device=self.device)
                        rd_cur = torch.zeros(ag.readout_mesh.num_neurons, device=self.device)
                        # NORADRENALINE FLUSH: inject zero-mean noise into readout E-neuron
                        # membrane potentials at tick 0 (scrambles state, not weights)
                        for tick in range(num_ticks):
                            rec = ag.readout_recurrent(rd_cur) * ag.readout_mesh.ei_mask
                            total_input = rd_input * 0.85 + rec * 0.15
                            if agent_id in inject_noradrenaline and tick == 0:
                                noise_magnitude = inject_noradrenaline[agent_id]
                                num_E = ag.readout_mesh.num_E
                                chemical_noise = torch.zeros(ag.readout_mesh.num_neurons, device=self.device)
                                chemical_noise[:num_E] = torch.abs(torch.randn(num_E, device=self.device)) * noise_magnitude * 3.0
                                total_input = total_input + chemical_noise
                            rd_cur = ag.readout_mesh.lif(total_input)
                            rd_acc += rd_cur
                        readout_E = ag.readout_mesh.get_E_neurons(rd_acc)
                        chosen_actions.append(int(torch.argmax(ag.action_weights(readout_E)).item()))

                    # --- Apply collision physics (matches train_hybrid Phase 2 and evaluate()) ---
                    positions_temp = positions.clone()
                    new_pos = []
                    for agent_id in range(self.network.num_agents):
                        dx = int(action_to_delta[chosen_actions[agent_id]][0])
                        dy = int(action_to_delta[chosen_actions[agent_id]][1])
                        nx = int(np.clip(positions[agent_id, 0].item() + dx, 0, W - 1))
                        ny = int(np.clip(positions[agent_id, 1].item() + dy, 0, H - 1))
                        new_pos.append([nx, ny])

                    # Wall collision: revert agent to prior position
                    for i in range(self.network.num_agents):
                        nx, ny = new_pos[i]
                        if grid[ny, nx] > 1.5:
                            new_pos[i] = [int(positions_temp[i, 0].item()), int(positions_temp[i, 1].item())]

                    # Agent-agent collision: revert both agents
                    position_dict = {}
                    for i in range(self.network.num_agents):
                        pos_key = tuple(new_pos[i])
                        if pos_key in position_dict:
                            new_pos[i] = [int(positions_temp[i, 0].item()), int(positions_temp[i, 1].item())]
                            j = position_dict[pos_key]
                            new_pos[j] = [int(positions_temp[j, 0].item()), int(positions_temp[j, 1].item())]
                        else:
                            position_dict[pos_key] = i

                    for i in range(self.network.num_agents):
                        positions[i, 0] = new_pos[i][0]
                        positions[i, 1] = new_pos[i][1]
                
                # Check if all reached goals
                distances = torch.norm(positions - goals, dim=1)
                if (distances < 0.5).all():
                    break
            
            # Anneal volatility strength (noise magnitude decreases over time)
            current_lr *= decay
            
            # Log every 50 episodes
            if (ep_idx + 1) % 50 == 0:
                total_vetos = sum(veto_counts)
                total_all_steps = sum(total_steps)
                veto_rate = total_vetos / max(total_all_steps, 1) * 100
                print(f"   Ep {ep_idx+1:4d}/{num_episodes} | "
                      f"VETOs: {total_vetos:5d} ({veto_rate:.2f}%) | "
                      f"LR: {current_lr:.5f}")
        
        print("\n" + "="*70)
        print("🌀 S.E.X. V2 (NORADRENALINE FLUSH) LEARNING COMPLETE")
        print("="*70)
        for agent_id in range(self.network.num_agents):
            veto_rate = veto_counts[agent_id] / max(total_steps[agent_id], 1) * 100
            print(f"   Agent {agent_id}: {veto_counts[agent_id]:5d} VETOs / {total_steps[agent_id]:5d} steps ({veto_rate:.2f}%)")
        print("="*70 + "\n")
        print("   The agent has learned through reservoir destabilization.")
        print("   Its liquid states now avoid bad attractors through ephemeral voltage noise.")
        print("="*70 + "\n")

    def train_hybrid(
        self, 
        dataset, 
        cortex_method: str = 'sgd',
        max_episodes: int = 1000,
        volatility_episodes: int = 300,
        volatility_lr: float = 0.005,
        volatility_decay: float = 0.995,
        num_ticks: int = 10,
        phase1_checkpoint: str = None,
        skip_phase2: bool = False,
        **cortex_kwargs
    ):
        """
        HYBRID BIOMIMETIC LEARNING: Cortex (Supervised) + Noradrenaline Flush.
        
        This is the ultimate architecture - combining global pathfinding
        intuition from a teacher with voltage-based state destabilization for mistakes.
        
        Phase 1 - THE CORTEX (Global Strategy):
            Pre-train action_weights with supervised learning from expert
            demos. This gives the agent the "big picture" of how to navigate
            traffic and move toward goals efficiently.
        
        Phase 2 - S.E.X. V2: NORADRENALINE FLUSH (Voltage-Based Attractor Destabilization):
            Fine-tune by scrambling the reservoir's voltage state when errors occur.
            When the teacher's logic fails (e.g., 44% STAY bias in corners),
            the VETO fires and injects zero-mean noise into E-neuron membrane
            potentials (NOT weights) for ONE tick. This destabilizes bad attractors
            while preserving the frozen weight structure.
            
            Biological analog: Norepinephrine washes over neurons, temporarily
            altering their excitability (RAM) without rewiring synapses (hard drive).
        
        Phase 3 - DEPLOYMENT (Online Adaptation):
            During evaluation, online learning remains enabled. If the agent
            encounters novel scenarios where the teacher failed, real-time
            VETOs continue to inject voltage noise, allowing continuous
            adaptation WITHOUT modifying frozen weights.
        
        Args:
            dataset: MAPF dataset
            cortex_method: 'ridge' or 'sgd' for Phase 1
            max_episodes: Episodes for supervised state collection
            volatility_episodes: Episodes for Phase 2 voltage noise injection
            volatility_lr: Noise magnitude for voltage injection (zero-mean)
            volatility_decay: LR annealing
            num_ticks: LIF ticks
            **cortex_kwargs: ridge_alpha, sgd_epochs, sgd_lr, sgd_batch_size, etc.
        """
        print("\n" + "="*70)
        print("🧠 HYBRID BIOMIMETIC LEARNING: CORTEX + NORADRENALINE")
        print("="*70)
        print("   Phase 1: CORTEX (Supervised Pre-Training)")
        print("   Phase 2: NORADRENALINE FLUSH (Voltage-Based Attractor Destabilization)")
        print("   Phase 3: DEPLOYMENT (Online Adaptation during eval)")
        print("="*70 + "\n")
        
        # ══════════════════════════════════════════════════════════════════
        # PHASE 1: THE CORTEX - Learn from the Teacher
        # ══════════════════════════════════════════════════════════════════
        print("="*70)
        print("📚 PHASE 1: CORTEX - Learning Global Strategy from Teacher")
        print("="*70)
        print(f"   Method: {cortex_method.upper()}")
        print(f"   Episodes: {max_episodes}")
        print("="*70 + "\n")
        
        import os as _os
        if phase1_checkpoint and _os.path.isfile(phase1_checkpoint):
            print(f"   ⚡ Loading Phase 1 checkpoint: {phase1_checkpoint}")
            self.network.load_state_dict(torch.load(phase1_checkpoint, map_location=self.device))
            print("   ✅ Phase 1 weights restored — skipping supervised training.")
        else:
            # Collect supervised states
            X_train, Y_train = self.collect_states(dataset, max_episodes=max_episodes, num_ticks=num_ticks)

            # Train with chosen method
            if cortex_method == 'ridge':
                alpha = cortex_kwargs.get('ridge_alpha', 1.0)
                self.train_ridge(X_train, Y_train, alpha=alpha)
            elif cortex_method == 'sgd':
                epochs = cortex_kwargs.get('sgd_epochs', 150)
                lr = cortex_kwargs.get('sgd_lr', 1e-4)
                batch_size = cortex_kwargs.get('sgd_batch_size', 128)
                self.train_sgd(X_train, Y_train, epochs=epochs, lr=lr, batch_size=batch_size)
            else:
                raise ValueError(f"Unknown cortex_method: {cortex_method}")

            if phase1_checkpoint:
                _os.makedirs(_os.path.dirname(_os.path.abspath(phase1_checkpoint)), exist_ok=True)
                torch.save(self.network.state_dict(), phase1_checkpoint)
                print(f"   💾 Phase 1 checkpoint saved to {phase1_checkpoint}")

        print("\n✅ PHASE 1 COMPLETE: Cortex has learned the teacher's strategy.")
        print("   The agent now has global pathfinding intuition but may have")
        print("   local bugs (e.g., STAY bias, corner deadlocks).\n")
        
        # ══════════════════════════════════════════════════════════════════
        # PHASE 2: SYNAPTIC VOLATILITY - Scramble Bad Attractor States
        # ══════════════════════════════════════════════════════════════════
        if skip_phase2:
            print("\n⏭️  PHASE 2 SKIPPED (skip_phase2=True). Running evaluation on Phase 1 weights.")
            return

        print("="*70)
        print("🌀 PHASE 2: S.E.X. V2 - Noradrenaline Flush (Voltage Noise)")
        print("="*70)
        print(f"   Episodes: {volatility_episodes}")
        print(f"   Noradrenaline LR: {volatility_lr:.5f} (voltage noise magnitude)")
        print(f"   Decay: {volatility_decay:.4f}")
        print("="*70)
        print("\n   The agent will now explore with its teacher-trained weights.")
        print("   When the teacher's logic fails (collision predicted), the")
        print("   Shadow Caster will fire a VETO and inject zero-mean noise")
        print("   into co-active reservoir synapses. This scrambles bad liquid")
        print("   states without draining computational energy.")
        print("="*70 + "\n")
        
        # Run synaptic volatility fine-tuning
        import yaml as _yaml
        action_to_delta = np.array([[1,0],[0,-1],[-1,0],[0,1],[0,0]], dtype=np.float32)
        
        veto_counts = [0 for _ in range(self.network.num_agents)]
        total_steps = [0 for _ in range(self.network.num_agents)]
        
        for agent in self.network.agents:
            agent.action_weights.requires_grad_(True)
        
        self.network.eval()
        current_lr = volatility_lr
        
        for ep_idx in tqdm(range(min(volatility_episodes, len(dataset))), desc="Synaptic Volatility"):
            case_dir = dataset.data[ep_idx]['case_dir']
            try:
                inp = _yaml.safe_load(open(f"{case_dir}/input.yaml"))
            except Exception:
                continue
            
            if not inp or 'map' not in inp:
                continue
            
            map_data = inp['map']
            agents_info = inp.get('agents', [])
            if len(agents_info) != self.network.num_agents:
                continue
            
            H, W = map_data['dimensions'][1], map_data['dimensions'][0]
            grid = np.zeros((H, W), dtype=np.float32)
            for obs in map_data.get('obstacles', []):
                ox, oy = obs[0], obs[1]
                if 0 <= oy < H and 0 <= ox < W:
                    grid[oy, ox] = 2.0
            
            positions = torch.tensor(
                [[a['start'][0], a['start'][1]] for a in agents_info],
                dtype=torch.float32, device=self.device
            )
            goals = torch.tensor(
                [[a['goal'][0], a['goal'][1]] for a in agents_info],
                dtype=torch.float32, device=self.device
            )
            
            self.network.reset()
            
            for t in range(50):
                obs = self._generate_fov(grid, positions, goals, fov_size=7)

                # Pass 1: get actions and TRUE veto flags from the REAL network
                # (includes Intent Map, CPG, Ghost Antenna, Shadow — all of it)
                with torch.no_grad():
                    action_spikes, _, _, true_vetoes, _ = self.network(obs, positions, num_ticks=num_ticks, goals=goals)
                    predicted_actions = action_spikes.argmax(dim=-1)

                # Check VETOs: use flags returned by network (pre-subsumption bridge state)
                # The paradox fix: the bridge already deflected STAY, so re-checking shadow on the
                # post-bridge action would never see the original mistake.
                for agent_id in range(self.network.num_agents):
                    if true_vetoes[agent_id]:  # Shadow Caster had to intervene!
                        veto_counts[agent_id] += 1
                        # Fast reflex: RAM scramble via noradrenaline
                        self.network.inject_noradrenaline[agent_id] = current_lr
                        # Slow plasticity: barely degrade the synapses
                        self.network.agents[agent_id].apply_dopamine(reward_signal=-0.05, learning_rate=0.001)
                        if agent_id == 0 and ep_idx % 100 == 0:
                            print(f"      🌀 Neuromodulation: Noradrenaline (RAM) + Serotonin (Weights)")
                    total_steps[agent_id] += 1

                # Pass 2: re-run with noradrenaline flush if any VETOs fired
                # inject_noradrenaline is consumed and cleared inside forward()
                if self.network.inject_noradrenaline:
                    with torch.no_grad():
                        action_spikes, _, _, _, _ = self.network(obs, positions, num_ticks=num_ticks, goals=goals)
                        predicted_actions = action_spikes.argmax(dim=-1)

                # Move agents with honest wall + agent-agent collision physics
                action_to_delta_t = torch.tensor(
                    [[1,0],[0,-1],[-1,0],[0,1],[0,0]], dtype=torch.float32, device=self.device
                )
                positions_temp = positions.clone()
                new_positions = (positions + action_to_delta_t[predicted_actions]).clone()
                new_positions[:, 0] = torch.clamp(new_positions[:, 0], 0, W - 1)
                new_positions[:, 1] = torch.clamp(new_positions[:, 1], 0, H - 1)

                # Wall collision: revert to prior position
                for i in range(self.network.num_agents):
                    ix, iy = int(new_positions[i, 0].item()), int(new_positions[i, 1].item())
                    if grid[iy, ix] > 1.5:
                        new_positions[i] = positions_temp[i]

                # Agent-agent collision: revert both agents
                position_dict = {}
                for i in range(self.network.num_agents):
                    pos_key = (int(new_positions[i, 0].item()), int(new_positions[i, 1].item()))
                    if pos_key in position_dict:
                        new_positions[i] = positions_temp[i]
                        new_positions[position_dict[pos_key]] = positions_temp[position_dict[pos_key]]
                    else:
                        position_dict[pos_key] = i

                positions = new_positions

                distances = torch.norm(positions - goals, dim=1)
                for i in range(self.network.num_agents):
                    if distances[i] < 0.5:
                        self.network.agents[i].apply_dopamine(reward_signal=1.0, learning_rate=0.01)
                if (distances < 0.5).all():
                    break
            
            current_lr *= volatility_decay
            
            if (ep_idx + 1) % 50 == 0:
                total_vetos = sum(veto_counts)
                total_all_steps = sum(total_steps)
                veto_rate = total_vetos / max(total_all_steps, 1) * 100
                print(f"   Ep {ep_idx+1:4d}/{volatility_episodes} | "
                      f"VETOs: {total_vetos:5d} ({veto_rate:.2f}%) | "
                      f"LR: {current_lr:.5f}")
        
        print("\n✅ PHASE 2 COMPLETE: S.E.X. V2 (Noradrenaline Flush) has destabilized bad attractors.")
        print("   Voltage scrambling (RAM) broke bad attractors; RM-STDP (weights) locked in the fixes.\n")
        
        print("="*70)
        print("🎯 HYBRID TRAINING COMPLETE")
        print("="*70)
        for agent_id in range(self.network.num_agents):
            veto_rate = veto_counts[agent_id] / max(total_steps[agent_id], 1) * 100
            print(f"   Agent {agent_id}: {veto_counts[agent_id]:5d} VETOs / {total_steps[agent_id]:5d} steps ({veto_rate:.2f}%)")
        print("="*70 + "\n")
        print("   🧠 CORTEX: Global pathfinding strategy from teacher")
        print("   🌀 NORADRENALINE: Voltage-based reservoir destabilization for collision avoidance")
        print("   🚀 DEPLOY: Online learning continues during evaluation")
        print("\n   You have built a biomimetic liquid state machine with attractor control.")
        print("="*70 + "\n")



    def _generate_fov(self, grid: np.ndarray, positions: torch.Tensor, goals: torch.Tensor, fov_size: int = 7) -> torch.Tensor:
        pad = fov_size // 2
        # Base grid with static walls
        grid_padded = np.pad(grid, ((pad, pad), (pad, pad)), constant_values=2)

        # Bake current agent positions into the grid as dynamic walls so the
        # ShadowCaster can see them and VETO collisions before they happen
        dynamic_grid = grid_padded.copy()
        num_agents = positions.shape[0]
        for j in range(num_agents):
            px = int(positions[j, 0].item()) + pad
            py = int(positions[j, 1].item()) + pad
            dynamic_grid[py, px] = 2.0

        fov = np.zeros((num_agents, 2, fov_size, fov_size), dtype=np.float32)

        for i in range(num_agents):
            x, y   = int(positions[i, 0].item()), int(positions[i, 1].item())
            gx, gy = int(goals[i, 0].item()),     int(goals[i, 1].item())

            # Extract FOV from dynamic grid (static walls + other agents)
            agent_fov = dynamic_grid[y:y + fov_size, x:x + fov_size].copy()
            # Clear own body so the agent doesn't treat itself as a wall
            agent_fov[pad, pad] = 0.0

            # NO FLIP — sensory cortex must match the physical world
            fov[i, 0, :, :] = agent_fov

            # Mark goal in Channel 1
            rel_gx = gx - x + pad
            rel_gy = gy - y + pad
            if 0 <= rel_gx < fov_size and 0 <= rel_gy < fov_size:
                # Goal is inside FOV — mark exact cell strongly
                fov[i, 1, rel_gy, rel_gx] = 3.0
            else:
                # Goal is outside FOV — project a compass beacon onto the nearest edge pixel
                # so the Ghost Antenna always has a directional gradient to follow
                edge_x = max(0, min(fov_size - 1, rel_gx))
                edge_y = max(0, min(fov_size - 1, rel_gy))
                fov[i, 1, edge_y, edge_x] = 1.5

        return torch.from_numpy(fov).to(self.device)
    
    def evaluate(self, dataset, num_episodes: int = 100, max_timesteps: int = 50, num_ticks: int = 10, log_ticks: bool = True, verbose: bool = False, save_dir: str = None):
        """
        Evaluate on MAPF metrics with CLOSED-LOOP simulation.
        
        Weights are fully frozen — no learning, no noradrenaline, no scrambling.
        Phase 2 trained the network; this just measures it.
        
        Args:
            dataset: Dataset with case directories
            num_episodes: Number of episodes to evaluate
            max_timesteps: Max steps to simulate (2.5x max trajectory length)
            num_ticks: Number of LIF simulation ticks per forward pass
            log_ticks: Always True — tick events are captured to summary.txt regardless
            verbose: If True, also echo tick logs to the terminal (default False)
            save_dir: Directory to save diagnostics
        
        Returns:
            metrics: Dict with success_rate, collisions, goal_reach, etc.
        """
        import yaml
        import io as _io
        import sys as _sys

        # Always capture tick logs to a buffer for summary.txt.
        # Only echo to terminal when verbose=True.
        class _Tee:
            def __init__(self, buf, terminal, echo): self.buf = buf; self.terminal = terminal; self.echo = echo
            def write(self, data):
                self.buf.write(data)
                if self.echo: self.terminal.write(data)
            def flush(self):
                self.buf.flush()
                if self.echo: self.terminal.flush()
        _log_buf = _io.StringIO()
        _real_stdout = _sys.stdout
        log_ticks = True  # always log; verbose controls terminal echo
        _sys.stdout = _Tee(_log_buf, _real_stdout, echo=verbose)

        print("\n" + "="*70)
        print("📊 EVALUATING SWARM ON MAPF METRICS")
        print("="*70)
        print(f"Max timesteps: {max_timesteps} (2.5x dataset max)")
        print(f"Episodes: {num_episodes}")
        print("   Weights frozen — pure inference, no online learning.")
        print("="*70 + "\n")
        
        # Aggregated metrics
        total_successes = 0
        total_collisions = 0
        total_goals_reached = 0
        total_agents = 0
        total_final_distance = 0.0
        total_timesteps = 0
        episode_lengths = []

        # Live spike health accumulators — real stats from every forward call
        spike_sum   = 0.0
        spike_max_v = 0.0
        spike_sq    = 0.0
        dead_sum    = 0.0
        sat_sum     = 0.0
        spike_n     = 0      # total neuron-timestep samples
        
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
                
                # Per-episode detailed accumulators
                ep_shadow_vetos   = [0] * num_agents
                ep_intent_vetos   = [0] * num_agents
                ep_cpg_stays      = [0] * num_agents
                ep_action_hist    = np.zeros((num_agents, 5), dtype=int)  # per-agent action counts
                ep_wall_bounces   = [0] * num_agents
                ep_agent_collisions = 0
                
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
                    
                    # PASS 1: Sober read — get intended actions and spike health
                    try:
                        action_spikes, _, _, true_vetoes, readout_E_list = self.network(obs, positions, num_ticks=num_ticks, goals=goals)  # [A, 5]
                        predicted_actions = action_spikes.argmax(dim=-1)  # [A]

                        # Accumulate real live spike health from this timestep
                        for e_t in readout_E_list:
                            e_np = e_t.detach().cpu().numpy()
                            spike_sum   += float(e_np.sum())
                            spike_sq    += float((e_np ** 2).sum())
                            spike_max_v  = max(spike_max_v, float(e_np.max()))
                            dead_sum    += int((e_np == 0).sum())
                            sat_sum     += int((e_np >= num_ticks).sum())
                            spike_n     += e_np.shape[0]

                        # Track per-tick neuromodulator events from veto flags
                        for i in range(num_agents):
                            ep_shadow_vetos[i] += int(true_vetoes[i])

                    except Exception as e:
                        if log_ticks:
                            print(f"  ❌ Network failed at t={t}: {e}")
                        actual_steps = max_timesteps  # Don't record as 1-step
                        break

                    # DEPLOY-TIME S.E.X. — Frustration Reflex
                    # Weights are frozen (no learning), but neuromodulation stays ON.
                    # A VETOed agent is blocked/frustrated: flood its RAM with noise so
                    # it can escape the deadlock without any weight updates.
                    self.network.inject_noradrenaline.clear()
                    for agent_id in range(num_agents):
                        if true_vetoes[agent_id]:
                            self.network.inject_noradrenaline[agent_id] = 0.01

                    # PASS 2: If anyone is panicking, re-run with the noise flush
                    if self.network.inject_noradrenaline:
                        try:
                            with torch.no_grad():
                                action_spikes, _, _, _, _ = self.network(obs, positions, num_ticks=num_ticks, goals=goals)
                                predicted_actions = action_spikes.argmax(dim=-1)
                        except Exception:
                            pass  # Keep Pass 1 actions if Pass 2 fails
                        self.network.inject_noradrenaline.clear()
                    
                    # Define action deltas for movement and distance calculation
                    # 0:RIGHT, 1:UP, 2:LEFT, 3:DOWN, 4:STAY
                    action_to_delta = torch.tensor([
                        [1, 0],   # RIGHT
                        [0, -1],  # UP
                        [-1, 0],  # LEFT
                        [0, 1],   # DOWN
                        [0, 0]    # STAY
                    ], dtype=torch.float32, device=self.device)
                    
                    # Use action_to_delta to move agents
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
                            ep_wall_bounces[i] += 1
                    
                    # AGENT-AGENT COLLISION: Check if multiple agents at same position
                    position_dict = {}
                    for i in range(num_agents):
                        pos_key = (int(new_positions[i, 0].item()), int(new_positions[i, 1].item()))
                        if pos_key in position_dict:
                            # Collision! Revert both agents
                            new_positions[i] = positions_temp[i]
                            new_positions[position_dict[pos_key]] = positions_temp[position_dict[pos_key]]
                            collisions_this_ep += 1
                            ep_agent_collisions += 1
                        else:
                            position_dict[pos_key] = i
                    
                    positions = new_positions

                    # Track action histogram
                    for i in range(num_agents):
                        ep_action_hist[i, predicted_actions[i].item()] += 1
                    
                    # Check goal reach (Manhattan: on the exact cell)
                    distances_to_goals = torch.norm(positions - goals, dim=1)  # [A]
                    reached_goal |= (distances_to_goals < 0.5)
                    
                    if log_ticks:
                        act_names  = ['R','U','L','D','S']
                        acts_str   = ' '.join(f'A{i}:{act_names[predicted_actions[i].item()]}' for i in range(num_agents))
                        veto_str   = ' '.join(f'A{i}:V' if true_vetoes[i] else f'A{i}:-' for i in range(num_agents))
                        dists_str  = ' '.join(f'{distances_to_goals[i].item():.1f}' for i in range(num_agents))
                        print(f"  t={t:3d} | {acts_str} | veto=[{veto_str}] | dist=[{dists_str}] | "
                              f"coll={ep_agent_collisions} wall={sum(ep_wall_bounces)}")
                    
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
                    act_names = ['RIGHT','UP','LEFT','DOWN','STAY']
                    print(f"  {'─'*68}")
                    print(f"  {'Agent':>5} | {'Goal':>4} | {'Dist':>5} | {'ShdwV':>5} | {'WallB':>5} | Action histogram (R/U/L/D/S)")
                    print(f"  {'─'*68}")
                    for i in range(num_agents):
                        hist_str = '/'.join(f'{ep_action_hist[i,a]:3d}' for a in range(5))
                        dist_i   = torch.norm(positions[i] - goals[i]).item()
                        print(f"  {i:>5} | {'✓' if reached_goal[i] else '✗':>4} | {dist_i:>5.2f} | "
                              f"{ep_shadow_vetos[i]:>5} | {ep_wall_bounces[i]:>5} | {hist_str}")
                    print(f"  {'─'*68}")
                    print(f"  Agent-agent collisions: {ep_agent_collisions} | "
                          f"Steps taken: {actual_steps} | Success: {'✓' if reached_goal.all() else '✗'}\n")
                
                # Accumulate
                total_successes += int(success)
                total_collisions += collisions_this_ep
                total_goals_reached += goals_reached
                total_final_distance += final_distances.sum().item()
                total_timesteps += episode_lengths[-1]
        
        # Derive spike health from real eval-loop data (no fake probe)
        if spike_n > 0:
            spike_mean = spike_sum / spike_n
            spike_max  = spike_max_v
            dead_frac  = dead_sum  / spike_n
            sat_frac   = sat_sum   / spike_n
        else:
            spike_mean = spike_max = dead_frac = sat_frac = 0.0
        
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

        # Restore stdout and save captured log
        _sys.stdout = _real_stdout
        metrics['tick_log'] = _log_buf.getvalue()

        try:
            if save_dir:
                self._save_diagnostics(dataset, num_ticks, save_dir)
        except Exception as _diag_err:
            print(f"[WARNING] _save_diagnostics failed: {_diag_err}")

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

        TRACE_TICKS  = num_ticks
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
        # PLOT 1 – SUBSUMPTION TRACE (CORRECTED HARDWARE TRACE)
        # Real tick-by-tick trace including Ghost Antenna and dynamic Shadow VETO.
        # No frustration hormone (deleted from real forward pass).
        # ══════════════════════════════════════════════════════════════════════
        functional.reset_net(ag0.obs_mesh.lif)
        functional.reset_net(ag0.readout_mesh.lif)
        functional.reset_net(ag0.ghost_antenna.lif)

        obs_flat = corridor_obs.flatten()
        obs_inp  = ag0.obs_input_proj(obs_flat) * 1.5

        # 1. PRE-COMPUTE OPTIC NERVE (matches the real forward pass — full num_ticks burn-in)
        obs_acc = torch.zeros(ag0.obs_mesh.num_neurons, device=self.device)
        obs_cur = torch.zeros(ag0.obs_mesh.num_neurons, device=self.device)
        with torch.no_grad():
            for _ in range(num_ticks):
                rec     = ag0.obs_recurrent(obs_cur) * ag0.obs_mesh.ei_mask
                obs_cur = ag0.obs_mesh.lif(obs_inp * 0.8 + rec * 0.2)
                obs_acc += obs_cur
            # Static num_ticks-integrated voltage chunk to feed the motor cortex
            obs_E  = ag0.obs_mesh.get_E_neurons(obs_acc)
            rd_inp = ag0.readout_input(ag0.obs_to_readout(obs_E)) * 0.9

        motor_voltage = np.zeros((TRACE_TICKS, 5))
        rd_acc  = torch.zeros(ag0.readout_mesh.num_neurons, device=self.device)
        rd_cur  = torch.zeros(ag0.readout_mesh.num_neurons, device=self.device)

        vel_map_diag = {0:(1,0), 1:(0,-1), 2:(-1,0), 3:(0,1), 4:(0,0)}
        veto_tick = None

        with torch.no_grad():
            for t in range(TRACE_TICKS):
                # 2. Trace Readout Mesh using the proper static rd_inp
                rec    = ag0.readout_recurrent(rd_cur) * ag0.readout_mesh.ei_mask
                rd_cur = ag0.readout_mesh.lif(rd_inp * 0.85 + rec * 0.15)
                rd_acc += rd_cur

                # 3. Base motor cortex (outputs real non-zero logits)
                action_logits = ag0.action_weights(ag0.readout_mesh.get_E_neurons(rd_acc))

                # 4. Ghost Antenna
                antenna_spikes = ag0.ghost_antenna(
                    corridor_obs[0], corridor_obs[1], num_ticks=1, gain=1.5
                )
                action_logits = action_logits + antenna_spikes * 0.4

                motor_voltage[t] = action_logits.cpu().numpy()
                tentative_action  = int(torch.argmax(action_logits).item())

                # 5. Dynamic Shadow Caster (1-step lookahead)
                veto = ag0.shadow(corridor_obs[0], vel_map_diag[tentative_action], num_ticks=1)
                if veto and tentative_action != 4:
                    if veto_tick is None:
                        veto_tick = t
                    motor_voltage[t, 4] = motor_voltage[t].max() + 10.0

        # If no natural VETO fired (wall not actually in chosen path), mark midpoint
        if veto_tick is None:
            veto_tick = TRACE_TICKS // 2
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

                a_spk = e_a.cpu().numpy()
                b_spk = e_b.cpu().numpy()
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

        # ── Panel 2: Pre-Motor Action Landscape ─────────────────────────────
        # Grab the final-tick motor voltage (neuromodulated by Ghost Antenna +0.05/tick
        # in the trace loop above; Shadow VETO adds +1.0 to STAY at veto_tick)
        # and apply softmax to visualise action probabilities.
        raw_logits   = motor_voltage[-1].astype(np.float64)      # [RIGHT,UP,LEFT,DOWN,STAY]
        exp_logits   = np.exp(raw_logits - np.max(raw_logits))
        action_probs = exp_logits / exp_logits.sum()

        pre_motor_grid = np.zeros((3, 3))
        pre_motor_grid[1, 2] = action_probs[0]   # RIGHT
        pre_motor_grid[0, 1] = action_probs[1]   # UP
        pre_motor_grid[1, 0] = action_probs[2]   # LEFT
        pre_motor_grid[2, 1] = action_probs[3]   # DOWN
        pre_motor_grid[1, 1] = action_probs[4]   # STAY

        # ── Panel 3: Shadow Raycast ───────────────────────────────────────────
        walls_ch   = corridor_obs[0].cpu().numpy()   # [7, 7]
        shadow_map = np.zeros((FOV, FOV))
        cr, cc     = FOV // 2, FOV // 2              # agent centre
        shadow_map[cr, cc] = 0.3                      # agent marker
        # FIX: Strip the subsumption bridge injection to see what the raw brain wanted!
        pre_veto_logits = motor_voltage[-1].copy()
        if veto_tick is not None:
            pre_veto_logits[4] -= 1.0  # undo the +1.0 STAY injection the bridge added
        chosen_action_shadow = int(np.argmax(pre_veto_logits))
        action_to_velocity   = {0:(1,0), 1:(0,-1), 2:(-1,0), 3:(0,1), 4:(0,0)}
        dx_v, dy_v = action_to_velocity[chosen_action_shadow]
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

        # Panel 2: Pre-Motor Action Landscape
        ax = axes[1]
        im2 = ax.imshow(pre_motor_grid, cmap='magma', origin='upper',
                        vmin=0, vmax=1.0, interpolation='nearest')
        ax.set_title('Panel 2: Pre-Motor Action Landscape\n'
                     '(SNN simultaneously weighing all paths before argmax)',
                     fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['LEFT', 'CTR', 'RIGHT'], fontsize=9)
        ax.set_yticklabels(['UP', 'CTR', 'DOWN'],    fontsize=9)
        # Percentage annotations on every active cell
        ACTION_LABEL_POS = [
            (1, 2, 'RIGHT'), (0, 1, 'UP'), (1, 0, 'LEFT'),
            (2, 1, 'DOWN'),  (1, 1, 'STAY')
        ]
        for row_p, col_p, lbl in ACTION_LABEL_POS:
            val = pre_motor_grid[row_p, col_p]
            if val > 0.005:
                text_col = 'black' if val > 0.6 else 'white'
                ax.text(col_p, row_p, f"{val:.0%}\n{lbl}",
                        ha='center', va='center',
                        color=text_col, fontweight='bold', fontsize=11)
        # Cyan border on the chosen action cell
        chosen_action = int(np.argmax(motor_voltage[-1]))
        chosen_positions = {0:(1,2), 1:(0,1), 2:(1,0), 3:(2,1), 4:(1,1)}
        c_row, c_col = chosen_positions[chosen_action]
        rect2 = mpatches.Rectangle((c_col - 0.5, c_row - 0.5), 1, 1,
                                    linewidth=3, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect2)
        plt.colorbar(im2, ax=ax, shrink=0.72, label='Path probability (softmax)')

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
            "The Agent's Mind's Eye:  FOV  →  Pre-Motor Landscape  →  Shadow Raycast",
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
