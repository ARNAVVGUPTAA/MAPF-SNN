"""
Neurotransmitter Receptor Dynamics for Biologically-Inspired SNNs
================================================================

This module implements the four major receptor types found in the brain:
- AMPA: Fast excitatory (glutamate), τ ≈ 2ms
- NMDA: Slow excitatory (glutamate), τ ≈ 50ms, voltage-dependent
- GABAA: Fast inhibitory (GABA), τ ≈ 6ms
- GABAB: Slow inhibitory (GABA), τ ≈ 150ms

Key Features:
- Proper time constants for each receptor type
- Conductance-based currents (not artificial multiplication)
- NMDA voltage-dependent Mg2+ block
- Neuromodulation: Dopamine → AMPA/NMDA, GABA → GABAA/GABAB
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class ReceptorDynamics(nn.Module):
    """
    Models the four major neurotransmitter receptors with proper dynamics.
    
    Receptors respond to presynaptic spikes and decay exponentially:
    - AMPA: Fast excitation for immediate responses
    - NMDA: Slow excitation for learning and temporal integration
    - GABAA: Fast inhibition for rapid spike control
    - GABAB: Slow inhibition for network stabilization
    """
    
    def __init__(self, 
                 dt: float = 1.0,  # Timestep in ms
                 v_rest: float = -70.0,  # Resting potential (mV)
                 v_threshold: float = -55.0,  # Spike threshold (mV) - lowered to increase excitability
                 e_ampa: float = 0.0,  # AMPA reversal potential (mV)
                 e_nmda: float = 0.0,  # NMDA reversal potential (mV)
                 e_gabaa: float = -70.0,  # GABAA reversal potential (mV)
                 e_gabab: float = -90.0):  # GABAB reversal potential (mV)
        super().__init__()
        
        self.dt = dt
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        
        # Reversal potentials (driving forces)
        self.e_ampa = e_ampa
        self.e_nmda = e_nmda
        self.e_gabaa = e_gabaa
        self.e_gabab = e_gabab
        
        # Time constants (in ms) - biologically realistic
        self.tau_ampa = 1.0   # Fast excitation
        self.tau_nmda = 25.0  # Slow excitation
        self.tau_gabaa = 3.0  # Fast inhibition
        self.tau_gabab = 75.0  # Slow inhibition
        
        # Decay factors (exponential decay per timestep)
        self.decay_ampa = torch.exp(torch.tensor(-dt / self.tau_ampa))
        self.decay_nmda = torch.exp(torch.tensor(-dt / self.tau_nmda))
        self.decay_gabaa = torch.exp(torch.tensor(-dt / self.tau_gabaa))
        self.decay_gabab = torch.exp(torch.tensor(-dt / self.tau_gabab))
        
        # Maximum conductances (learnable parameters) - balanced for E/I ratio ~3-5
        self.g_max_ampa = nn.Parameter(torch.tensor(4.5))   # Fast excitation
        self.g_max_nmda = nn.Parameter(torch.tensor(2.0))   # Slow excitation
        self.g_max_gabaa = nn.Parameter(torch.tensor(7.5))  # Boosted from 0.8 to balance E/I
        self.g_max_gabab = nn.Parameter(torch.tensor(3.9))  # Boosted from 0.3 to balance E/I
        
        # NMDA voltage-dependent Mg2+ block parameters
        self.nmda_mg_conc = 1.0  # mM
        self.nmda_alpha = 0.062  # Voltage dependence
        self.nmda_beta = 3.57  # Mg2+ sensitivity
        
        # State variables (will be initialized per batch)
        self.register_buffer('g_ampa', None)
        self.register_buffer('g_nmda', None)
        self.register_buffer('g_gabaa', None)
        self.register_buffer('g_gabab', None)
    
    def init_state(self, batch_size: int, num_neurons: int, device: torch.device):
        """Initialize receptor conductance states"""
        self.g_ampa = torch.zeros(batch_size, num_neurons, device=device)
        self.g_nmda = torch.zeros(batch_size, num_neurons, device=device)
        self.g_gabaa = torch.zeros(batch_size, num_neurons, device=device)
        self.g_gabab = torch.zeros(batch_size, num_neurons, device=device)
    
    def reset_state(self):
        """Reset all receptor states to zero"""
        if self.g_ampa is not None:
            self.g_ampa.zero_()
            self.g_nmda.zero_()
            self.g_gabaa.zero_()
            self.g_gabab.zero_()
    
    def compute_nmda_gate(self, v_membrane: torch.Tensor) -> torch.Tensor:
        """
        Compute voltage-dependent NMDA Mg2+ block.
        
        At rest (-70mV): Gate ≈ 0.1 (mostly blocked)
        At threshold (-50mV): Gate ≈ 0.5 (partially open)
        At depolarized states (>-40mV): Gate → 1.0 (fully open)
        
        This creates Hebbian learning: NMDA only conducts when:
        1. Presynaptic glutamate is present (NMDA receptors activated)
        2. Postsynaptic neuron is depolarized (Mg2+ block removed)
        """
        mg_block = 1.0 / (1.0 + self.nmda_mg_conc * torch.exp(-self.nmda_alpha * v_membrane) / self.nmda_beta)
        return mg_block
    
    def update_receptors(self, 
                        excitatory_input: torch.Tensor,
                        inhibitory_input: torch.Tensor,
                        dopamine_level: float = 1.0,
                        gaba_level: float = 1.0) -> None:
        """
        Update receptor conductances based on presynaptic inputs.
        
        Args:
            excitatory_input: Excitatory presynaptic activity [batch, neurons]
            inhibitory_input: Inhibitory presynaptic activity [batch, neurons]
            dopamine_level: Dopamine neuromodulation (0.5-2.0 typical)
            gaba_level: GABA neuromodulation (0.5-2.0 typical)
        """
        # Exponential decay
        self.g_ampa = self.g_ampa * self.decay_ampa
        self.g_nmda = self.g_nmda * self.decay_nmda
        self.g_gabaa = self.g_gabaa * self.decay_gabaa
        self.g_gabab = self.g_gabab * self.decay_gabab
        
        # Add new conductance from spikes (modulated by neurotransmitters)
        # Dopamine enhances excitatory transmission (D1 receptor effect)
        dopamine_modulation = dopamine_level
        self.g_ampa = self.g_ampa + self.g_max_ampa * excitatory_input * dopamine_modulation
        self.g_nmda = self.g_nmda + self.g_max_nmda * excitatory_input * dopamine_modulation
        
        # GABA enhances inhibitory transmission
        gaba_modulation = gaba_level
        self.g_gabaa = self.g_gabaa + self.g_max_gabaa * inhibitory_input * gaba_modulation
        self.g_gabab = self.g_gabab + self.g_max_gabab * inhibitory_input * gaba_modulation
    
    def compute_currents(self, v_membrane: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute synaptic currents from receptor conductances.
        
        Current = Conductance × (V_membrane - E_reversal)
        
        Returns:
            Dictionary with individual and total currents
        """
        # AMPA current (fast excitation)
        i_ampa = self.g_ampa * (v_membrane - self.e_ampa)
        
        # NMDA current (slow excitation with voltage gating)
        nmda_gate = self.compute_nmda_gate(v_membrane)
        i_nmda = self.g_nmda * nmda_gate * (v_membrane - self.e_nmda)
        
        # GABAA current (fast inhibition)
        i_gabaa = self.g_gabaa * (v_membrane - self.e_gabaa)
        
        # GABAB current (slow inhibition)
        i_gabab = self.g_gabab * (v_membrane - self.e_gabab)
        
        # Total excitatory and inhibitory currents
        i_excitatory = i_ampa + i_nmda
        i_inhibitory = i_gabaa + i_gabab
        
        # Net current: excitatory depolarizes (+), inhibitory hyperpolarizes (-)
        # Since current flows FROM reversal TO membrane: I = g * (E - V)
        # At rest (-70mV): E_AMPA (0mV) - V(-70mV) = +70mV → positive current → depolarize
        # At rest (-70mV): E_GABA (-70mV) - V(-70mV) = 0 → no current (need to go below -70)
        i_total = i_excitatory - i_inhibitory
        
        return {
            'i_ampa': i_ampa,
            'i_nmda': i_nmda,
            'i_gabaa': i_gabaa,
            'i_gabab': i_gabab,
            'i_excitatory': i_excitatory,
            'i_inhibitory': i_inhibitory,
            'i_total': i_total
        }
    
    def get_conductances(self) -> Dict[str, torch.Tensor]:
        """Get current conductance states for monitoring"""
        # Return zeros if not initialized
        if self.g_ampa is None:
            device = next(self.parameters()).device
            zero = torch.tensor(0.0, device=device)
            return {
                'g_ampa': zero,
                'g_nmda': zero,
                'g_gabaa': zero,
                'g_gabab': zero
            }
        
        return {
            'g_ampa': self.g_ampa.clone(),
            'g_nmda': self.g_nmda.clone(),
            'g_gabaa': self.g_gabaa.clone(),
            'g_gabab': self.g_gabab.clone()
        }
    
    def get_ei_ratio(self) -> torch.Tensor:
        """
        Compute excitation/inhibition ratio.
        
        Healthy E/I ratio: ~1.0-1.5
        Too high (>2.0): Network may become unstable
        Too low (<0.5): Network may not spike enough
        """
        # Return default if not initialized
        if self.g_ampa is None:
            return torch.tensor(1.0)
        
        total_excitation = self.g_ampa.mean() + self.g_nmda.mean()
        total_inhibition = self.g_gabaa.mean() + self.g_gabab.mean()
        
        # Avoid division by zero
        ei_ratio = total_excitation / (total_inhibition + 1e-6)
        return ei_ratio


class ReceptorLIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with receptor-based dynamics.
    
    This replaces the complex EILIFLayer with a simpler, more biologically
    accurate model that uses proper neurotransmitter receptors.
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dt: float = 1.0,
                 tau_mem: float = 20.0,  # Membrane time constant (ms)
                 v_threshold: float = 1.0,  # Normalized threshold
                 v_reset: float = 0.0,
                 v_rest: float = 0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dt = dt
        self.tau_mem = tau_mem
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.debug = True  # Enable debugging
        self.forward_count = 0
        
        # Membrane decay
        self.decay_mem = torch.exp(torch.tensor(-dt / tau_mem))
        
        # Input projections (80% excitatory, 20% inhibitory - Dale's principle)
        self.n_excitatory = int(output_dim * 0.8)
        self.n_inhibitory = output_dim - self.n_excitatory
        
        # Excitatory pathway
        self.w_excitatory = nn.Linear(input_dim, self.n_excitatory)
        
        # Inhibitory pathway
        self.w_inhibitory = nn.Linear(input_dim, self.n_inhibitory)
        
        # Receptor dynamics
        self.receptors = ReceptorDynamics(dt=dt)
        
        # State variables
        self.register_buffer('v_membrane', None)
        self.register_buffer('spikes', None)
    
    def init_state(self, batch_size: int, device: torch.device):
        """Initialize membrane potential and receptor states"""
        self.v_membrane = torch.ones(batch_size, self.output_dim, device=device) * self.v_rest
        self.spikes = torch.zeros(batch_size, self.output_dim, device=device)
        self.receptors.init_state(batch_size, self.output_dim, device)
    
    def reset_state(self):
        """Reset all states"""
        if self.v_membrane is not None:
            self.v_membrane.fill_(self.v_rest)
            self.spikes.zero_()
            self.receptors.reset_state()
    
    def forward(self, 
                x: torch.Tensor,
                dopamine_level: float = 1.0,
                gaba_level: float = 1.0) -> torch.Tensor:
        """
        Forward pass with receptor-based dynamics.
        
        Args:
            x: Input tensor [batch, input_dim]
            dopamine_level: Dopamine modulation (affects AMPA/NMDA)
            gaba_level: GABA modulation (affects GABAA/GABAB)
        
        Returns:
            spikes: Output spikes [batch, output_dim]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize states if needed
        if self.v_membrane is None or self.v_membrane.shape[0] != batch_size:
            self.init_state(batch_size, device)
        
        # Compute excitatory and inhibitory inputs
        # Use stronger activation: tanh gives [-1, 1], then shift to [0, 2] for non-negative conductance drive
        excitatory_input = torch.relu(self.w_excitatory(x))  # [batch, n_excitatory] - allow strong positive
        inhibitory_input = torch.relu(self.w_inhibitory(x))  # [batch, n_inhibitory]
        
        # Pad to full output dimension
        excitatory_padded = torch.cat([
            excitatory_input, 
            torch.zeros(batch_size, self.n_inhibitory, device=device)
        ], dim=1)
        inhibitory_padded = torch.cat([
            torch.zeros(batch_size, self.n_excitatory, device=device),
            inhibitory_input
        ], dim=1)
        
        # Update receptor conductances
        self.receptors.update_receptors(
            excitatory_padded, 
            inhibitory_padded,
            dopamine_level,
            gaba_level
        )
        
        # Compute synaptic currents
        currents = self.receptors.compute_currents(self.v_membrane)
        i_synaptic = currents['i_total']
        
        # Add baseline drive current to help initial spiking (biological background activity)
        # Target spike rate: 10-30% for good learning (not 90%+ saturation)
        baseline_current = 0.01  # Sweet spot: prevents both saturation and silence
        
        # Membrane dynamics: dV/dt = (V_rest - V) / tau + I_synaptic + baseline
        self.v_membrane = (self.v_membrane * self.decay_mem + 
                          (1.0 - self.decay_mem) * self.v_rest + 
                          (i_synaptic + baseline_current) * self.dt)
        
        # Spike generation
        self.spikes = (self.v_membrane >= self.v_threshold).float()
        
        # Reset membrane potential for spiked neurons
        self.v_membrane = torch.where(
            self.spikes.bool(),
            torch.ones_like(self.v_membrane) * self.v_reset,
            self.v_membrane
        )
        
        return self.spikes
    
    def get_spike_rate(self) -> float:
        """Get current spike rate for monitoring"""
        if self.spikes is not None:
            return self.spikes.mean().item()
        return 0.0
    
    def get_ei_ratio(self) -> float:
        """Get excitation/inhibition ratio"""
        return self.receptors.get_ei_ratio().item()
    
    def get_receptor_stats(self) -> Dict[str, float]:
        """Get detailed receptor statistics"""
        conductances = self.receptors.get_conductances()
        return {
            'g_ampa_mean': conductances['g_ampa'].mean().item(),
            'g_nmda_mean': conductances['g_nmda'].mean().item(),
            'g_gabaa_mean': conductances['g_gabaa'].mean().item(),
            'g_gabab_mean': conductances['g_gabab'].mean().item(),
            'ei_ratio': self.get_ei_ratio(),
            'spike_rate': self.get_spike_rate()
        }


# Convenience function for creating receptor-based layers
def create_receptor_layer(input_dim: int, 
                         output_dim: int,
                         config: dict = None) -> ReceptorLIFNeuron:
    """
    Create a receptor-based LIF layer with config overrides.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        config: Optional config dictionary with parameters
    
    Returns:
        ReceptorLIFNeuron instance
    """
    if config is None:
        config = {}
    
    return ReceptorLIFNeuron(
        input_dim=input_dim,
        output_dim=output_dim,
        dt=config.get('lif_dt', 1.0),
        tau_mem=config.get('lif_tau', 20.0),
        v_threshold=config.get('lif_v_threshold', 1.0),
        v_reset=config.get('lif_v_reset', 0.0),
        v_rest=config.get('lif_v_rest', 0.0)
    )
