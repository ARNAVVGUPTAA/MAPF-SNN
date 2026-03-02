"""
Trainable Neuromorphic LSM for MAPF
====================================

BDSM: BRAIN DON'T SIMPLY MULTIPLY! 🧠⚡

Dual-mode operation:
- Forward: Neuromorphic-friendly (bit-shifts, additions, sparse)
- Backward: Gradient flow (surrogate gradients, straight-through estimators)

This allows:
✅ Training with SGD/BPTT
✅ Deployment to neuromorphic hardware
✅ Best of both worlds!

Compatible with: Loihi, TrueNorth, SpiNNaker (after deployment quantization)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional

# Import neuromorphic operations with surrogate gradients
from .neuromorphic_ops import (
    NeuromorphicEILIF,
    QuantizedLinear,
    QuantizedSpikeLinear
)


class TrainableNeuromorphicMesh(nn.Module):
    """
    Trainable neuromorphic mesh with surrogate gradients.
    
    Forward: Neuromorphic-friendly operations (quantized weights, bit-shift decay)
    Backward: Gradient flow through surrogate gradients
    
    Weights are TRAINABLE with quantization-aware training!
    """
    
    def __init__(self, 
                 num_neurons=256,
                 E_ratio=0.8,
                 intra_E_density=0.5,
                 intra_I_density=0.7,
                 E_to_I_density=0.6,
                 I_to_E_density=0.8,
                 v_threshold=1.0,
                 v_reset=0.0,
                 tau_E=2.0,
                 tau_I=1.5,
                 dt=1.0,
                 weight_bits=4,
                 weight_scale=8.0,
                 surrogate_type='fast_sigmoid',
                 surrogate_beta=10.0):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.E_ratio = E_ratio
        
        # EILIF neurons with surrogate gradients
        self.neurons = NeuromorphicEILIF(
            num_neurons=num_neurons,
            E_ratio=E_ratio,
            v_threshold=v_threshold,
            v_reset=v_reset,
            tau_E=tau_E,
            tau_I=tau_I,
            dt=dt,
            surrogate_type=surrogate_type,
            surrogate_beta=surrogate_beta
        )
        
        num_E = self.neurons.num_E
        num_I = self.neurons.num_I
        
        print(f"      Quantized weights: {weight_bits}-bit, scale=±{weight_scale}")
        
        # Quantized synaptic connections (TRAINABLE!)
        # These use straight-through estimators for gradients
        
        # E→E: Excitatory recurrence
        self.syn_EE = self._create_sparse_connection(
            num_E, num_E, intra_E_density, weight_bits, weight_scale
        )
        
        # E→I: Drive inhibition
        self.syn_EI = self._create_sparse_connection(
            num_E, num_I, E_to_I_density, weight_bits, weight_scale
        )
        
        # I→E: Inhibitory feedback
        self.syn_IE = self._create_sparse_connection(
            num_I, num_E, I_to_E_density, weight_bits, weight_scale
        )
        
        # I→I: Inhibitory competition
        self.syn_II = self._create_sparse_connection(
            num_I, num_I, intra_I_density, weight_bits, weight_scale
        )
        
        print(f"      Connectivity: EE={intra_E_density:.1%}, II={intra_I_density:.1%}, " +
              f"EI={E_to_I_density:.1%}, IE={I_to_E_density:.1%}")
    
    def _create_sparse_connection(self, in_features, out_features, density, 
                                  weight_bits, weight_scale):
        """Create sparse quantized connection"""
        # Use QuantizedSpikeLinear for spike-driven computation
        layer = QuantizedSpikeLinear(in_features, out_features, 
                                     weight_bits=weight_bits, 
                                     weight_scale=weight_scale)
        
        # Initialize with sparsity
        with torch.no_grad():
            mask = torch.rand(out_features, in_features) < density
            mask.fill_diagonal_(False)
            layer.linear.weight.data *= mask.float()
            
            # Normalize to reasonable range
            layer.linear.weight.data *= 0.1 / weight_scale
        
        return layer
    
    def forward(self, external_input, e_spikes_prev, i_spikes_prev):
        """
        Forward with neuromorphic operations + gradient flow.
        
        Args:
            external_input: [batch, num_neurons]
            e_spikes_prev: [batch, num_E] previous E spikes
            i_spikes_prev: [batch, num_I] previous I spikes
        
        Returns:
            spikes, e_spikes, i_spikes
        """
        num_E = self.neurons.num_E
        num_I = self.neurons.num_I
        
        # Split external input
        current_E = external_input[:, :num_E]
        current_I = external_input[:, num_E:]
        
        if e_spikes_prev is not None and i_spikes_prev is not None:
            # Sparse spike-driven accumulation (event-based!)
            # Uses quantized weights with gradient flow
            
            # E→E: Excitatory recurrence
            i_EE = self.syn_EE(e_spikes_prev)
            current_E = current_E + i_EE
            
            # I→E: Inhibition
            i_IE = self.syn_IE(i_spikes_prev)
            current_E = current_E + i_IE
            
            # E→I: Drive inhibition
            i_EI = self.syn_EI(e_spikes_prev)
            current_I = current_I + i_EI
            
            # I→I: Inhibitory competition
            i_II = self.syn_II(i_spikes_prev)
            current_I = current_I + i_II
        
        # Combine currents
        total_current = torch.cat([current_E, current_I], dim=1)
        
        # Update neurons (bit-shift decay + surrogate spike function)
        spikes, e_spikes, i_spikes = self.neurons(total_current)
        
        return spikes, e_spikes, i_spikes
    
    def reset_state(self):
        """Reset neuron states"""
        self.neurons.reset_state()
    
    def get_quantized_weights(self):
        """Get integer weights for deployment"""
        return {
            'W_EE': self.syn_EE.linear.get_integer_weights(),
            'W_EI': self.syn_EI.linear.get_integer_weights(),
            'W_IE': self.syn_IE.linear.get_integer_weights(),
            'W_II': self.syn_II.linear.get_integer_weights(),
        }


class TrainableNeuromorphicCircularReservoir(nn.Module):
    """
    Trainable neuromorphic circular LSM with surrogate gradients.
    
    Forward: Neuromorphic-friendly (quantized, sparse, bit-shifts)
    Backward: Full gradient flow (end-to-end trainable!)
    
    Can be trained with SGD/Adam/BPTT like a standard RNN.
    """
    
    def __init__(self, config):
        super().__init__()
        
        lsm_cfg = config['lsm']
        
        self.num_meshes = lsm_cfg['num_meshes']
        self.neurons_per_mesh = lsm_cfg['neurons_per_mesh']
        self.num_ticks = lsm_cfg['num_ticks']
        self.E_ratio = lsm_cfg['E_ratio']
        
        self.total_neurons = self.num_meshes * self.neurons_per_mesh
        
        print(f"\n🧠 TRAINABLE Neuromorphic Circular Mesh LSM")
        print(f"   BDSM: BRAIN DON'T SIMPLY MULTIPLY! ⚡")
        print(f"   Architecture: {self.num_meshes} meshes × {self.neurons_per_mesh}N = {self.total_neurons}N")
        print(f"   ✅ Neuromorphic forward pass (quantized, sparse)")
        print(f"   ✅ Gradient flow backward (surrogate gradients)")
        print(f"   ✅ End-to-end trainable with SGD!")
        
        # Input dimension
        fov_size = config.get('fov_size', 7)
        fov_channels = config.get('fov_channels', 2)
        self.input_dim = fov_channels * fov_size * fov_size
        
        # Get neuron parameters
        v_threshold = lsm_cfg.get('v_threshold', 1.0)
        v_reset = lsm_cfg.get('v_reset', 0.0)
        tau_E = lsm_cfg.get('tau_E', 2.0)
        tau_I = lsm_cfg.get('tau_I', 1.5)
        dt = lsm_cfg.get('dt', 1.0)
        
        # Quantization parameters
        weight_bits = lsm_cfg.get('weight_bits', 4)
        weight_scale = lsm_cfg.get('weight_scale', 8.0)
        input_scale = lsm_cfg.get('input_scale', 1.0)
        
        self.input_scale = input_scale
        
        # Surrogate gradient settings
        surrogate_type = lsm_cfg.get('surrogate_type', 'fast_sigmoid')
        surrogate_beta = lsm_cfg.get('surrogate_beta', 10.0)
        
        print(f"   Neuron: τ_E={tau_E}ms, τ_I={tau_I}ms, threshold={v_threshold}")
        print(f"   Quantization: {weight_bits}-bit weights, scale=±{weight_scale}")
        print(f"   Surrogate: {surrogate_type}, β={surrogate_beta}")
        
        # Create trainable meshes
        self.meshes = nn.ModuleList([
            TrainableNeuromorphicMesh(
                num_neurons=self.neurons_per_mesh,
                E_ratio=lsm_cfg['E_ratio'],
                intra_E_density=lsm_cfg['intra_mesh_E_density'],
                intra_I_density=lsm_cfg['intra_mesh_I_density'],
                E_to_I_density=lsm_cfg['E_to_I_density'],
                I_to_E_density=lsm_cfg['I_to_E_density'],
                v_threshold=v_threshold,
                v_reset=v_reset,
                tau_E=tau_E,
                tau_I=tau_I,
                dt=dt,
                weight_bits=weight_bits,
                weight_scale=weight_scale,
                surrogate_type=surrogate_type,
                surrogate_beta=surrogate_beta
            )
            for _ in range(self.num_meshes)
        ])
        
        # Input projection (trainable quantized weights)
        self.input_projections = nn.ModuleList([
            QuantizedLinear(self.input_dim, self.neurons_per_mesh,
                           weight_bits=weight_bits, weight_scale=weight_scale)
            for _ in range(self.num_meshes)
        ])
        
        # Inter-mesh connections (trainable, E-only)
        inter_density = lsm_cfg['inter_mesh_E_density']
        num_E_per_mesh = int(self.neurons_per_mesh * self.E_ratio)
        
        self.inter_connections = nn.ModuleList([
            self._create_inter_connection(num_E_per_mesh, inter_density, 
                                         weight_bits, weight_scale)
            for _ in range(self.num_meshes)
        ])
        
        print(f"   Inter-mesh: E-only, density={inter_density:.1%} (circular)")
        
        # State tracking
        self.mesh_e_spikes = [None] * self.num_meshes
        self.mesh_i_spikes = [None] * self.num_meshes
    
    def _create_inter_connection(self, num_E, density, weight_bits, weight_scale):
        """Create sparse inter-mesh connection"""
        layer = QuantizedLinear(num_E, self.neurons_per_mesh,
                               weight_bits=weight_bits, weight_scale=weight_scale)
        
        # Initialize with sparsity
        with torch.no_grad():
            mask = torch.rand(self.neurons_per_mesh, num_E) < density
            layer.weight.data *= mask.float()
            layer.weight.data *= 0.05 / weight_scale  # Weak inter-mesh
        
        return layer
    
    def reset_state(self):
        """Reset all mesh states"""
        for mesh in self.meshes:
            mesh.reset_state()
        self.mesh_e_spikes = [None] * self.num_meshes
        self.mesh_i_spikes = [None] * self.num_meshes
    
    def forward(self, x, return_all_states=False):
        """
        Forward pass with gradient flow.
        
        Args:
            x: [batch, input_dim] input features
        
        Returns:
            liquid_state: [batch, total_neurons] accumulated spikes
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Scale input (but keep as float - quantization happens in layers)
        x_scaled = x * self.input_scale
        
        # Initialize spike states
        if self.mesh_e_spikes[0] is None:
            num_E = int(self.neurons_per_mesh * self.E_ratio)
            num_I = self.neurons_per_mesh - num_E
            for i in range(self.num_meshes):
                self.mesh_e_spikes[i] = torch.zeros(batch_size, num_E, device=device)
                self.mesh_i_spikes[i] = torch.zeros(batch_size, num_I, device=device)
        
        # Spike accumulator
        spike_accumulator = torch.zeros(batch_size, self.total_neurons, device=device)
        
        if return_all_states:
            all_states = []
        
        # Simulate for multiple ticks (BPTT through time)
        for tick in range(self.num_ticks):
            new_e_spikes = []
            new_i_spikes = []
            
            # Process each mesh
            for mesh_idx, mesh in enumerate(self.meshes):
                # Input projection (quantized weights, gradient flow)
                external_input = self.input_projections[mesh_idx](x_scaled)
                
                # Inter-mesh input (from previous mesh in circle)
                prev_mesh_idx = (mesh_idx - 1) % self.num_meshes
                if self.mesh_e_spikes[prev_mesh_idx] is not None:
                    inter_input = self.inter_connections[prev_mesh_idx](
                        self.mesh_e_spikes[prev_mesh_idx]
                    )
                    external_input = external_input + inter_input
                
                # Process mesh (surrogate gradients enable backprop!)
                spikes, e_spikes, i_spikes = mesh(
                    external_input,
                    self.mesh_e_spikes[mesh_idx],
                    self.mesh_i_spikes[mesh_idx]
                )
                
                new_e_spikes.append(e_spikes)
                new_i_spikes.append(i_spikes)
                
                # Accumulate spikes
                start_idx = mesh_idx * self.neurons_per_mesh
                end_idx = start_idx + self.neurons_per_mesh
                spike_accumulator[:, start_idx:end_idx] += spikes
            
            # Update spike states
            self.mesh_e_spikes = new_e_spikes
            self.mesh_i_spikes = new_i_spikes
            
            if return_all_states:
                all_states.append(spike_accumulator.clone())
        
        if return_all_states:
            return torch.stack(all_states, dim=1)
        else:
            return spike_accumulator
    
    def get_mesh_stats(self):
        """Get statistics for monitoring"""
        stats = {
            'mesh_spike_rates': [],
            'mesh_e_rates': [],
            'mesh_i_rates': [],
        }
        
        for i in range(self.num_meshes):
            if self.mesh_e_spikes[i] is not None:
                e_rate = self.mesh_e_spikes[i].float().mean().item()
                i_rate = self.mesh_i_spikes[i].float().mean().item()
                stats['mesh_e_rates'].append(e_rate)
                stats['mesh_i_rates'].append(i_rate)
                stats['mesh_spike_rates'].append((e_rate + i_rate) / 2)
            else:
                stats['mesh_e_rates'].append(0.0)
                stats['mesh_i_rates'].append(0.0)
                stats['mesh_spike_rates'].append(0.0)
        
        stats['avg_spike_rate'] = np.mean(stats['mesh_spike_rates'])
        stats['avg_e_rate'] = np.mean(stats['mesh_e_rates'])
        stats['avg_i_rate'] = np.mean(stats['mesh_i_rates'])
        stats['ei_balance'] = stats['avg_e_rate'] / (stats['avg_i_rate'] + 1e-8)
        
        return stats
    
    def get_quantized_weights(self):
        """Get all integer weights for hardware deployment"""
        weights = {
            'input_projections': [proj.get_integer_weights() for proj in self.input_projections],
            'inter_connections': [conn.get_integer_weights() for conn in self.inter_connections],
            'meshes': [mesh.get_quantized_weights() for mesh in self.meshes]
        }
        return weights


class TrainableNeuromorphicLSMNetwork(nn.Module):
    """
    Complete trainable neuromorphic LSM with dual-mode operation.
    
    Training: Float with surrogate gradients (trainable end-to-end)
    Deployment: Integer quantized weights (neuromorphic hardware)
    
    BDSM compliant: Additions, bit-shifts, sparse events! 🧠⚡
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Create trainable neuromorphic reservoir
        self.reservoir = TrainableNeuromorphicCircularReservoir(config)
        
        # Readout layer (can also be quantized)
        num_actions = config['num_actions']
        read_weight_bits = config['lsm'].get('readout_weight_bits', 8)
        read_weight_scale = config['lsm'].get('readout_weight_scale', 16.0)
        
        self.readout = QuantizedLinear(
            self.reservoir.total_neurons, 
            num_actions,
            weight_bits=read_weight_bits,
            weight_scale=read_weight_scale,
            bias=True
        )
        
        print(f"\n✅ Trainable Neuromorphic LSM Network")
        print(f"   Reservoir: {self.reservoir.total_neurons}N (TRAINABLE, neuromorphic-friendly)")
        print(f"   Readout: {self.reservoir.total_neurons} → {num_actions} (quantized, {read_weight_bits}-bit)")
        print(f"   🧠 BDSM: BRAIN DON'T SIMPLY MULTIPLY!")
        print(f"   🎓 Training: Surrogate gradients enable SGD/BPTT")
        print(f"   🚀 Deployment: Export integer weights for neuromorphic chips")
    
    def forward(self, fov):
        """Forward pass with gradient flow"""
        if len(fov.shape) == 4:
            batch_size = fov.shape[0]
            fov_flat = fov.view(batch_size, -1)
        else:
            fov_flat = fov
        
        # Normalize input
        fov_norm = fov_flat / (fov_flat.abs().max() + 1e-8)
        
        # Pass through trainable neuromorphic reservoir
        liquid_state = self.reservoir(fov_norm)
        
        # Readout (quantized, with gradient flow)
        action_logits = self.readout(liquid_state)
        
        return action_logits
    
    def get_liquid_state(self, fov):
        """Get reservoir state without readout"""
        if len(fov.shape) == 4:
            batch_size = fov.shape[0]
            fov_flat = fov.view(batch_size, -1)
        else:
            fov_flat = fov
        
        fov_norm = fov_flat / (fov_flat.abs().max() + 1e-8)
        return self.reservoir(fov_norm)
    
    def reset_state(self):
        """Reset reservoir state"""
        self.reservoir.reset_state()
    
    def get_stats(self):
        """Get reservoir statistics"""
        return self.reservoir.get_mesh_stats()
    
    def export_for_neuromorphic_hardware(self):
        """Export quantized integer weights for deployment"""
        weights = self.reservoir.get_quantized_weights()
        weights['readout'] = self.readout.get_integer_weights()
        
        print("\n📦 Exporting for neuromorphic hardware...")
        print(f"   ✓ All weights quantized to integers")
        print(f"   ✓ Ready for Loihi, TrueNorth, SpiNNaker, etc.")
        
        return weights
