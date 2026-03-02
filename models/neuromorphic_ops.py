"""
Neuromorphic-Friendly Operations with Surrogate Gradients
==========================================================

Custom autograd functions for brain-like computation:
- Forward: Integer math, bit-shifts, additions (neuromorphic)
- Backward: Surrogate gradients (differentiable approximations)

Key principle: BRAIN DON'T SIMPLY MULTIPLY (BDSM)
- No standard floating-point multiplications in forward pass
- Use accumulation, bit-shifts, and integer operations
- Maintain gradient flow for training via surrogate gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np


class SpikeFunction(Function):
    """
    Surrogate gradient for Heaviside step function (spike generation).
    
    Forward: Binary spike (Heaviside)
    Backward: Smooth approximation (various choices)
    """
    
    @staticmethod
    def forward(ctx, v, threshold, surrogate_type='fast_sigmoid', beta=10.0):
        """
        Forward: Spike if v >= threshold
        
        Args:
            v: Membrane potential
            threshold: Spike threshold
            surrogate_type: Type of surrogate gradient
            beta: Steepness parameter
        """
        ctx.save_for_backward(v, torch.tensor(threshold))
        ctx.beta = beta
        ctx.surrogate_type = surrogate_type
        
        # Binary spike output
        spikes = (v >= threshold).float()
        return spikes
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: Surrogate gradient approximation
        
        Common choices:
        1. Rectangular: grad = 1 if |v - threshold| < width else 0
        2. Sigmoid: grad = beta * sigmoid(beta * (v - threshold)) * (1 - sigmoid(...))
        3. Fast Sigmoid: grad = 1 / (beta * |v - threshold| + 1)^2
        4. Triangular: grad = max(0, 1 - beta * |v - threshold|)
        """
        v, threshold = ctx.saved_tensors
        beta = ctx.beta
        surrogate_type = ctx.surrogate_type
        
        # Compute surrogate gradient
        if surrogate_type == 'fast_sigmoid':
            # Fast sigmoid: 1 / (1 + beta * |x|)^2
            temp = (v - threshold)
            surrogate_grad = 1.0 / (1.0 + torch.abs(beta * temp)).pow(2)
        
        elif surrogate_type == 'sigmoid':
            # Sigmoid derivative
            sig = torch.sigmoid(beta * (v - threshold))
            surrogate_grad = beta * sig * (1 - sig)
        
        elif surrogate_type == 'triangular':
            # Piecewise linear (triangle)
            temp = torch.abs(v - threshold)
            surrogate_grad = torch.clamp(1.0 - beta * temp, min=0.0)
        
        elif surrogate_type == 'rectangular':
            # Rectangular window
            surrogate_grad = ((torch.abs(v - threshold) < 1.0 / beta).float())
        
        else:
            raise ValueError(f"Unknown surrogate type: {surrogate_type}")
        
        # Apply chain rule
        grad_v = grad_output * surrogate_grad
        
        # Return gradients (v, threshold, surrogate_type, beta)
        return grad_v, None, None, None


class BitShiftDecay(Function):
    """
    Neuromorphic decay using bit-shift with gradient flow.
    
    Forward: v_new = v - (v >> shift)  (integer bit-shift)
    Backward: Approximate as v_new = v * (1 - 2^(-shift))  (for gradient)
    """
    
    @staticmethod
    def forward(ctx, v, shift):
        """
        Forward: Bit-shift decay
        
        v_new = v - (v >> shift)
        Equivalent to: v_new = v * (1 - 1/2^shift)
        """
        ctx.shift = shift
        ctx.decay_factor = 1.0 - (2.0 ** -shift)
        
        # For integer mode: actual bit-shift
        # For float mode: simulate bit-shift
        if v.dtype in [torch.int32, torch.int64]:
            v_new = v - (v >> shift)
        else:
            # Float simulation of bit-shift
            v_new = v - torch.floor(v / (2 ** shift))
        
        return v_new
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: Treat as multiplication by decay factor
        
        d(v_new)/dv = (1 - 2^(-shift))
        """
        decay_factor = ctx.decay_factor
        grad_v = grad_output * decay_factor
        
        # Return gradients (v, shift)
        return grad_v, None


class QuantizedLinear(nn.Module):
    """
    Quantization-Aware Linear Layer (Synaptic Weights).
    
    Training: Float weights with quantization noise
    Inference: Integer weights and accumulation
    
    Forward: Quantized to integers, accumulation only
    Backward: Straight-through estimator (STE)
    """
    
    def __init__(self, in_features, out_features, weight_bits=8, 
                 weight_scale=1.0, bias=False):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.weight_scale = weight_scale
        
        # Float weights (trainable)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Quantization params
        self.register_buffer('quant_min', torch.tensor(-(2 ** (weight_bits - 1))))
        self.register_buffer('quant_max', torch.tensor(2 ** (weight_bits - 1) - 1))
    
    def quantize_weight(self, w):
        """
        Quantize weights to integers.
        
        w_int = clamp(round(w * scale), -2^(b-1), 2^(b-1) - 1)
        """
        # Scale to integer range
        w_scaled = w * self.weight_scale
        
        # Round and clamp
        w_quant = torch.clamp(
            torch.round(w_scaled),
            self.quant_min,
            self.quant_max
        )
        
        return w_quant
    
    def forward(self, x):
        """
        Forward with quantized weights.
        
        Uses Straight-Through Estimator (STE) for gradients.
        """
        # Quantize weights
        w_quant = self.quantize_weight(self.weight)
        
        # Straight-through estimator: forward uses quantized, backward uses float
        w_ste = w_quant + (self.weight * self.weight_scale - w_quant).detach()
        
        # Linear operation (accumulation of products)
        # In hardware: MAC operations (multiply-accumulate)
        # But weights are integers, so more efficient
        out = F.linear(x, w_ste / self.weight_scale, self.bias)
        
        return out
    
    def get_integer_weights(self):
        """Get integer weights for deployment"""
        with torch.no_grad():
            return self.quantize_weight(self.weight).to(torch.int32)


class QuantizedSpikeLinear(nn.Module):
    """
    Spike-based linear layer with event-driven accumulation.
    
    Only accumulates when input spikes (reduces computation).
    Weights are quantized integers.
    """
    
    def __init__(self, in_features, out_features, weight_bits=8, weight_scale=1.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Use QuantizedLinear
        self.linear = QuantizedLinear(
            in_features, out_features,
            weight_bits=weight_bits,
            weight_scale=weight_scale,
            bias=False
        )
    
    def forward(self, spikes):
        """
        Forward with spike-driven accumulation.
        
        Args:
            spikes: [batch, in_features] binary (0 or 1)
        
        Returns:
            currents: [batch, out_features]
        """
        # Only accumulate weights for spiking inputs
        # In hardware: event-driven (skip non-spiking neurons)
        
        # Standard matmul (but weights are quantized)
        currents = self.linear(spikes)
        
        return currents
    
    def forward_sparse(self, spikes):
        """
        Sparse forward (only for non-zero spikes).
        More efficient when spike rate is low.
        """
        # Get quantized weights
        w_int = self.linear.get_integer_weights()
        
        # Find spiking neurons
        batch_size = spikes.shape[0]
        device = spikes.device
        
        output = torch.zeros(batch_size, self.out_features, device=device)
        
        # For each batch element
        for b in range(batch_size):
            spike_indices = torch.nonzero(spikes[b] > 0.5, as_tuple=True)[0]
            
            if len(spike_indices) > 0:
                # Only accumulate weights of spiking neurons
                output[b] = w_int[:, spike_indices].sum(dim=1).float() / self.linear.weight_scale
        
        return output


class NeuromorphicEILIF(nn.Module):
    """
    Neuromorphic EILIF neuron with integer operations and surrogate gradients.
    
    Forward: Integer math (bit-shifts, adds)
    Backward: Surrogate gradients for training
    """
    
    def __init__(self, num_neurons, E_ratio=0.8, v_threshold=1.0, v_reset=0.0,
                 tau_E=2.0, tau_I=1.5, dt=1.0, surrogate_type='fast_sigmoid',
                 surrogate_beta=10.0, quantization_bits=16):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.num_E = int(num_neurons * E_ratio)
        self.num_I = num_neurons - self.num_E
        
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_type = surrogate_type
        self.surrogate_beta = surrogate_beta
        
        # Convert time constants to bit-shift amounts
        # tau = dt / log(2) * shift  =>  shift = log2(e^(dt/tau))
        # Approximation: shift ≈ log2(tau / dt) for decay ≈ 1 - dt/tau
        self.shift_E = int(np.round(np.log2(tau_E / dt)))
        self.shift_I = int(np.round(np.log2(tau_I / dt)))
        
        # Ensure shifts are at least 1
        self.shift_E = max(1, self.shift_E)
        self.shift_I = max(1, self.shift_I)
        
        # Neuron type mask
        self.register_buffer('is_excitatory', torch.zeros(num_neurons, dtype=torch.bool))
        self.is_excitatory[:self.num_E] = True
        
        # State
        self.register_buffer('v_membrane', None)
        
        print(f"   NeuromorphicEILIF: {num_neurons} neurons ({self.num_E} E, {self.num_I} I)")
        print(f"      Decay shifts: E={self.shift_E} ({1.0 - 2**-self.shift_E:.3f}), " +
              f"I={self.shift_I} ({1.0 - 2**-self.shift_I:.3f})")
    
    def init_state(self, batch_size, device):
        """Initialize membrane potential"""
        self.v_membrane = torch.zeros(batch_size, self.num_neurons, device=device)
    
    def reset_state(self):
        """Reset membrane potential"""
        if self.v_membrane is not None:
            self.v_membrane.zero_()
    
    def forward(self, input_current):
        """
        Forward with neuromorphic operations.
        
        Args:
            input_current: [batch, num_neurons]
        Returns:
            spikes, e_spikes, i_spikes
        """
        if self.v_membrane is None:
            self.init_state(input_current.shape[0], input_current.device)
        
        # Decay using bit-shift (neuromorphic-friendly)
        # Apply different decay to E and I neurons
        v_E = self.v_membrane[:, :self.num_E]
        v_I = self.v_membrane[:, self.num_E:]
        
        # Use custom bit-shift decay with gradient flow
        v_E = BitShiftDecay.apply(v_E, self.shift_E)
        v_I = BitShiftDecay.apply(v_I, self.shift_I)
        
        # Recombine
        self.v_membrane = torch.cat([v_E, v_I], dim=1)
        
        # Integrate (addition only - brain-friendly!)
        self.v_membrane = self.v_membrane + input_current
        
        # Fire using surrogate gradient spike function
        spikes = SpikeFunction.apply(
            self.v_membrane,
            self.v_threshold,
            self.surrogate_type,
            self.surrogate_beta
        )
        
        # Reset (subtract, not multiply!)
        reset_value = spikes * (self.v_membrane - self.v_reset)
        self.v_membrane = self.v_membrane - reset_value
        
        # Split E and I spikes
        e_spikes = spikes[:, :self.num_E]
        i_spikes = spikes[:, self.num_E:]
        
        return spikes, e_spikes, i_spikes


# Spike function convenience wrapper
def spike_with_surrogate(v, threshold=1.0, surrogate_type='fast_sigmoid', beta=10.0):
    """Convenience function for spike generation with surrogate gradient"""
    return SpikeFunction.apply(v, threshold, surrogate_type, beta)


def bit_shift_decay(v, shift):
    """Convenience function for bit-shift decay with gradient"""
    return BitShiftDecay.apply(v, shift)


# Export list
__all__ = [
    'SpikeFunction',
    'BitShiftDecay',
    'QuantizedLinear',
    'QuantizedSpikeLinear',
    'NeuromorphicEILIF',
    'spike_with_surrogate',
    'bit_shift_decay'
]
