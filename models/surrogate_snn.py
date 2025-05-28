import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from spikingjelly.activation_based.layer import SynapseFilter, LinearRecurrentContainer
from config import config


class RecurrentLIFNode(neuron.BaseNode):
    """
    A recurrent leaky integrate-and-fire (LIF) neuron layer.
    """
    def __init__(self, tau=5.0, v_reset=0.0, v_threshold=0.3, 
                 surrogate_function=surrogate.Sigmoid(alpha=4.0), detach_reset=True):
        # Initialize BaseNode with surrogate gradient and reset behavior
        alpha = config.get('surrogate_alpha', 4.0)
        super().__init__(surrogate_function=surrogate.Sigmoid(alpha=alpha), detach_reset=detach_reset)
        self.tau = config.get('lif_tau', tau) if tau is None else tau
        self.v_reset = config.get('lif_v_reset', v_reset)
        self.v_threshold = config.get('lif_v_threshold', v_threshold)
        self.register_memory('mem', None)  # ⚡ Maintain state across time steps
        self.register_memory('spike', None)

    def neuronal_charge(self, x):
        # Always reset mem if shape does not match x
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        # ⚡ Proper exponential decay with input integration
        decay = math.exp(-1.0 / self.tau)
        self.mem = self.mem * decay + x * (1 - decay)

    def neuronal_fire(self):
        # Use hard threshold for spiking
        return (self.mem >= self.v_threshold).float()

    def forward(self, x, log_spikes=False):
        x = x
        self.neuronal_charge(x)
        self.spike = self.neuronal_fire()
        self.v = self.mem  # ensure v is tensor before reset
        
        # Debug: print membrane stats when log_spikes is True
        if log_spikes:
            print(f"  LIF mem min: {self.mem.min().item():.4f}, max: {self.mem.max().item():.4f}, mean: {self.mem.mean().item():.4f}")
            print(f"  LIF spikes: {self.spike.sum().item():.0f} / {self.spike.numel()}")
        
        self.neuronal_reset(self.spike)
        return self.spike


# Adaptive LIF neuron with spike-triggered adaptation
class AdaptiveLIFNode(RecurrentLIFNode):
    """
    Extends RecurrentLIFNode by adding an adaptation current that raises threshold after each spike.
    """
    def __init__(self, tau=5.0, adaptation_tau=200.0, alpha=0.01, v_reset=0.0, v_threshold=0.3,
                 surrogate_function=None, detach_reset=True):
        super().__init__(tau, v_reset, v_threshold,
                         surrogate_function or surrogate.Sigmoid(alpha=4.0), detach_reset)
        self.adaptation_tau = adaptation_tau
        self.alpha = alpha
        self.register_memory('b', None)  # adaptation current

    def forward(self, x, log_spikes=False):
        # initialize adaptation buffer
        if self.b is None or self.b.shape != x.shape:
            self.b = torch.zeros_like(x)
        # standard charge
        self.neuronal_charge(x)
        # decay adaptation
        self.b = self.b * (1 - 1.0 / self.adaptation_tau)
        # fire with dynamic threshold
        self.spike = (self.mem >= (self.v_threshold + self.b)).float()
        # increment adaptation on spike
        self.b = self.b + self.alpha * self.spike
        self.v = self.mem
        
        # Debug: print adaptive membrane stats when log_spikes is True
        if log_spikes:
            effective_thresh = self.v_threshold + self.b
            print(f"  Adaptive LIF mem min: {self.mem.min().item():.4f}, max: {self.mem.max().item():.4f}, mean: {self.mem.mean().item():.4f}")
            print(f"  Adaptive thresh min: {effective_thresh.min().item():.4f}, max: {effective_thresh.max().item():.4f}, mean: {effective_thresh.mean().item():.4f}")
            print(f"  Adaptive spikes: {self.spike.sum().item():.0f} / {self.spike.numel()}")
        
        self.neuronal_reset(self.spike)
        return self.spike


class SurrogateSNN(nn.Module):
    """
    Simplified Spiking Neural Network with fully connected layers and batch normalization.
    Focuses on core SNN functionality for better learning convergence.
    """
    def __init__(self, input_shape, num_classes, config=None):
        super().__init__()
        C, H, W = input_shape
        input_size = C * H * W  # Flatten spatial dimensions
        cfg = config or {}

        # Batch normalization parameters
        bn_momentum = float(cfg.get('batch_norm_momentum', 0.1))
        bn_eps = float(cfg.get('batch_norm_eps', 1e-5))
        self.use_batch_norm = True  # Always use batch norm as requested

        # Layer 1: Input to hidden layer 1 with batch norm
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            layer.Linear(input_size, 256),
            nn.BatchNorm1d(256, momentum=bn_momentum, eps=bn_eps),
            RecurrentLIFNode(
                tau=cfg.get('lif_tau', 10.0),
                v_threshold=cfg.get('lif_v_threshold', 0.5),
                v_reset=cfg.get('lif_v_reset', 0.0)
            )
        )
        
        # Layer 2: Hidden layer 1 to hidden layer 2 with batch norm
        self.fc2 = nn.Sequential(
            layer.Linear(256, 128),
            nn.BatchNorm1d(128, momentum=bn_momentum, eps=bn_eps),
            AdaptiveLIFNode(
                tau=cfg.get('lif_tau', 10.0),
                adaptation_tau=cfg.get('adapt_tau', 100.0),
                alpha=cfg.get('adapt_alpha', 0.02),
                v_threshold=cfg.get('lif_v_threshold', 0.5),
                v_reset=cfg.get('lif_v_reset', 0.0)
            )
        )
        
        # Layer 3: Hidden layer 2 to output layer with batch norm
        self.fc3 = nn.Sequential(
            layer.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=bn_momentum, eps=bn_eps),
            RecurrentLIFNode(
                tau=cfg.get('lif_tau', 10.0),
                v_threshold=cfg.get('lif_v_threshold', 0.5),
                v_reset=cfg.get('lif_v_reset', 0.0)
            )
        )

        # Classification head with batch normalization
        self.classifier = nn.Sequential(
            layer.Linear(64, num_classes),
            nn.BatchNorm1d(num_classes, momentum=bn_momentum, eps=bn_eps)
        )

        self.num_classes = num_classes
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layer.Linear):
                # Xavier initialization for better convergence
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # Standard batch norm initialization
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, log_spikes=False):
        # Debug: print input stats
        if log_spikes:
            print(f"Input min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
        
        # Input preprocessing - adjust based on whether batch norm is used
        if self.use_batch_norm:
            # Light preprocessing since batch norm will handle normalization
            x = (x - x.mean()) / (x.std() + 1e-8)  # Basic normalization
            input_scale = 1.5  # Conservative scaling with batch norm
        else:
            # More aggressive preprocessing without batch norm
            x = (x - x.mean()) / (x.std() + 1e-8)  # Normalize to mean=0, std=1
            input_scale = 3.0  # Higher scale to reach firing threshold
        
        x = x * input_scale
        
        if log_spikes:
            print(f"Preprocessed input min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
        
        # Handle agent dimension
        if x.dim() == 5:
            B, A, C, H, W = x.shape
            x = x.reshape(B*A, C, H, W)
        
        # Forward pass - adjust amplification based on batch norm usage
        if log_spikes:
            print("FC1 forward...")
        x1 = self.fc1(x)
        if log_spikes:
            print(f"FC1 spikes: {x1.sum().item():.0f} / {x1.numel()}")
        
        # Adjust amplification based on batch norm
        amplification = 1.2 if self.use_batch_norm else 2.0
        x1_amplified = x1 * amplification
        
        if log_spikes:
            print("FC2 forward...")
        x2 = self.fc2(x1_amplified)
        if log_spikes:
            print(f"FC2 spikes: {x2.sum().item():.0f} / {x2.numel()}")
        
        # Apply same amplification for consistency
        x2_amplified = x2 * amplification
        
        if log_spikes:
            print("FC3 forward...")
        x3 = self.fc3(x2_amplified)
        if log_spikes:
            print(f"FC3 spikes: {x3.sum().item():.0f} / {x3.numel()}")
        
        # Classification
        output = self.classifier(x3)
        
        if log_spikes:
            print(f"Output min: {output.min().item():.4f}, max: {output.max().item():.4f}")
        
        return output
