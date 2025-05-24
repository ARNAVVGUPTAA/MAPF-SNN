import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, base
from config import config


class AdaptiveLIFNode(neuron.LIFNode):
    """
    LIF neuron that adapts to different batch sizes between train/validation.
    Key improvement: Better parameter initialization for consistent spiking.
    """
    def __init__(self, tau=None, v_threshold=None, v_reset=None, surrogate_alpha=None, detach_reset=None, *args, **kwargs):
        # Use config parameters for better spiking
        super().__init__(
            tau=tau if tau is not None else config.get('lif_tau', 10.0), 
            v_threshold=v_threshold if v_threshold is not None else config.get('lif_v_threshold', 0.5), 
            v_reset=v_reset if v_reset is not None else config.get('lif_v_reset', 0.0),
            detach_reset=detach_reset if detach_reset is not None else config.get('detach_reset', True),
            *args, **kwargs
        )

    def reset(self):
        """Reset to scalar voltage to adapt to new batch sizes"""
        neuron.BaseNode.reset(self)
        self.v = self.v_reset  # Reset to scalar for shape adaptation


class AttentionGate(nn.Module):
    """
    Simple attention mechanism to focus on important spatial/feature locations.
    This helps the SNN pay attention to relevant parts of the input.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention_fc = nn.Linear(input_dim, input_dim)
        self.gate_fc = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # Compute attention weights
        attention_weights = torch.sigmoid(self.attention_fc(x))
        # Compute gating values  
        gate_values = torch.tanh(self.gate_fc(x))
        # Apply attention: element-wise multiplication
        return x + (attention_weights * gate_values * 0.5)  # Residual + attended features


class RSNNBlock(base.MemoryModule):
    """
    Recurrent SNN block with improved temporal processing and attention.
    Key improvements:
    - Multiple time steps for temporal context
    - Attention mechanism for feature focus
    - Better parameter scaling
    """
    def __init__(self, input_size, hidden_size, cfg=None):
        super().__init__()
        
        # Get SNN parameters from global config
        self.time_steps = config.get('snn_time_steps', 3)
        self.input_scale = config.get('input_scale', 2.0)
        
        # Input processing with attention
        self.attention = AttentionGate(input_size)
        self.input_fc = nn.Linear(input_size, hidden_size)
        
        # Main spiking neuron with config parameters
        self.lif = AdaptiveLIFNode(
            tau=config.get('lif_tau', 10.0),
            v_threshold=config.get('lif_v_threshold', 0.5),
            v_reset=config.get('lif_v_reset', 0.0)
        )
        
        # Recurrent connection for temporal context
        self.recurrent_fc = nn.Linear(hidden_size, hidden_size)
        
        # Output processing
        self.output_fc = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        """
        Process input through multiple time steps with attention and recurrence.
        """
        self.lif.reset()  # Reset neuron state
        
        # Apply attention to input features
        attended_x = self.attention(x)
        
        # Scale input to encourage spiking - use config value
        input_current = self.input_fc(attended_x) * self.input_scale
        
        spike_accumulator = 0
        hidden_state = torch.zeros_like(input_current)
        
        # Process through multiple time steps for temporal dynamics
        for t in range(self.time_steps):
            # Combine input and recurrent currents
            if t == 0:
                total_current = input_current
            else:
                recurrent_current = self.recurrent_fc(hidden_state) * 0.3  # Gentle recurrence
                total_current = input_current * 0.7 + recurrent_current  # Weighted combination
            
            # Generate spikes
            spikes = self.lif(total_current)
            spike_accumulator += spikes
            hidden_state = spikes  # Update hidden state
        
        # Average spikes over time steps for rate coding
        avg_spikes = spike_accumulator / self.time_steps
        
        # Final processing with attention on spike outputs
        output = self.output_fc(avg_spikes)
        return output

    def reset(self):
        """Reset all stateful components"""
        self.lif.reset()
        super().reset()


class Network(base.MemoryModule):
    """
    Improved SNN for MAPF with better attention and temporal processing.
    
    Key improvements:
    1. Multi-step temporal processing
    2. Attention mechanisms for focus
    3. Better parameter tuning for consistent spiking
    4. Cleaner, more readable architecture
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config["num_actions"]
        
        # Network dimensions
        input_dim = config["input_dim"] 
        hidden_dim = config["hidden_dim"]
        action_dim = config["num_actions"]
        
        # SNN parameters from config
        self.time_steps = config.get("snn_time_steps", 3)
        self.input_scale = config.get("input_scale", 2.0)
        
        # Input preprocessing with normalization
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Main SNN processing block
        self.snn_block = RSNNBlock(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            cfg=config  # Pass entire config
        )
        
        # Output processing with attention
        self.output_attention = AttentionGate(hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, action_dim)
        
        # Final output neuron with config parameters
        self.output_neuron = AdaptiveLIFNode(
            tau=config.get("lif_tau", 10.0) * 0.5,  # Faster dynamics for output
            v_threshold=config.get("lif_v_threshold", 0.5) * 0.3,  # Lower threshold for output spikes
            v_reset=config.get("lif_v_reset", 0.0)
        )

    def forward(self, x, gso=None, return_spikes=False):
        """
        Forward pass through the improved SNN.
        
        Args:
            x: Input tensor [batch, agents, channels, height, width] or flattened
            gso: Graph structure (not used in this SNN)
            return_spikes: Whether to return spike statistics
            
        Returns:
            Action probabilities for each agent
        """
        # Reset all stateful components
        self.reset()
        
        # Handle input reshaping
        if x.dim() == 5:
            batch, agents, c, h, w = x.size()
            x = x.reshape(batch * agents, c * h * w)
        
        # Input preprocessing and normalization
        x_norm = self.input_norm(x)  # Normalize for stable training
        projected = self.input_projection(x_norm)
        
        # Scale input to encourage spiking - use config value
        projected = projected * self.input_scale
        
        # Process through main SNN block (with attention and temporal context)
        snn_output = self.snn_block(projected)
        
        # Apply output attention for action selection focus
        attended_output = self.output_attention(snn_output)
        
        # Generate action logits
        action_logits = self.output_fc(attended_output)
        
        # Final spiking output (rate-coded action values)
        output_spikes = self.output_neuron(action_logits * 3.0)  # Aggressive scaling for output spikes
        
        # Ensure output shape is correct
        if output_spikes.dim() != 2 or output_spikes.shape[-1] != self.num_classes:
            raise ValueError(f"Unexpected output shape: {output_spikes.shape}")
        
        if return_spikes:
            # Return spike statistics for STDP learning
            return output_spikes, {
                'snn_block_spikes': snn_output,
                'output_spikes': output_spikes,
                'input_current': projected
            }
        
        return output_spikes

    def reset(self):
        """Reset all memory modules for clean state between episodes"""
        super().reset()
        
    def get_attention_weights(self, x):
        """
        Debug method to visualize attention weights.
        Useful for understanding what the network focuses on.
        """
        if x.dim() == 5:
            batch, agents, c, h, w = x.size()
            x = x.reshape(batch * agents, c * h * w)
        
        x_norm = self.input_norm(x)
        projected = self.input_projection(x_norm)
        
        # Get attention weights from input attention
        input_attention = torch.sigmoid(self.snn_block.attention.attention_fc(projected))
        
        return {
            'input_attention': input_attention,
            'projected_features': projected
        }
