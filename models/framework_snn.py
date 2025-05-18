import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer

# Define AdaptiveLIFNode to handle dynamic batch sizes between training and validation
class AdaptiveLIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.v_reset is the scalar initial voltage (e.g., 0.0) stored by LIFNode's __init__.
        # self.v is also initialized to this scalar value by LIFNode's __init__.

    def reset(self):
        # Call BaseNode.reset() to reset time step counter (self.t = 0)
        # and other base properties. BaseNode.reset() also attempts to set
        # self.v = self.v_init (the initial scalar voltage) if self.v_init was set
        # (which happens on the first forward pass if self.v was scalar).
        neuron.BaseNode.reset(self)
        
        # Crucially, ensure self.v is reset to its original scalar v_reset value.
        # This makes the neuron forget its previous batch-specific tensor shape.
        # On the next forward(x) call, if self.v is scalar, BaseNode.forward()
        # will re-initialize self.v = torch.full_like(x.data, self.v_reset), # Note: SpikingJelly uses self.v_init here
        # thus adapting to the new input x's shape.
        self.v = self.v_reset


class RSNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, tau, time_steps, v_threshold=1.0, detach_reset=True):
        super(RSNNBlock, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.lif = AdaptiveLIFNode(tau=tau, v_threshold=v_threshold, detach_reset=detach_reset)
        self.time_steps = time_steps

    def forward(self, x):
        # x is (batch_size * num_agents, features)
        # We simulate SNN dynamics over self.time_steps for this static input
        spk_seq = []
        # The input x to this block's forward is already the current for the first layer of this block.
        # If RSNNBlock is meant to be a multi-layer block, the structure would be different.
        # Assuming fc is the input layer to the LIF neurons for this block.
        for _ in range(self.time_steps): 
            current_after_fc = self.fc(x) # This implies x is static over time_steps or fc is applied each time.
                                          # If x is features that are constant over the SNN sim window,
                                          # then fc(x) is also constant. This is typical for rate coding from static features.
            spk = self.lif(current_after_fc) 
            spk_seq.append(spk)
        
        return torch.stack(spk_seq, dim=0).mean(dim=0)

    def reset(self):
        self.lif.reset()


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.config = config
        
        input_dim = config["input_dim"] 
        hidden_dim = config["hidden_dim"]
        action_dim = config["num_actions"]

        # SNN specific parameters from config, with defaults
        snn_time_steps = config.get("snn_time_steps", 10) # Time steps for SNN block simulation
        tau = config.get("tau", 2.0)
        v_threshold = config.get("v_threshold", 1.0)
        detach_reset = config.get("detach_reset", True)

        tau_output = config.get("tau_output", tau) # Default to main tau if not specified
        v_threshold_output = config.get("v_threshold_output", v_threshold)
        detach_reset_output = config.get("detach_reset_output", detach_reset)

        # This initial linear layer processes the raw input features (e.g., FOV)
        # Its output will be the input to the RSNNBlock's internal fc layer if that's the design.
        # Or, if RSNNBlock's fc is the first processing step, then fc_snn here might be redundant
        # or intended for a different purpose (e.g. pre-processing before temporal dynamics).
        # Based on typical usage, an initial fc projects input to hidden_dim, then SNN block operates.
        self.fc_snn_input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.snn_block = RSNNBlock(
            input_size=hidden_dim,  # RSNNBlock's internal fc takes hidden_dim
            hidden_size=hidden_dim, 
            tau=tau,
            time_steps=snn_time_steps,
            v_threshold=v_threshold, 
            detach_reset=detach_reset
        )
        
        self.fc_out = nn.Linear(hidden_dim, action_dim) 
        
        self.out_neuron = AdaptiveLIFNode(
            tau=tau_output, 
            v_threshold=v_threshold_output, 
            detach_reset=detach_reset_output
        )

    def forward(self, x, gso=None, return_spikes=False):
        # x shape: (batch_size * num_agents, input_dim)
        
        # Project input features to hidden dimension
        projected_x = self.fc_snn_input_projection(x) # (batch_size * num_agents, hidden_dim)
        
        # Process projected input through SNN block (temporal dynamics simulation)
        snn_block_output = self.snn_block(projected_x)   # (batch_size * num_agents, hidden_dim) - rate coded spikes
        
        pre_synaptic_potential_out = self.fc_out(snn_block_output) # (batch_size * num_agents, action_dim)
        post_synaptic_spikes_out = self.out_neuron(pre_synaptic_potential_out) # (batch_size * num_agents, action_dim)
        
        output = post_synaptic_spikes_out 

        if output.dim() != 2 or output.shape[-1] != self.config["num_actions"]:
            raise ValueError(f"Unexpected output spikes shape: {output.shape}. Expected 2D tensor with last dim {self.config['num_actions']}")

        if return_spikes:
            # Return the actual spikes required by STDPLearner
            # 'in_spike_for_fc_out': input spikes to the plastic layer (fc_out)
            # 'out_spike_from_out_neuron': output spikes from the post-synaptic neuron (out_neuron)
            return output, {'in_spike_for_fc_out': snn_block_output, 
                           'out_spike_from_out_neuron': post_synaptic_spikes_out}
        return output

    def reset(self):
        # Reset all stateful components
        if hasattr(self.fc_snn_input_projection, 'reset') and callable(self.fc_snn_input_projection.reset):
            self.fc_snn_input_projection.reset() # If it were a stateful layer like another SNN block
        self.snn_block.reset()
        self.out_neuron.reset()