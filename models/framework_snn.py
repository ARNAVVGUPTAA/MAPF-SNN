import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, functional, surrogate, layer

class RSNNBlock(nn.Module):

    def __init__(self, input_size, hidden_size, time_steps):
        super(RSNNBlock, self).__init__()
        self.time_steps = time_steps
        self.fc = nn.Linear(input_size, hidden_size)
        self.lif = neuron.LIFNode(
            surrogate_function=surrogate.Sigmoid(),
            detach_reset=True,
        ) 
    def forward(self, x):
        spk_seq = []
        for t in range(self.time_steps):
            out = self.fc(x)
            spk = self.lif(out)
            spk_seq.append(spk)
        return torch.stack(spk_seq, dim=0).mean(dim=0)  # Average over time steps

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        input_dim = config['input_dim']
        hidden_dim = config['hidden_dim']
        output_dim = config['num_agents']
        self.rsnn = RSNNBlock(input_dim, hidden_dim, time_steps=10)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.out_neuron = neuron.LIFNode()

    def forward(self, x, gso=None, return_spikes=False):
        x = self.rsnn(x)
        pre_spikes = self.fc_out(x)
        post_spikes = self.out_neuron(pre_spikes)
        batch_size, num_agents, _ = post_spikes.shape[0], post_spikes.shape[1], post_spikes.shape[2]
        output = post_spikes.view(batch_size, num_agents, -1)
        if return_spikes:
            return output, {'pre': pre_spikes, 'post': post_spikes}
        return output

    def reset(self):
        self.rsnn.lif.reset()