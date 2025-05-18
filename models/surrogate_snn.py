import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, layer, functional

class SurrogateSNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.num_classes = num_classes
        C, H, W = input_shape  # Input shape: (channels, height, width)

        # Convolutional + Recurrent Spiking Neurons
        self.conv1 = layer.Conv2d(C, 32, kernel_size=1, padding=1)
        self.sn1 = RecurrentLIFNode(
            tau=2.0,
            surrogate_function=surrogate.Sigmoid(),
            detach_reset=True
        )

        self.conv2 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
        self.sn2 = RecurrentLIFNode(
            tau=2.0,
            surrogate_function=surrogate.Sigmoid(),
            detach_reset=True
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):  # x shape: (T, B, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(0)  # Add time dimension

        T, B, C, H, W = x.shape

        # Initialize accumulator for time-summed features
        spk_sum = torch.zeros((B, self.fc.in_features), device=x.device)

        for t in range(T):
            out = self.conv1(x[t])
            out = self.sn1(out)

            out = self.conv2(out)
            out = self.sn2(out)

            out = self.pool(out).view(B, -1)  # Shape: (B, 64)
            spk_sum += out

        out = spk_sum / T
        return self.fc(out)

class RecurrentLIFNode(neuron.BaseNode):
    def __init__(self, tau=2.0, surrogate_function=None, detach_reset=True):
        super().__init__(surrogate_function=surrogate_function, detach_reset=detach_reset)
        self.tau = tau
        self.mem = None
        self.spike = None

    def reset(self):
        self.mem = None
        self.spike = None

    def forward(self, x):
        if self.mem is None:
            self.mem = torch.zeros_like(x)
        # recurrent: mem carries over from previous time step
        self.mem = self.mem * (1 - 1/self.tau) + x
        # spike using surrogate function
        self.spike = self.surrogate_function(self.mem - 1.0)  # threshold at 1.0
        # reset membrane potential where spike occurred
        if self.detach_reset:
            self.mem = self.mem * (1 - self.spike.detach())
        else:
            self.mem = self.mem * (1 - self.spike)
        return self.spike