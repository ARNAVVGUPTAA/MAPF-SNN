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
    def __init__(self, tau=2.5, v_reset=0.0, v_threshold=0.1, 
                 surrogate_function=surrogate.Sigmoid(alpha=4.0), detach_reset=True):
        super().__init__(surrogate_function=surrogate_function, detach_reset=detach_reset)
        #self.initialize_weights()
        # Use global config values or fallbacks
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
        # ⚡ Proper exponential decay with input scaling
        decay = math.exp(-1.0 / self.tau)
        self.mem = self.mem * (1 - 1/self.tau) + x / self.tau

    def neuronal_fire(self):
        # Use hard threshold for spiking
        return (self.mem >= self.v_threshold).float()

    def forward(self, x, log_spikes=False):
        x = x
        self.neuronal_charge(x)
        self.spike = self.neuronal_fire()
        self.v = self.mem  # ensure v is tensor before reset
        self.neuronal_reset(self.spike)
        return self.spike


# Adaptive LIF neuron with spike-triggered adaptation
class AdaptiveLIFNode(RecurrentLIFNode):
    """
    Extends RecurrentLIFNode by adding an adaptation current that raises threshold after each spike.
    """
    def __init__(self, tau=2.5, adaptation_tau=100.0, alpha=0.1, v_reset=0.0, v_threshold=0.1,
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
        self.neuronal_reset(self.spike)
        return self.spike


class SurrogateSNN(nn.Module):
    """
    Spiking neural network with surrogate gradients and recurrent LIF neurons.
    Designed for classification tasks with configurable time steps and neuron dynamics.
    """
    def __init__(self, input_shape, num_classes, config=None):
        super().__init__()
        C, H, W = input_shape
        cfg = config or {}

        # Split conv layers into blocks for feedback
        self.conv1 = nn.Sequential(
            layer.Conv2d(C, 64, kernel_size=7, stride=1, padding=2),
            RecurrentLIFNode(
                tau=cfg['lif_tau'] * cfg.get('lif_scale_conv1', 1.0)
            ),
             # Synapse filter with fixed time constant (non-learnable)
            SynapseFilter(tau=cfg['syn_tau'], learnable=False),
        )
        self.conv2 = nn.Sequential(
            layer.Conv2d(64, 128, kernel_size=5, padding=1),
            AdaptiveLIFNode(
                tau=cfg['lif_tau'] * cfg.get('lif_scale_conv2', 0.75),
                adaptation_tau=cfg['adapt_tau'],
                alpha=cfg['adapt_alpha']
            ),
        )
        self.conv3 = nn.Sequential(
            layer.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            RecurrentLIFNode(
                tau=cfg['lif_tau'] * cfg.get('lif_scale_conv3', 0.6)
            )
        )
        # ⚡ Recurrent feedback from conv2 to conv1 via 1x1 conv + LIF
        self.feedback = nn.Sequential(
            RecurrentLIFNode(tau=cfg.get('lif_tau', 2.0)*0.5),
            layer.Conv2d(128, 64, kernel_size=1)
        )

        # ⚡ Single-step attention mechanism
        self.attention = LinearRecurrentContainer(
            RecurrentLIFNode(
                tau=cfg['lif_tau'] * cfg.get('lif_scale_attention', 0.5)
            ),
             step_mode='s',  # single-step mode for 2D inputs
            in_features=256,
            out_features=256,
             bias=True
         )        
        # Additional conv4 block integrating post-attention features
        self.conv4 = nn.Sequential(
            layer.Conv2d(256, 128, kernel_size=5, padding=2),
            AdaptiveLIFNode(
                tau=cfg['lif_tau'] * cfg.get('lif_scale_conv4', 0.75),
                adaptation_tau=cfg['adapt_tau'],
                alpha=cfg['adapt_alpha']
            ),
        )

        # classification head now expects 128 features from conv4
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            layer.Linear(128, num_classes)
        )

        self.num_classes = num_classes  # Save num_classes for reshaping outputs

        # residual projection from conv2 to conv3
        self.res_conv = nn.Conv2d(128, 256, kernel_size=1)
        # transformer-based spatial self-attention
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=cfg['transformer_nhead'],
                dim_feedforward=cfg['transformer_ff_dim']
            ),
            num_layers=cfg['transformer_layers']
        )
        # Second transformer encoder for deeper context
        self.transformer2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=cfg['transformer_nhead'],
                dim_feedforward=cfg['transformer_ff_dim']
            ),
            num_layers=cfg['transformer2_layers']
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x, log_spikes=False):
        """
        Process single time step with shape [B, A, C, H, W]
        Maintains internal state between calls until reset
        """
        # ⚡ Handle agent dimension
        if x.dim() == 5:
            B, A, C, H, W = x.shape
            x = x.reshape(B*A, C, H, W)
        
        # Feature extraction with feedback
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        # compute spatial feedback from conv2 and add to x1
        fb = self.feedback(x2)  # [batch,64,H,W]
        x1 = x1 + fb
        x2 = self.conv2(x1)
        # conv3 with residual skip and self-attention
        skip = self.res_conv(x2)
        x3 = self.conv3(x2) + skip
        # apply transformer over spatial positions
        Bz, C, H3, W3 = x3.shape
        seq = x3.view(Bz, C, H3*W3).permute(2, 0, 1)
        attn = self.transformer(seq)
        attn = attn.permute(1, 2, 0).view(Bz, C, H3, W3)
        out = x3 + attn

        # Second attention block before conv4
        seq2 = out.view(Bz, C, H3*W3).permute(2, 0, 1)
        attn2 = self.transformer2(seq2)
        attn2 = attn2.permute(1, 2, 0).view(Bz, C, H3, W3)
        out = out + attn2

        # Attention with recurrence
        att = self.attention(
            F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        )
        att = att.view(out.shape[0], 256, 1, 1)
        out = out * att.sigmoid()
        # further process through conv4
        out = self.conv4(out)
        # Classification
        return self.classifier(out)
