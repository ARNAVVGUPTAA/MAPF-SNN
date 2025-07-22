import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.utils_weights import weights_init
import numpy as np
from copy import copy

# Optional: Only import spikingjelly if needed
from spikingjelly.activation_based import layer, neuron


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.S = None
        self.num_agents = self.config["num_agents"]
        self.map_shape = self.config["map_shape"]
        self.num_actions = 5
        self.net_type = config.get("net_type", "baseline")

        dim_encoder_mlp = self.config["encoder_layers"]
        self.compress_Features_dim = self.config["encoder_dims"]
        dim_action_mlp = self.config["action_layers"]
        action_features = [self.num_actions]

        # CNN
        self.conv_dim_W = []
        self.conv_dim_H = []
        self.conv_dim_W.append(self.map_shape[0])
        self.conv_dim_H.append(self.map_shape[1])
        channels = [2] + self.config["channels"]
        num_conv = len(channels) - 1
        strides = [1] * num_conv
        padding_size = [1] * num_conv
        filter_taps = [3] * num_conv

        conv_layers = []
        H_tmp = copy(self.map_shape[0])
        W_tmp = copy(self.map_shape[1])
        for l in range(num_conv):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=channels[l],
                    out_channels=channels[l + 1],
                    kernel_size=filter_taps[l],
                    stride=strides[l],
                    padding=padding_size[l],
                    bias=True,
                )
            )
            conv_layers.append(nn.BatchNorm2d(num_features=channels[l + 1]))
            conv_layers.append(nn.ReLU(inplace=True))
            W_tmp = int((W_tmp - filter_taps[l] + 2 * padding_size[l]) / strides[l]) + 1
            H_tmp = int((H_tmp - filter_taps[l] + 2 * padding_size[l]) / strides[l]) + 1
            self.conv_dim_W.append(W_tmp)
            self.conv_dim_H.append(H_tmp)
        self.convLayers = nn.Sequential(*conv_layers)

        # MLP Encoder
        self.compress_Features_dim = self.config["last_convs"] + self.compress_Features_dim
        mlp_encoder = []
        for l in range(dim_encoder_mlp):
            mlp_encoder.append(
                nn.Linear(
                    self.compress_Features_dim[l], self.compress_Features_dim[l + 1]
                )
            )
            mlp_encoder.append(nn.ReLU(inplace=True))
        self.compressMLP = nn.Sequential(*mlp_encoder)

        # MLP Action
        action_features = [self.compress_Features_dim[-1]] + action_features
        mlp_action = []
        for l in range(dim_action_mlp - 1):
            mlp_action.append(nn.Linear(action_features[l], action_features[l + 1]))
            mlp_action.append(nn.ReLU(inplace=True))

        # For SNN: use spiking output layer
        if self.net_type == "snn" and layer is not None and neuron is not None:
            self.actionMLP = nn.Sequential(*mlp_action)
            self.fc_out = layer.Linear(action_features[-2], action_features[-1], bias=True)
            self.out_neuron = neuron.LIFNode()
        else:
            mlp_action.append(nn.Linear(action_features[-2], action_features[-1]))
            self.actionMLP = nn.Sequential(*mlp_action)
            self.fc_out = None
            self.out_neuron = None

        self.apply(weights_init)

    def forward(self, states, return_spikes=False):
        batch_size = states.shape[0]
        action_logits = torch.zeros(batch_size, self.num_agents, self.num_actions).to(
            self.config["device"]
        )
        pre_spikes_list = []
        post_spikes_list = []
        for id_agent in range(self.num_agents):
            agent_state = states[:, id_agent, :, :, :]
            features = self.convLayers(agent_state)
            features_flatten = features.reshape(features.size(0), -1)
            encoded_feats = self.compressMLP(features_flatten)
            encoded_feats_flat = encoded_feats.reshape(encoded_feats.size(0), -1)
            x = self.actionMLP(encoded_feats_flat)
            if self.net_type == "snn" and self.fc_out is not None and self.out_neuron is not None:
                pre_spikes = x
                x = self.fc_out(x)
                post_spikes = self.out_neuron(x)
                action_logits[:, id_agent, :] = post_spikes
                pre_spikes_list.append(pre_spikes)
                post_spikes_list.append(post_spikes)
            else:
                action_logits[:, id_agent, :] = x

        if self.net_type == "snn" and return_spikes:
            return action_logits, {
                'pre': torch.stack(pre_spikes_list, dim=1),
                'post': torch.stack(post_spikes_list, dim=1)
            }
        #print("net_type:", self.net_type, "layer:", layer, "neuron:", neuron)
        return action_logits

