import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
from data_loader import GNNDataLoader, SNNDataLoader

from spikingjelly.activation_based import learning, functional

import argparse

from models.surrogate_snn import SurrogateSNN

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="configs/config_snn.yaml")
args = parser.parse_args()

with open(args.config, "r") as config_path:
    config = yaml.safe_load(config_path)

net_type = config["net_type"]
exp_name = config["exp_name"]
tests_episodes = config["tests_episodes"]
config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if net_type == "baseline":
    from models.framework_baseline import Network
elif net_type == "gnn":
    from models.framework_gnn_message import Network
elif net_type == "snn":
    from models.framework_snn import Network

results_path = os.path.join("results", exp_name)
os.makedirs(results_path, exist_ok=True)

with open(os.path.join(results_path, "config.yaml"), "w") as config_path:
    yaml.dump(config, config_path)

if __name__ == "__main__":
    print("----- Training stats -----")
    if net_type == "gnn":
        data_loader = GNNDataLoader(config)
    elif net_type == "snn":
        # Initialize SNN data loader and infer channel dimensions from data
        data_loader = SNNDataLoader(config)
        # Sample first batch to get C, H, W
        sample_states, _, _ = next(iter(data_loader.train_loader))
        shape = sample_states.shape
        if len(shape) == 6:
            _, _, _, C, H, W = shape
        elif len(shape) == 5:
            _, C, H, W = shape
        else:
            raise RuntimeError(f"Unexpected SNN data shape: {shape}")
        config["channels"] = C
        config["height"] = H
        config["width"] = W
    else:
        data_loader = None  # baseline case or others
    model = Network(config)
    model.to(config["device"])

    if net_type == "snn":
        # Use SurrogateSNN and Adam optimizer as per prompt
        channels = config.get("channels", 1)
        height = config.get("height", config["board_size"])
        width = config.get("width", config["board_size"])
        num_actions = config.get("num_actions", 5)
        device = config["device"]
        epochs = config["epochs"]
        model = SurrogateSNN(input_shape=(channels, height, width), num_classes=num_actions).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
    else:
        optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

    losses = []
    success_rate_final = []
    flow_time_final = []

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch}")
        model.train()
        if net_type == "snn":
            train_loss = 0.0
            for states, trajectories, _ in data_loader.train_loader:
                # states: [batch, T, agents, C, H, W]
                batch, T, agents, C, H, W = states.shape
                states = states.to(device)
                trajectories = trajectories.to(device)  # [batch, agents]

                # Reset and forward through time
                functional.reset_net(model)
                # Compute average loss over time steps and agents instead of summing
                loss_batch = torch.zeros(1, device=device)
                for t in range(T):
                    fov_t = states[:, t]  # [batch, agents, C, H, W]
                    # Treat each agent as separate sample within batch
                    x_t = fov_t.reshape(batch * agents, C, H, W)
                    out_t = model(x_t)  # [batch*agents, num_actions]
                    out_t = out_t.reshape(batch, agents, -1)
                    # sum loss over agents and normalize by agent count
                    loss_t = 0
                    for a in range(agents):
                        loss_t += loss_fn(out_t[:, a, :], trajectories[:, a])
                    loss_batch += loss_t / agents
                # average over time steps
                loss_batch = loss_batch / T

                # Backprop and step
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
                train_loss += loss_batch.item()
                print(f"Epoch {epoch}, Loss: {loss_batch.item():.4f}")
            losses.append(train_loss)
        else:
            train_loss = 0.0
            for i, (states, trajectories, gso) in enumerate(data_loader.train_loader):
                states = states.to(config["device"])
                gso = gso.to(config["device"])

                # Remove SNN+STDP logic, keep only standard supervised learning for non-SNN
                optimizer.zero_grad()
                trajectories = trajectories.to(config["device"])
                output = model(states, gso)
                total_loss = torch.zeros(1, requires_grad=True).to(config["device"])
                for agent in range(trajectories.shape[1]):
                    loss = criterion(output[:, agent, :], trajectories[:, agent].long())
                    total_loss = total_loss + (loss / trajectories.shape[1])
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()
            print(f"Loss: {train_loss}")
            losses.append(train_loss)

        model.eval()
        success_rate = []
        flow_time = []
        for episode in range(tests_episodes):
            goals = create_goals(config["board_size"], config["num_agents"])
            obstacles = create_obstacles(config["board_size"], config["obstacles"])
            env = GraphEnv(config, goal=goals, obstacles=obstacles)
            emb = env.getEmbedding()
            obs = env.reset()
            if net_type == "snn":
                functional.reset_net(model)

            for step_idx in range(config["max_steps"]):
                current_fov_tensor = torch.tensor(obs["fov"]).float().to(config["device"])
                # For SNN, keep spatial shape: [agents, C, H, W]
                fov_for_model = current_fov_tensor  # shape [agents, C, H, W]
                gso_for_model = torch.tensor(obs["adj_matrix"]).float().unsqueeze(0).to(config["device"])

                with torch.no_grad():
                    if net_type == "snn":
                        functional.reset_net(model)
                        action_logits = model(fov_for_model)
                    else:
                        action_logits = model(fov_for_model, gso_for_model)

                action_np = action_logits.cpu().numpy()
                action_indices = np.argmax(action_np, axis=1)
                obs, reward, done, info = env.step(action_indices, emb)
                if done:
                    break

            metrics = env.computeMetrics()
            success_rate.append(metrics[0])
            flow_time.append(metrics[1])

        success_rate_mean = np.mean(success_rate)
        flow_time_mean = np.mean(flow_time)
        success_rate_final.append(success_rate_mean)
        flow_time_final.append(flow_time_mean)
        print(f"Success rate: {success_rate_mean}")
        print(f"Flow time: {flow_time_mean}")



    np.save(os.path.join(results_path, "success_rate.npy"), np.array(success_rate_final))
    np.save(os.path.join(results_path, "flow_time.npy"), np.array(flow_time_final))
    np.save(os.path.join(results_path, "loss.npy"), np.array(losses))
    torch.save(model.state_dict(), os.path.join(results_path, "model.pt"))
