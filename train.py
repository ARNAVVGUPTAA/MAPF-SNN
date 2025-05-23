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
import torch.nn.functional as F  # Add this import

from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
from data_loader import GNNDataLoader, SNNDataLoader

from spikingjelly.activation_based import learning, functional

from config import config
from models.surrogate_snn import SurrogateSNN, RecurrentLIFNode

def compute_success_rate(success_list):
    """
    Compute the average success rate from a list of per-episode successes.
    Args:
        success_list (list of float): successes per episode (0.0 to 1.0)
    Returns:
        float: average success rate, or 0.0 if list is empty
    """
    if not success_list:
        return 0.0
    return sum(success_list) / len(success_list)

net_type = config["net_type"]
exp_name = config["exp_name"]
tests_episodes = config["tests_episodes"]
# config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        # Retrieve and cast hyperparameters from config
        lr = float(config.get('learning_rate', 1e-3))
        wd = float(config.get('weight_decay', 1e-4))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        # LR scheduler: reduce LR when success rate plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=float(config.get('lr_decay_factor', 0.5)),
            patience=int(config.get('lr_patience', 5)),
            verbose=True
        )
        early_stop_counter = 0
        loss_fn = nn.CrossEntropyLoss()
    else:
        optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss okkk")

    losses = []
    success_rate_final = []
    flow_time_final = []

    # Removed debug prints and kept only essential logs
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch}")
        model.train()
        if net_type == "snn":
            train_loss = 0.0
            total_spikes = 0  # Initialize total spikes counter
            for i, (states, trajectories, _) in enumerate(data_loader.train_loader):
                states = states.to(device)
                trajectories = trajectories.to(device)  # [batch, agents]

                T = states.shape[1]  # Number of time steps

                # Reset and forward through time
                functional.reset_net(model)
                spike_reg = 0.0  # initialize spike regularization accumulator
                total_loss = 0.0
                total_spikes = 0
                for t in range(T):
                    fov_t = states[:, t]
                    out_t = model(fov_t)
                    out_t = out_t.view(-1, trajectories.shape[2], model.num_classes)

                    # compute loss for time step
                    loss = 0
                    for agent in range(trajectories.shape[2]):
                        logits = out_t[:, agent]
                        targets = trajectories[:, t, agent].long()
                        loss += F.cross_entropy(logits, targets)
                    loss /= trajectories.shape[2]

                    # accumulate spike regularization and count spikes
                    for m in model.modules():
                        if isinstance(m, RecurrentLIFNode) and m.spike is not None:
                            spike_reg += m.spike.mean().item()
                            total_spikes += m.spike.sum().item()

                    total_loss += loss
                    functional.detach_net(model)

                # average over time
                total_loss /= T
                spike_reg /= T  # average regularization over time
                # include L1 spike regularization in loss
          
                total_loss += 0.5 * spike_reg

                # Backpropagation and optimization
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                train_loss += total_loss.item()
            print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Spikes: {total_spikes} | SpikeReg: {spike_reg:.4f}")
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

        model.eval()
        # Run test episodes and collect metrics per episode
        total_agents_reached = 0
        total_agents_counted = 0
        episode_success_rates = []
        episode_flow_times = []
        for ep in range(tests_episodes):
            # reset SNN internal state for this episode
            if net_type == "snn":
                functional.reset_net(model)

            # first generate obstacle array, then pick goals avoiding those cells
            obstacles = create_obstacles(config["board_size"], config.get("obstacles", 0))
            goals = create_goals(config["board_size"], config["num_agents"], obstacles)
            env = GraphEnv(config, goal=goals, obstacles=obstacles)
            emb = env.getEmbedding()
            obs = env.reset()
            done = False
            step_count = 0
            max_eval_steps = config.get("max_time", 100)
            while not done and step_count < max_eval_steps:
                fov_t = torch.tensor(obs["fov"]).float().to(config["device"])
                gso_t = torch.tensor(obs["adj_matrix"]).float().unsqueeze(0).to(config["device"])
                # Ensure batch dimension for SNN input
                if net_type == "snn" and fov_t.dim() == 4:
                    fov_t = fov_t.unsqueeze(0)
                with torch.no_grad():
                    if net_type == "snn":
                        logits = model(fov_t)
                    else:
                        logits = model(fov_t, gso_t)
                actions = torch.argmax(logits, dim=1).cpu().numpy()
                obs, _, done, _ = env.step(actions, emb)
                step_count += 1
            # episode done or max steps reached, compute metrics
            sr, ft, tot_agents, num_reached = env.computeMetrics()
            episode_success_rates.append(sr)
            episode_flow_times.append(ft)
            total_agents_reached += num_reached
            total_agents_counted += tot_agents
        # aggregate across episodes
        episode_success_rate = (sum(episode_success_rates) / len(episode_success_rates)) if episode_success_rates else 0.0
        avg_flow_time = (sum(episode_flow_times) / len(episode_flow_times)) if episode_flow_times else 0.0
        success_rate_final.append(episode_success_rate)
        flow_time_final.append(avg_flow_time)
        print(f"Episode success rate: {episode_success_rate:.4f}")
        print(f"Average flow time: {avg_flow_time:.2f}")
        print(f"Total agents reached: {total_agents_reached}/{total_agents_counted}")
        # Step scheduler and early stop check
        scheduler.step(episode_success_rate)
        # Early stopping if success_rate below threshold for many epochs
        if episode_success_rate < config.get('early_stop_threshold', 0.1):
            early_stop_counter += 1
        else:
            early_stop_counter = 0
        if early_stop_counter >= config.get('early_stop_patience', 15):
            print(f"Early stopping: success rate < {config['early_stop_threshold']} for {config['early_stop_patience']} epochs.")
            break

    np.save(os.path.join(results_path, "success_rate.npy"), np.array(success_rate_final))
    np.save(os.path.join(results_path, "flow_time.npy"), np.array(flow_time_final))
    np.save(os.path.join(results_path, "loss.npy"), np.array(losses))
    torch.save(model.state_dict(), os.path.join(results_path, "model.pt"))
