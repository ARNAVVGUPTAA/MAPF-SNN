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
from spikingjelly.activation_based.neuron import LIFNode
from config import config
from models.surrogate_snn import SurrogateSNN, RecurrentLIFNode
from collision_utils import (
    compute_collision_loss, 
    extract_positions_from_fov, 
    predict_future_collisions,
    reconstruct_positions_from_trajectories,
    load_initial_positions_from_dataset
)

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
    
    # Create model - framework_snn Network only takes config
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
            
            # Load initial positions for collision detection if enabled
            use_collision_loss = config.get('use_collision_loss', True)
            initial_positions_cache = {}
            
            for i, (states, trajectories, _) in enumerate(data_loader.train_loader):
                states = states.to(device)
                trajectories = trajectories.to(device)  # [batch, time, agents]

                T = states.shape[1]  # Number of time steps
                batch_size = states.shape[0]
                num_agents = trajectories.shape[2]

                # Load initial positions for this batch if collision loss is enabled
                if use_collision_loss:
                    try:
                        # Get case indices for this batch (simplified approach)
                        batch_start_idx = i * batch_size
                        case_indices = list(range(batch_start_idx, batch_start_idx + batch_size))
                        
                        # Load initial positions from dataset
                        initial_positions = load_initial_positions_from_dataset(
                            config['train']['root_dir'], case_indices, mode='train'
                        )
                        initial_positions = initial_positions.to(device)
                        
                        # Handle batch size mismatch if needed
                        if initial_positions.shape[0] > batch_size:
                            initial_positions = initial_positions[:batch_size]
                        elif initial_positions.shape[0] < batch_size:
                            # Pad with dummy positions if needed
                            dummy_positions = torch.randint(0, 28, (batch_size - initial_positions.shape[0], num_agents, 2), 
                                                           device=device, dtype=torch.float32)
                            initial_positions = torch.cat([initial_positions, dummy_positions], dim=0)
                            
                    except Exception as e:
                        print(f"Warning: Failed to load initial positions: {e}. Using dummy positions.")
                        initial_positions = torch.randint(0, 28, (batch_size, num_agents, 2), 
                                                         device=device, dtype=torch.float32)

                # Reset and forward through time
                functional.reset_net(model)
                spike_reg = 0.0  # initialize spike regularization accumulator
                collision_reg = 0.0  # initialize collision regularization accumulator
                total_loss = 0.0
                total_spikes = 0
                total_collisions = 0
                
                # Get collision loss configuration
                collision_config = {
                    'vertex_collision_weight': config.get('vertex_collision_weight', 1.0),
                    'edge_collision_weight': config.get('edge_collision_weight', 0.5),
                    'collision_loss_type': config.get('collision_loss_type', 'l2')
                }
                collision_loss_weight = config.get('collision_loss_weight', 2.0)
                
                prev_positions = None
                for t in range(T):
                    fov_t = states[:, t]
                    out_t, spike_info = model(fov_t, return_spikes=True)
                    out_t = out_t.view(-1, trajectories.shape[2], model.num_classes)

                    # compute standard cross-entropy loss for time step
                    loss = 0
                    for agent in range(trajectories.shape[2]):
                        logits = out_t[:, agent]
                        targets = trajectories[:, t, agent].long()
                        loss += F.cross_entropy(logits, targets)
                    loss /= trajectories.shape[2]

                    # Add collision loss if enabled
                    if use_collision_loss:
                        # Reconstruct current positions from trajectories
                        current_positions = reconstruct_positions_from_trajectories(
                            initial_positions, trajectories, current_time=t, 
                            board_size=config['board_size'][0]
                        )
                        
                        # Compute collision loss based on predicted actions
                        logits_flat = out_t.view(-1, model.num_classes)
                        collision_loss, collision_info = compute_collision_loss(
                            logits_flat, current_positions, prev_positions, collision_config
                        )
                        
                        # Add collision loss to total loss
                        loss += collision_loss_weight * collision_loss
                        collision_reg += float(collision_loss.item())
                        total_collisions += collision_info['total_collisions']
                        
                        # Update previous positions for next timestep
                        prev_positions = current_positions

                    spike_reg    += float(spike_info['output_spikes'].mean().item())
                    total_spikes += int(  spike_info['output_spikes'].sum().item())

                    total_loss += loss
                    functional.detach_net(model)
                    #out_t, spike_info = model(fov_t, return_spikes=True)
                    # print(
                    #     f"[Epoch {epoch} Step {t}] "
                    #     f"Input μ/σ: {spike_info['input_current'].mean():.3f}/"
                    #     f"{spike_info['input_current'].std():.3f} | "
                    #     f"Block spikes avg: {spike_info['snn_block_spikes'].mean():.3f} | "
                    #     f"Final spikes sum: {spike_info['output_spikes'].sum():.0f}"
                    # )

                # average over time
                total_loss /= T
                spike_reg /= T  # average regularization over time
                collision_reg /= T  # average collision regularization over time
                
                # include L1 spike regularization in loss
                spike_decay = max(0.03, 0.1 * (1 - epoch / 100))
                total_loss += spike_decay * spike_reg

                # Backpropagation and optimization
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                train_loss += total_loss.item()
            print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Spikes: {total_spikes} | SpikeReg: {spike_reg:.4f} | CollisionReg: {collision_reg:.4f} | Collisions: {total_collisions}")
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
