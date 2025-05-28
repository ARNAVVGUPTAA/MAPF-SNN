import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import yaml
import numpy as np
import math
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
from expert_utils import (
    load_expert_demonstrations,
    create_mixed_batch,
    compute_expert_loss_weight
)

def cosine_annealing_schedule(epoch, initial_value, T_max, eta_min_ratio=0.01, restart_period=0):
    """
    Cosine annealing schedule for regularizers.
    
    Args:
        epoch: Current epoch
        initial_value: Initial value of the regularizer
        T_max: Period of the cosine annealing
        eta_min_ratio: Minimum value ratio (min_value = eta_min_ratio * initial_value)
        restart_period: Period for cosine restart (0 = no restart)
    
    Returns:
        Scheduled value for the regularizer
    """
    # Ensure all parameters are floats to avoid type errors
    initial_value = float(initial_value)
    eta_min_ratio = float(eta_min_ratio)
    T_max = float(T_max)
    restart_period = float(restart_period)
    
    eta_min = eta_min_ratio * initial_value
    
    if restart_period > 0:
        # Cosine annealing with warm restarts
        epoch = epoch % restart_period
        T_max = restart_period
    
    # Cosine annealing formula
    value = eta_min + (initial_value - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
    return value

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

def evaluate_on_validation(model, data_loader, config, num_samples=10):
    """
    Evaluate model performance on validation dataset.
    Args:
        model: The neural network model
        data_loader: Data loader with valid_loader attribute
        config: Configuration dictionary
        num_samples: Number of validation samples to evaluate
    Returns:
        dict: Validation metrics including success rate and agents reached
    """
    if data_loader.valid_loader is None:
        return {"val_success_rate": 0.0, "val_agents_reached": 0, "val_total_agents": 0}
    
    model.eval()
    device = config["device"]
    net_type = config["net_type"]
    
    # Reset SNN state if using SNN
    if net_type == "snn":
        functional.reset_net(model)
    
    val_success_rates = []
    val_agents_reached = 0
    val_total_agents = 0
    
    # Sample a subset of validation data
    val_iter = iter(data_loader.valid_loader)
    samples_processed = 0
    
    with torch.no_grad():
        for _ in range(min(num_samples, len(data_loader.valid_loader))):
            try:
                batch_data = next(val_iter)
                if len(batch_data) == 4:  # SNN case with case_idx
                    states, trajectories, gso_or_dummy, case_idx = batch_data
                else:  # GNN case without case_idx
                    states, trajectories, gso_or_dummy = batch_data
                
                states = states.to(device)
                trajectories = trajectories.to(device)
                
                batch_size = states.shape[0]
                T = states.shape[1]
                num_agents = trajectories.shape[2]
                
                # Reset SNN for each validation batch
                if net_type == "snn":
                    functional.reset_net(model)
                
                # Process each time step
                correct_predictions = 0
                total_predictions = 0
                
                for t in range(T):
                    fov_t = states[:, t]
                    
                    if net_type == "snn":
                        logits = model(fov_t)
                    else:
                        gso_t = gso_or_dummy.to(device)
                        logits = model(fov_t, gso_t)
                    
                    logits = logits.view(-1, num_agents, model.num_classes)
                    
                    # Compare predictions with ground truth
                    for agent in range(num_agents):
                        predicted_actions = torch.argmax(logits[:, agent], dim=1)
                        true_actions = trajectories[:, t, agent].long()
                        correct_predictions += (predicted_actions == true_actions).sum().item()
                        total_predictions += batch_size
                    
                    if net_type == "snn":
                        functional.detach_net(model)
                
                # Calculate success rate for this batch (percentage of correct predictions)
                batch_success_rate = correct_predictions / total_predictions if total_predictions > 0 else 0.0
                val_success_rates.append(batch_success_rate)
                
                # Count agents (simplified metric)
                val_agents_reached += int(batch_success_rate * batch_size * num_agents)
                val_total_agents += batch_size * num_agents
                
                samples_processed += 1
                
            except StopIteration:
                break
    
    # Calculate average validation metrics
    avg_val_success_rate = sum(val_success_rates) / len(val_success_rates) if val_success_rates else 0.0
    
    return {
        "val_success_rate": avg_val_success_rate,
        "val_agents_reached": val_agents_reached,
        "val_total_agents": val_total_agents,
        "val_samples_processed": samples_processed
    }

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
        batch_data = next(iter(data_loader.train_loader))
        if len(batch_data) == 4:  # With case_idx
            sample_states, _, _, _ = batch_data
        else:  # Without case_idx (backward compatibility)
            sample_states, _, _ = batch_data
        
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

    # Load expert demonstrations for teacher forcing (first 5 epochs)
    expert_demos = {}
    use_expert_training = config.get('use_expert_training', True)
    max_expert_epochs = config.get('max_expert_epochs', 5)
    expert_ratio = config.get('expert_ratio', 0.25)
    
    if use_expert_training and net_type == "snn":
        print("Loading expert demonstrations for teacher forcing...")
        # Load expert demonstrations for available training cases
        all_case_indices = list(range(1000))  # Assuming up to 1000 training cases
        expert_demos = load_expert_demonstrations(
            dataset_root=config['train']['root_dir'],
            case_indices=all_case_indices,
            mode='train',
            num_agents=config['train']['nb_agents'],
            max_time=config['train']['max_time_dl']
        )
        print(f"Loaded {len(expert_demos)} expert demonstrations")

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
            
            for i, (states, trajectories, _, case_indices) in enumerate(data_loader.train_loader):
                states = states.to(device)
                trajectories = trajectories.to(device)  # [batch, time, agents]
                
                # Apply expert training for first few epochs
                expert_mask = None
                expert_loss_weight = 0.0
                if use_expert_training and epoch < max_expert_epochs and len(expert_demos) > 0:
                    # Create mixed batch with expert trajectories
                    batch_case_indices = case_indices.tolist() if hasattr(case_indices, 'tolist') else case_indices
                    # Filter out invalid case indices (negative values)
                    valid_case_indices = [idx if idx >= 0 else 0 for idx in batch_case_indices]
                    states, trajectories, _, expert_mask = create_mixed_batch(
                        (states, trajectories, _),
                        expert_demos,
                        valid_case_indices,
                        expert_ratio
                    )
                    trajectories = trajectories.to(device)
                    expert_mask = expert_mask.to(device)
                    expert_loss_weight = compute_expert_loss_weight(epoch, max_expert_epochs)

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
                real_collision_reg = 0.0  # initialize real collision regularization accumulator
                future_collision_reg = 0.0  # initialize future collision regularization accumulator
                total_loss = 0.0
                total_spikes = 0
                total_real_collisions = 0
                total_future_collisions = 0
                
                # Get collision loss configuration with cosine annealing
                use_cosine = config.get('use_cosine_annealing', True)
                T_max = config.get('cosine_T_max', 50)
                eta_min_ratio = config.get('cosine_eta_min_ratio', 0.01)
                restart_period = config.get('cosine_restart_period', 0)
                
                # Apply cosine annealing to collision loss weight
                if use_cosine:
                    collision_loss_weight = cosine_annealing_schedule(
                        epoch, 
                        config.get('collision_loss_weight', 2.0),
                        T_max, eta_min_ratio, restart_period
                    )
                    future_collision_weight = cosine_annealing_schedule(
                        epoch,
                        config.get('future_collision_weight', 0.8),
                        T_max, eta_min_ratio, restart_period
                    )
                    spike_reg_weight = cosine_annealing_schedule(
                        epoch,
                        config.get('spike_reg_weight', 5e-3),
                        T_max, eta_min_ratio, restart_period
                    )
                else:
                    collision_loss_weight = config.get('collision_loss_weight', 2.0)
                    future_collision_weight = config.get('future_collision_weight', 0.8)
                    # Legacy linear decay for spike regularization
                    spike_reg_weight = max(
                        float(config.get('max_spike_decay', 1e-3)), 
                        config.get('spike_decay_coeff', 0.05) * (1 - epoch / config.get('epochs', 50))
                    )
                
                collision_config = {
                    'vertex_collision_weight': config.get('vertex_collision_weight', 1.0),
                    'edge_collision_weight': config.get('edge_collision_weight', 0.5),
                    'collision_loss_type': config.get('collision_loss_type', 'l2'),
                    'use_future_collision_penalty': config.get('use_future_collision_penalty', False),
                    'future_collision_steps': config.get('future_collision_steps', 2),
                    'future_collision_weight': future_collision_weight,  # Use scheduled value
                    'future_step_decay': config.get('future_step_decay', 0.7),
                    'separate_collision_types': config.get('separate_collision_types', True)
                }
                
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
                        agent_loss = F.cross_entropy(logits, targets, reduction='none')
                        
                        # Apply expert loss weighting if using expert training
                        if expert_mask is not None and expert_loss_weight > 0:
                            # Weight loss based on whether samples use expert data
                            expert_weight = torch.where(expert_mask, 
                                                      1.0 + expert_loss_weight,  # Higher weight for expert samples
                                                      1.0)                        # Normal weight for regular samples
                            agent_loss = agent_loss * expert_weight
                        
                        loss += agent_loss.mean()
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
                        real_collision_reg += collision_info['real_collision_loss']
                        future_collision_reg += collision_info['future_collision_loss']
                        total_real_collisions += collision_info['total_real_collisions']
                        # Track if we're using future collision penalty (when no real collisions)
                        if not collision_info.get('using_real_collision_loss', True):
                            total_future_collisions += 1  # Count timesteps where future penalty was applied
                        
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
                real_collision_reg /= T  # average real collision regularization over time
                future_collision_reg /= T  # average future collision regularization over time
                
                # Apply spike regularization with scheduled weight
                total_loss += spike_reg_weight * spike_reg

                # Backpropagation and optimization
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                train_loss += total_loss.item()
            
            # Get current learning rate from optimizer
            current_lr = optimizer.param_groups[0]['lr']
            expert_info = ""
            if use_expert_training and epoch < max_expert_epochs and expert_loss_weight > 0:
                expert_info = f" | ExpertWeight: {expert_loss_weight:.3f} | ExpertRatio: {expert_ratio:.1%}"
            print(f"Epoch {epoch} | Loss: {total_loss:.4f} | LR: {current_lr:.2e} | CollisionWeight: {collision_loss_weight:.2e} | SpikeRegWeight: {spike_reg_weight:.2e} | Spikes: {total_spikes} | SpikeReg: {spike_reg:.4f} | RealCollisionReg: {real_collision_reg:.4f} | FutureCollisionReg: {future_collision_reg:.4f} | RealCollisions: {total_real_collisions} | FutureCollisionSteps: {total_future_collisions}{expert_info}")
        else:
            train_loss = 0.0
            for i, batch_data in enumerate(data_loader.train_loader):
                if len(batch_data) == 4:  # With case_idx
                    states, trajectories, gso, case_idx = batch_data
                else:  # Without case_idx (backward compatibility)
                    states, trajectories, gso = batch_data
                    
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
        
        # Evaluate on validation dataset
        if net_type == "snn" and data_loader.valid_loader is not None:
            val_metrics = evaluate_on_validation(model, data_loader, config, num_samples=10)
            print(f"Validation success rate: {val_metrics['val_success_rate']:.4f}")
            print(f"Validation agents reached: {val_metrics['val_agents_reached']}/{val_metrics['val_total_agents']}")
            print(f"Validation samples processed: {val_metrics['val_samples_processed']}")
        
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
