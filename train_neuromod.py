#!/usr/bin/env python3
"""
Minimal Neuromodulated Training Script
=====================================

- Uses ONLY neuromodulated RL loss (dopamine/GABA)
- Supports latest model features (set_neuromodulators, reset_state)
- Sequence-level optimization (one backward per batch)
- Simple, robust logging

Run:
  python train_neuromod.py --epochs 50 --batch_size 16
"""
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import sys
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random

# Project imports
from config import config
from data_loader import SNNDataLoader
from models.framework_snn import Network
from optimizer_utils import setup_training_optimizer
from spikingjelly.activation_based.base import MemoryModule

# 📝 LOGGING SETUP - Capture all output to file
import sys
from datetime import datetime

class TeeOutput:
    """Capture print output to both console and file"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

# Setup logging with timestamp in logs folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)  # Create logs directory if it doesn't exist
log_file_path = os.path.join(logs_dir, f"training_output_{timestamp}.txt")
tee = TeeOutput(log_file_path)
sys.stdout = tee

print(f"📝 Logging all output to: {log_file_path}")

# Neuromodulated loss + utilities
from neuromodulated_loss import (
    create_neuromodulated_loss,
    create_training_mode_controller,
    detect_collisions,
    detect_goal_reached,
)


## Visualization components removed for clean academic version.


def initialize_weights(m):
    """Custom weight initialization for SNNs.
    
    Uses Kaiming initialization for better gradient flow and small positive
    biases to give neurons a head start toward their firing threshold.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        # Kaiming init is good for ReLU-like activations (like spikes)
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            # Give neurons a small positive bias to encourage firing initially
            torch.nn.init.constant_(m.bias, 0.1)


def safe_reset_snn(model: nn.Module):
    """Safely reset SNN memory modules and recurrent states."""
    for m in model.modules():
        if isinstance(m, MemoryModule):
            m.reset()
        # Also reset custom recurrent states in EILIFLayer
        if hasattr(m, 'reset_recurrent_state'):
            m.reset_recurrent_state()
        # Reset any stored previous states 
        if hasattr(m, 'prev_e_spikes'):
            m.prev_e_spikes = None
        if hasattr(m, 'prev_i_spikes'):
            m.prev_i_spikes = None


def update_positions_from_actions(positions: torch.Tensor, actions: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Apply actions to positions. actions shape: [batch, agents], positions: [batch, agents, 2]."""
    # 0=stay, 1=right, 2=up, 3=left, 4=down
    deltas = torch.tensor(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=positions.device, dtype=positions.dtype
    )
    move = deltas[actions.clamp(min=0, max=4)]  # [batch, agents, 2]
    new_pos = positions + move
    new_pos[..., 0] = new_pos[..., 0].clamp(0, grid_size - 1)
    new_pos[..., 1] = new_pos[..., 1].clamp(0, grid_size - 1)
    return new_pos


def ensure_positions_goals(device, batch_size, num_agents, grid_size, positions=None, goals=None):
    """Ensure we have valid positions/goals tensors; synthesize if missing."""
    if positions is None or positions.numel() == 0:
        # Simple spread pattern
        xs = torch.arange(num_agents, device=device) % grid_size
        ys = (torch.arange(num_agents, device=device) * 2) % grid_size
        pos = torch.stack([xs, ys], dim=-1).float()  # [agents, 2]
        positions = pos.unsqueeze(0).repeat(batch_size, 1, 1)
    if goals is None or goals.numel() == 0:
        # Opposite side pattern
        xs = (grid_size - 1) - (torch.arange(num_agents, device=device) % grid_size)
        ys = (grid_size - 1) - ((torch.arange(num_agents, device=device) * 2) % grid_size)
        goal = torch.stack([xs, ys], dim=-1).float()
        goals = goal.unsqueeze(0).repeat(batch_size, 1, 1)
    return positions, goals


def train(args):
    
    # Load config and set overrides
    cfg = dict(config)
    cfg['epochs'] = args.epochs
    cfg['batch_size'] = args.batch_size
    # Override batch sizes in train/valid configs
    if 'train' in cfg:
        cfg['train']['batch_size'] = args.batch_size
    if 'valid' in cfg:
        cfg['valid']['batch_size'] = max(1, args.batch_size // 2)  # Half for validation
    # Use GPU if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg['device'] = device
    grid_size = cfg.get('board_size', [9, 9])[0]

    print(f"🚀 Neuromod training on {device} | grid={grid_size} | epochs={args.epochs} | batch={args.batch_size}")

    # Visualization removed

    # Data (loads directly with padding - no sequence building needed!)
    loader = SNNDataLoader(cfg)
    train_loader = loader.train_loader
    valid_loader = loader.valid_loader
    
    if train_loader is None or len(train_loader) == 0:
        raise RuntimeError('No training data available!')
    
    print(f"📦 Data loaded | train_batches={len(train_loader)} | valid_batches={len(valid_loader) if valid_loader else 0}")

    # Model
    model = Network(cfg).to(device)
    model.apply(initialize_weights)  # Apply custom initialization
    print("🧠 Model weights initialized with Kaiming Uniform + positive bias.")
    model.train()

    # Optimizer
    optimizer = setup_training_optimizer(model, cfg)

    # Neuromodulated loss and controller - FIXED: Pass entire config
    loss_fn = create_neuromodulated_loss(cfg)
    total_epochs = cfg.get('epochs', args.epochs)
    mode_ctrl = create_training_mode_controller(total_epochs)

    # Hardcode RL loss scaling per prior decisions
    rl_loss_scale = 0.3

    # Expected expert trajectory length for switching to RL
    expected_expert_len = cfg.get('expected_expert_trajectory_length', 16)

    # Training tracking for visualization
    """
    🎯 TRAINING PHASES EXPLANATION:
    ===============================
    
    1. EXPLORATION (0-30% of training):
       - High dopamine baseline (+0.3 boost)
       - Reduced GABA inhibition (-0.1)
       - Encourages diverse action exploration
       - Higher reward sensitivity (1.2x)
       - Lower punishment sensitivity (0.8x)
    
    2. EXPLOITATION (30-80% of training):
       - Balanced neuromodulators (no boosts)
       - Standard reward/punishment scales (1.0x)
       - Network learns to exploit discovered patterns
       - Focuses on optimizing known good strategies
    
    3. STABILIZATION (80-100% of training):
       - Slightly reduced dopamine (-0.1)
       - Increased GABA for stability (+0.2)
       - Lower exploration, higher stability
       - Emphasis on punishment learning (1.1x)
       - Fine-tunes and stabilizes learned behaviors
    
    This mimics biological learning: explore → exploit → stabilize
    """

    all_losses = []
    all_rewards = []
    all_punishments = []
    all_dopamine = []
    all_gaba = []
    all_collisions = []
    all_goals = []
    prev_phase = None  # Track phase transitions

    for epoch in range(args.epochs):
        mode_ctrl.update_epoch(epoch)
        phase = mode_ctrl.get_training_phase()
        phase_adj = mode_ctrl.get_neuromodulator_adjustments()

        # 📊 Display phase transition information
        if epoch == 0 or (prev_phase and phase != prev_phase):
            progress = epoch / args.epochs * 100
            print(f"\n🧠 PHASE TRANSITION: Entering '{phase.upper()}' phase at epoch {epoch+1} ({progress:.1f}% complete)")
            print(f"   Adjustments: {phase_adj}")
        prev_phase = phase

        epoch_loss = 0.0
        epoch_rewards = []
        epoch_punishments = []
        epoch_dopamine = []
        epoch_gaba = []
        epoch_collisions = []
        epoch_goals = []
        batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch: states [B, T, A, 2, H, W], actions [B, T, A], gso [B, T, A, A]
            fovs, actions, gso = batch
            fovs, actions, gso = fovs.to(device), actions.to(device), gso.to(device)

            # 🎲 ADD INPUT NOISE DURING EARLY TRAINING 🎲
            if epoch < 5:  # Only during first 5 epochs
                noise_level = cfg.get('input_noise_level', 0.01)
                fovs = fovs + torch.randn_like(fovs) * noise_level

            B, T, A = actions.shape[0], actions.shape[1], actions.shape[2]
            if T == 0:
                print("⚠️  Skipping batch with T=0")
                continue
            
            if batch_idx == 0 and epoch == 0:
                print(f"\n📦 First batch: B={B}, T={T}, A={A}")
                print(f"   fovs: {tuple(fovs.shape)}, actions: {tuple(actions.shape)}, gso: {tuple(gso.shape)}")
            
            # Skip batch if no timesteps available
            if T <= 0:
                print(f"Skipping batch with T={T}")
                continue

            # Generate synthetic positions/goals (states already contain FOV data)
            positions = None
            goals = None
            current_positions, current_goals = ensure_positions_goals(device, B, A, grid_size, positions, goals)

            # Reset SNN state per batch
            safe_reset_snn(model)
            
            # Reset loss function state per batch to prevent batch size mismatches
            loss_fn.reset_state()
            
            optimizer.zero_grad()

            # Roll out sequence, accumulate all logits for trajectory learning
            all_logits = []  # Store logits at each timestep
            last_logits = None
            spike_counts = []
            voltage_stats = []
            position_trajectory = []  # 🔧 ADD: Store full position trajectory
            goal_trajectory = []      # 🔧 ADD: Store full goal trajectory
            
            with torch.set_grad_enabled(True):
                for t in range(T):
                    # 🔧 ADD: Store positions and goals at each timestep
                    position_trajectory.append(current_positions.clone())
                    goal_trajectory.append(current_goals.clone())
                    
                    # [B, agents, 2, H, W] -> [B*agents, 2, H, W]
                    fov_t = fovs[:, t]
                    bsz, nag, ch, h, w = fov_t.shape
                    fov_flat = fov_t.reshape(bsz * nag, ch, h, w)

                    # Coerce positions/goals to [B, A, 2] float to satisfy model expectations
                    try:
                        current_positions = current_positions.reshape(B, A, -1)[..., :2].contiguous().float()
                        current_goals = current_goals.reshape(B, A, -1)[..., :2].contiguous().float()
                    except Exception:
                        # Fallback to synthesized positions/goals if reshape fails
                        current_positions, current_goals = ensure_positions_goals(device, B, A, grid_size)

                    # Forward
                    out = model(fov_flat, positions=current_positions, goals=current_goals)
                    out = out.reshape(B, A, model.num_actions)  # [B, agents, actions]
                    all_logits.append(out)  # Store for sequence loss
                    last_logits = out

                    # 🔬 MONITOR RECEPTOR NEURON HEALTH - Track ReceptorNeuronLayer spikes
                    if t % 10 == 0 or t == T-1:
                        # Track spikes from ReceptorNeuronLayer
                        from models.receptor_dynamics import ReceptorLIFNeuron
                        receptor_spike_rates = []
                        receptor_ei_ratios = []
                        total_receptor_layers = 0
                        
                        # Scan all ReceptorNeuronLayer in the model
                        for name, module in model.named_modules():
                            if isinstance(module, ReceptorLIFNeuron):
                                total_receptor_layers += 1
                                spike_rate = module.get_spike_rate()
                                ei_ratio = module.get_ei_ratio()
                                receptor_spike_rates.append(spike_rate)
                                # Convert tensor to float if needed
                                receptor_ei_ratios.append(ei_ratio.item() if isinstance(ei_ratio, torch.Tensor) else ei_ratio)
                        
                        # Calculate average spike rate across all layers
                        if len(receptor_spike_rates) > 0:
                            overall_spike_rate = sum(receptor_spike_rates) / len(receptor_spike_rates)
                            avg_ei_ratio = sum(receptor_ei_ratios) / len(receptor_ei_ratios)
                        else:
                            overall_spike_rate = 0.0
                            avg_ei_ratio = 0.0
                        
                        spike_counts.append(overall_spike_rate)
                        
                        # 📊 Display detailed receptor stats at sequence end
                        if t == T-1:
                            print(f"\n    🔬 RECEPTOR NEURON HEALTH (t={t}, {total_receptor_layers} layers):")
                            print(f"       ⚡ Avg Spike Rate: {overall_spike_rate:.4f}")
                            print(f"       ⚖️  Avg E/I Ratio:  {avg_ei_ratio:.2f} (healthy: 3-5)")
                            
                            # Health diagnostics
                            if overall_spike_rate == 0:
                                print("       ❌ CRITICAL: ALL NEURONS SILENT!")
                            elif overall_spike_rate < 0.001:
                                print("       ⚠️  WARNING: Very low spike activity")
                            elif overall_spike_rate > 0.95:
                                print("       ⚠️  WARNING: Excessive spiking detected (saturation)")
                            elif avg_ei_ratio < 2.0:
                                print("       ⚠️  WARNING: Inhibition too strong")
                            elif avg_ei_ratio > 10.0:
                                print("       ⚠️  WARNING: Excitation too strong")
                            else:
                                print("       ✅ Receptor neurons functioning normally")

                    # Apply anti-stay bias to encourage movement
                    out_biased = out.clone()
                    out_biased[:, :, 0] -= 2.0  # Subtract bias from stay action (action 0)
                    
                    # Add exploration noise to encourage action diversity
                    if epoch <= 5:  # Only during early training
                        exploration_noise = torch.randn_like(out_biased) * 0.5
                        out_biased += exploration_noise
                    
                    # Update positions based on chosen actions (predictions)
                    pred_actions = torch.argmax(out_biased, dim=-1)  # [B, agents]
                    
                    current_positions = update_positions_from_actions(current_positions, pred_actions, grid_size)

            # Phase adjustments
            loss_fn.reward_scale = cfg.get('reward_scale', 1.0) * phase_adj['reward_scale']
            loss_fn.punishment_scale = cfg.get('punishment_scale', 1.0) * phase_adj['punishment_scale']

            # Use expert actions from dataset for supervision across the whole trajectory
            # Stack all timestep logits: [T, B, A, actions] -> [B*T, A, actions]
            stacked_logits = torch.stack(all_logits, dim=1)  # [B, T, A, actions]
            stacked_logits_flat = stacked_logits.reshape(B * T, A, -1)  # [B*T, A, actions]
            
            # Get expert actions: [B, T, A] -> [B*T, A]
            expert_actions_flat = actions.reshape(B * T, A).long()
            
            # Use final positions/goals for reward calculation (accumulated over trajectory)
            target_actions = expert_actions_flat[-B:]  # Last batch worth for final loss calc

            # Detect collisions and goal reached at the end of sequence
            collisions = detect_collisions(current_positions, threshold=0.5)
            goals_reached = detect_goal_reached(current_positions, current_goals, threshold=0.5)

            # Action distribution monitoring (for stay spam detection) - exclude goal-reached agents
            # Only count actions from agents who haven't reached their goals (actual model decisions)
            agents_not_at_goal = (goals_reached == 0).bool()  # Convert to boolean mask for agents still working toward goals
            
            if agents_not_at_goal.any():
                # Only analyze actions from agents who haven't reached goals
                active_actions = pred_actions[agents_not_at_goal]
                action_counts = torch.bincount(active_actions.flatten(), minlength=5)
                total_active_actions = active_actions.numel()
                action_dist = (action_counts.float() / total_active_actions * 100).cpu().numpy()
            else:
                # All agents reached goals - no active policy decisions to analyze
                action_dist = np.zeros(5)
                
            stay_pct = action_dist[0] if len(action_dist) > 0 else 0.0
            move_pct = sum(action_dist[1:]) if len(action_dist) > 1 else 0.0
            
            # Print action distribution at final timestep (only from agents still working toward goals)
            if t == T - 1:
                active_agent_count = agents_not_at_goal.sum().item()
                goal_reached_count = goals_reached.sum().item()
                print(f"    🎯 Action Distribution t={t} (Active: {active_agent_count}, At Goal: {goal_reached_count}): STAY={stay_pct:.1f}% | MOVE={move_pct:.1f}% | Actions: {action_dist}")
                if stay_pct > 80.0 and active_agent_count > 0:
                    print(f"    ⚠️  STAY SPAM DETECTED! {stay_pct:.1f}% stay actions from active agents")

            # Get spike rates for neuron death prevention
            try:
                spike_rates = model.get_spike_rates() if hasattr(model, 'get_spike_rates') else None
            except:
                spike_rates = None

            # Compute neuromodulated loss with spike rate monitoring
            loss_dict = loss_fn(
                model_outputs=last_logits,  # [B, agents, actions]
                target_actions=target_actions,  # [B, agents]
                positions=current_positions,  # [B, agents, 2]
                goals=current_goals,  # [B, agents, 2]
                collisions=collisions,
                goal_reached=goals_reached,
                phase_adjustments=phase_adj,  # Pass phase adjustments
                spike_rates=spike_rates,      # Pass spike rates for neuron death penalty
            )

            total_loss_tensor = loss_dict['loss']  # [B, agents]
            total_loss = total_loss_tensor.mean()
            final_loss = total_loss * rl_loss_scale
            
            # 🎲 ENTROPY REGULARIZATION FOR BETTER EXPLORATION 🎲
            action_probs = torch.softmax(last_logits, dim=-1)
            action_log_probs = torch.log(action_probs + 1e-8)  # Add epsilon to prevent log(0)
            entropy = -torch.sum(action_probs * action_log_probs, dim=-1).mean()
            
            entropy_coefficient = cfg.get('entropy_coefficient', 1.0)  # Configurable entropy weight
            entropy_loss = entropy_coefficient * entropy
            
            final_loss = final_loss - entropy_loss  # Subtract to encourage higher entropy (more exploration)
            
            # � ACTIVITY REGULARIZATION TO PREVENT NEURON DEATH 🔋
            # This encourages distributed activity and prevents "hero neuron" syndrome
            if spike_rates is not None and len(spike_rates) > 0:
                # Calculate the average spike rate across all monitored layers
                avg_network_spike_rate = torch.mean(torch.tensor([r for r in spike_rates.values()], device=final_loss.device))
                
                # The penalty is high when the average rate is close to zero
                # exp(-10x) gives: 0.1%→0.9999, 1%→0.9048, 3%→0.7408, 5%→0.6065
                activity_regularization_loss = torch.exp(-10.0 * avg_network_spike_rate)
                
                # Add it to the final loss
                regularization_strength = cfg.get('activity_regularization_strength', 0.05)  # Hyperparameter to tune
                final_loss = final_loss + regularization_strength * activity_regularization_loss
            
            # �🔥 ADDITIONAL LOSS SCALING FOR GRADIENT STABILITY 🔥
            if hasattr(cfg, 'loss_scale_factor') and cfg.loss_scale_factor != 1.0:
                final_loss = final_loss * cfg.loss_scale_factor

            # 🧠 DETAILED NEUROMODULATED LOSS BREAKDOWN - Show all components
            dopamine_level = loss_dict['dopamine'].mean().item()
            gaba_level = loss_dict['gaba'].mean().item()
            avg_reward = loss_dict['rewards'].mean().item()
            avg_punishment = loss_dict['punishments'].mean().item()
            
            # Calculate the ACTUAL scaled rewards used for neuromodulation
            # In exploration, punishment is divided by 5 (not 10) to maintain pressure
            reward_scale_factor = cfg.get('reward_scale_factor', 20.0)  # Same value as in neuromodulated_loss.py
            scaled_avg_reward = avg_reward / reward_scale_factor
            
            # Punishment scaling depends on phase: exploration = ÷5, others = ÷10
            if phase == 'exploration':
                punishment_divisor = 5.0  # Stricter in exploration
            else:
                punishment_divisor = 10.0  # More lenient later
            scaled_avg_punishment = avg_punishment / punishment_divisor
            
            entropy_value = entropy.item()
            
            # Activity regularization monitoring
            activity_reg_value = 0.0
            if spike_rates is not None and len(spike_rates) > 0:
                avg_network_spike_rate = np.mean(list(spike_rates.values()))
                activity_reg_value = np.exp(-10.0 * avg_network_spike_rate)
            
            collision_count = collisions.sum().item()
            goals_count = goals_reached.sum().item()
            total_agents = B * A
            
            # SNN Health Summary
            avg_spike_rate = np.mean(spike_counts) if spike_counts else 0
            spike_variance = np.var(spike_counts) if len(spike_counts) > 1 else 0
            
            # Diagnose neuronal death
            neuronal_death_warning = ""
            if avg_spike_rate < 0.001:
                neuronal_death_warning = " ⚠️  SILENT NEURONS DETECTED!"
            elif avg_spike_rate > 0.9:
                neuronal_death_warning = " ⚠️  OVER-SPIKING DETECTED!"
            elif spike_variance > 0.1:
                neuronal_death_warning = " ⚠️  UNSTABLE SPIKING!"
            
            # 📊 DISPLAY COMPREHENSIVE TRAINING STATS
            print("\n" + "=" * 85)
            print(f"🧠 NEUROMODULATED RL TRAINING | Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)}")
            print("=" * 85)
            
            # Loss Components
            print("\n📉 LOSS COMPONENTS:")
            print(f"   Raw RL Loss:       {total_loss.item():>10.6f}")
            print(f"   Scaled Loss:       {final_loss.item():>10.6f}  (×{rl_loss_scale})")
            print(f"   Entropy Bonus:     {entropy_value:>10.6f}  (coeff={entropy_coefficient:.2f})")
            print(f"   Activity Reg:      {activity_reg_value:>10.6f}  (str={cfg.get('activity_regularization_strength', 0.05):.3f})")
            
            # Reward/Punishment Breakdown
            print("\n🎁 REWARD/PUNISHMENT BREAKDOWN:")
            print(f"   Original Reward:   {avg_reward:>10.6f}  →  Scaled: {scaled_avg_reward:>8.6f}  (÷{reward_scale_factor:.0f})")
            print(f"   Original Punish:   {avg_punishment:>10.6f}  →  Scaled: {scaled_avg_punishment:>8.6f}  (÷{punishment_divisor:.0f})")
            
            # Neuromodulators
            print("\n🧪 NEUROMODULATORS:")
            print(f"   Dopamine:          {dopamine_level:>10.3f}  (baseline={loss_fn.dopamine_baseline:.2f})")
            print(f"   GABA:              {gaba_level:>10.3f}  (baseline={loss_fn.gaba_baseline:.2f})")
            print(f"   Training Phase:    {phase:>10s}")
            
            # Agent Performance
            print("\n🤖 AGENT PERFORMANCE:")
            collision_pct = (collision_count / total_agents * 100) if total_agents > 0 else 0
            goals_pct = (goals_count / total_agents * 100) if total_agents > 0 else 0
            print(f"   Collisions:        {int(collision_count):>4d}/{int(total_agents):<4d}  ({collision_pct:>5.1f}%)")
            print(f"   Goals Reached:     {int(goals_count):>4d}/{int(total_agents):<4d}  ({goals_pct:>5.1f}%)")
            
            # SNN Health Summary
            print(f"\n🔬 SNN HEALTH:")
            print(f"   Avg Spike Rate:    {avg_spike_rate:>10.4f}{neuronal_death_warning}")
            print(f"   Spike Variance:    {spike_variance:>10.4f}")
            
            print("\n" + "=" * 85)
            
            # (Visualization removed)
                
            final_loss.backward()
            
            # � EMERGENCY NaN/INF PROTECTION - DETECT AND FIX IMMEDIATELY 🚨
            if hasattr(cfg, 'use_nan_protection') and cfg.use_nan_protection:
                nan_detected = False
                inf_detected = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            param.grad.data.zero_()  # Zero out NaN gradients
                            nan_detected = True
                        elif torch.isinf(param.grad).any():
                            param.grad.data.zero_()  # Zero out inf gradients
                            inf_detected = True
                
                if nan_detected or inf_detected:
                    print(f"    🚨 EMERGENCY: {'NaN' if nan_detected else ''}{'/' if nan_detected and inf_detected else ''}{'Inf' if inf_detected else ''} gradients detected and zeroed!")
            
            # 🔥 GRADIENT CLIPPING BY NORM - MORE AGGRESSIVE FOR STABILITY 🔥
            max_norm = 0.5  # REDUCED from 1.0 - More aggressive clipping for high gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            
            # 🔍 VERIFY: Final gradient safety check
            max_final_grad = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    max_final_grad = max(max_final_grad, grad_norm)
            
            if max_final_grad > 0.1:  # Only warn if gradients are unexpectedly large
                print(f"    ⚠️  WARNING: max_grad={max_final_grad:.2e} after clamping!")
            
            # � VERIFY: Check final gradient safety
            max_final_grad = 0.0
            inf_count_final = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    max_final_grad = max(max_final_grad, grad_norm)
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        inf_count_final += 1
            
            print(f"    ✅ FINAL CHECK: max_grad={max_final_grad:.2e}, inf/nan_params={inf_count_final}")
            
            # Monitor gradients in SNN components (simplified)
            gradient_stats = {
                'feature_extractor': {'grad_norm': 0, 'params': 0},
                'snn_block': {'grad_norm': 0, 'params': 0},
                'output_layer': {'grad_norm': 0, 'params': 0}
            }
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Determine component
                    component = 'output_layer'  # default
                    if 'feature_extractor' in name:
                        component = 'feature_extractor'
                    elif 'snn_block' in name:
                        component = 'snn_block'
                    
                    grad_norm = param.grad.norm().item()
                    gradient_stats[component]['grad_norm'] += grad_norm
                    gradient_stats[component]['params'] += 1
            
            # Print simple gradient summary
            grad_details = []
            for comp, stats in gradient_stats.items():
                if stats['params'] > 0:
                    avg_grad = stats['grad_norm'] / stats['params']
                    grad_details.append(f"{comp}={avg_grad:.2e}")
            
            if grad_details:
                grad_str = ", ".join(grad_details)
                print(f"    🔥 Gradients: {grad_str}")
            
            optimizer.step()
            optimizer.zero_grad()

            # Optionally set neuromodulators on the model (if supported)
            if hasattr(model, 'set_neuromodulators'):
                avg_dopa = loss_dict['dopamine'].mean().item() + phase_adj['dopamine_boost']
                avg_gaba = loss_dict['gaba'].mean().item() + phase_adj['gaba_reduction']
                model.set_neuromodulators(avg_dopa, avg_gaba)

            # Accumulate metrics
            epoch_loss += total_loss.item()
            epoch_rewards.append(avg_reward)
            epoch_punishments.append(avg_punishment)
            epoch_dopamine.append(dopamine_level)
            epoch_gaba.append(gaba_level)
            epoch_collisions.append(collision_count)
            epoch_goals.append(goals_count)
            batches += 1

        # Store epoch averages
        all_losses.append(epoch_loss / max(batches, 1))
        all_rewards.append(np.mean(epoch_rewards) if epoch_rewards else 0)
        all_punishments.append(np.mean(epoch_punishments) if epoch_punishments else 0)
        all_dopamine.append(np.mean(epoch_dopamine) if epoch_dopamine else 0)
        all_gaba.append(np.mean(epoch_gaba) if epoch_gaba else 0)
        all_collisions.append(np.mean(epoch_collisions) if epoch_collisions else 0)
        all_goals.append(np.mean(epoch_goals) if epoch_goals else 0)

        # 📊 EPOCH SUMMARY with comprehensive stats
        print(f"\n{'='*85}")
        print(f"✅ EPOCH {epoch+1}/{args.epochs} COMPLETE | Phase: {phase.upper()}")
        print(f"{'='*85}")
        print(f"   Avg Loss:          {all_losses[-1]:>10.6f}")
        print(f"   Avg Reward:        {all_rewards[-1]:>10.6f}")
        print(f"   Avg Punishment:    {all_punishments[-1]:>10.6f}")
        print(f"   Avg Dopamine:      {all_dopamine[-1]:>10.3f}")
        print(f"   Avg GABA:          {all_gaba[-1]:>10.3f}")
        print(f"   Avg Collisions:    {all_collisions[-1]:>10.2f}")
        print(f"   Avg Goals:         {all_goals[-1]:>10.2f}")
        print(f"   Batches Processed: {batches:>10d}")
        print(f"{'='*85}\n")

        # VALIDATION PHASE - Run after each epoch
        if valid_loader is not None:
            model.eval()  # Switch to evaluation mode
            
            # CRITICAL: Reset all SNN states before validation to prevent batch size mismatches
            safe_reset_snn(model)
            
            # CRITICAL: Reset loss function state to prevent batch size mismatches
            loss_fn.reset_state()
            
            val_loss = 0.0
            val_rewards = []
            val_punishments = []
            val_collisions = []
            val_goals = []
            val_batches = 0
            
            print(f"🔍 Running validation after epoch {epoch+1}...")
            
            with torch.no_grad():  # Disable gradients for validation
                for val_batch_idx, val_batch in enumerate(valid_loader):
                    # Unpack validation batch
                    val_fovs, val_actions, val_gso = val_batch
                    val_fovs, val_actions, val_gso = val_fovs.to(device), val_actions.to(device), val_gso.to(device)
                    B, T, A = val_actions.shape[0], val_actions.shape[1], val_actions.shape[2]
                    
                    if T <= 0:
                        continue
                    
                    # Setup positions/goals for validation
                    val_current_positions, val_current_goals = ensure_positions_goals(device, B, A, grid_size, None, None)
                    
                    # Reset SNN state for validation batch
                    safe_reset_snn(model)
                    
                    # Run validation forward pass
                    val_last_logits = None
                    for t in range(T):
                        val_fov_t = val_fovs[:, t]
                        bsz, nag, ch, h, w = val_fov_t.shape
                        val_fov_flat = val_fov_t.reshape(bsz * nag, ch, h, w)
                        
                        val_current_positions = val_current_positions.reshape(B, A, -1)[..., :2].contiguous().float()
                        val_current_goals = val_current_goals.reshape(B, A, -1)[..., :2].contiguous().float()
                        
                        val_out = model(val_fov_flat, positions=val_current_positions, goals=val_current_goals)
                        val_out = val_out.reshape(B, A, model.num_actions)
                        val_last_logits = val_out
                        
                        # Update positions for next timestep
                        val_action_indices = val_actions[:, t].long().clamp(0, model.num_actions-1)
                        val_current_positions = update_positions_from_actions(val_current_positions, val_action_indices, grid_size)
                    
                    if val_last_logits is not None:
                        # Compute validation loss using same loss function
                        val_final_actions = val_actions[:, -1].long().clamp(0, model.num_actions-1)
                        # Get validation spike rates
                        try:
                            val_spike_rates = model.get_spike_rates() if hasattr(model, 'get_spike_rates') else None
                        except:
                            val_spike_rates = None
                            
                        val_loss_dict = loss_fn(
                            model_outputs=val_last_logits, 
                            target_actions=val_final_actions, 
                            positions=val_current_positions, 
                            goals=val_current_goals,
                            collisions=detect_collisions(val_current_positions),
                            phase_adjustments=phase_adj,
                            spike_rates=val_spike_rates,
                            goal_reached=detect_goal_reached(val_current_positions, val_current_goals)
                        )
                        
                        val_total_loss = val_loss_dict['loss'].mean()
                        val_avg_reward = val_loss_dict['rewards'].mean().item()
                        val_avg_punishment = val_loss_dict['punishments'].mean().item()
                        val_collision_count = detect_collisions(val_current_positions).sum().item()
                        val_goals_count = detect_goal_reached(val_current_positions, val_current_goals).sum().item()
                        val_total_agents = B * A
                        
                        val_loss += val_total_loss.item()
                        val_rewards.append(val_avg_reward)
                        val_punishments.append(val_avg_punishment)
                        val_collisions.append(val_collision_count)
                        val_goals.append(val_goals_count)
                        val_batches += 1
            
            # Calculate validation totals (not averages)
            if val_batches > 0:
                avg_val_loss = val_loss / val_batches
                avg_val_reward = sum(val_rewards) / len(val_rewards)
                avg_val_punishment = sum(val_punishments) / len(val_punishments)
                total_val_collisions = sum(val_collisions)
                total_val_goals = sum(val_goals)
                
                # Calculate percentages
                total_val_agents = val_batches * B * A
                val_collision_pct = (total_val_collisions / total_val_agents * 100) if total_val_agents > 0 else 0
                val_goals_pct = (total_val_goals / total_val_agents * 100) if total_val_agents > 0 else 0
                
                print(f"\n{'='*85}")
                print(f"📊 VALIDATION RESULTS | Epoch {epoch+1}")
                print(f"{'='*85}")
                print(f"   Val Loss:          {avg_val_loss:>10.6f}")
                print(f"   Val Reward:        {avg_val_reward:>10.6f}")
                print(f"   Val Punishment:    {avg_val_punishment:>10.6f}")
                print(f"   Val Collisions:    {total_val_collisions:>4d}  ({val_collision_pct:>5.1f}%)")
                print(f"   Val Goals:         {total_val_goals:>4d}  ({val_goals_pct:>5.1f}%)")
                print(f"   Val Batches:       {val_batches:>10d}")
                print(f"{'='*85}\n")
            else:
                print(f"⚠️  No validation batches processed")
            
            model.train()  # Switch back to training mode
        else:
            print(f"⚠️  No validation data available for epoch {epoch+1}")

        # MODEL SAVING - Save after every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_dir = cfg.get('save_dir', 'checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(save_dir, f'neuromod_snn_epoch_{epoch+1}.pt')
            
            # Save model state, optimizer state, and training info
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'training_metrics': {
                    'losses': all_losses,
                    'rewards': all_rewards,
                    'punishments': all_punishments,
                    'dopamine': all_dopamine,
                    'gaba': all_gaba,
                    'collisions': all_collisions,
                    'goals': all_goals
                },
                'training_phase': phase,
                'phase_adjustments': phase_adj
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Model checkpoint saved: {checkpoint_path}")
            
            # Also save as 'latest' for easy resuming
            latest_path = os.path.join(save_dir, 'neuromod_snn_latest.pt')
            torch.save(checkpoint, latest_path)
            print(f"💾 Latest checkpoint updated: {latest_path}")

    # (Validation & periodic visualization removed for minimal script)

    # (Dashboard cleanup removed)

    # FINAL MODEL SAVE
    save_dir = cfg.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    final_checkpoint_path = os.path.join(save_dir, 'neuromod_snn_final.pt')
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': cfg,
        'training_metrics': {
            'losses': all_losses,
            'rewards': all_rewards,
            'punishments': all_punishments,
            'dopamine': all_dopamine,
            'gaba': all_gaba,
            'collisions': all_collisions,
            'goals': all_goals
        },
        'final_phase': phase,
        'completed': True
    }
    
    torch.save(final_checkpoint, final_checkpoint_path)
    print(f"💾 Final model saved: {final_checkpoint_path}")

    print("🎉 Training complete!")
    
    # 📝 Close log file
    if hasattr(sys.stdout, 'close'):
        print(f"📝 Training log saved to: {log_file_path}")
        sys.stdout.close()
        sys.stdout = sys.__stdout__  # Restore original stdout


## Auxiliary visualization & validation helpers removed.


if __name__ == '__main__':
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\n🛑 Training interrupted by user")
        sys.exit(0)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    try:
        train(args)
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        # Close log file on interrupt
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        signal_handler(signal.SIGINT, None)
    finally:
        # Ensure log file is closed
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
            sys.stdout = sys.__stdout__
