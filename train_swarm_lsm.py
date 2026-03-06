"""
Hardware-Native Swarm LSM Training
===================================

Clean, minimal training script.
No test files. No diagnostics clutter.
"""

import os
import sys
import yaml
import argparse
import torch
from datetime import datetime

from models.swarm_lsm import SwarmLSM, SwarmTrainer
from data_loader import SNNDataset


def load_config(config_path):
    """Load YAML config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")
    
    return {
        'device': device,
        'num_agents': config['swarm']['num_agents'],
        'communication_range': config['swarm']['communication_range'],
        'max_episodes': config['training']['max_episodes'],
        'test_episodes': config['training']['test_episodes'],
        'ridge_alpha': config['training']['ridge_alpha'],
        'num_ticks': config['training']['num_ticks'],
        'optimizer': config['training'].get('optimizer', 'ridge'),
        'sgd_epochs': config['training'].get('sgd_epochs', 50),
        'sgd_lr': config['training'].get('sgd_lr', 1e-3),
        'sgd_batch_size': config['training'].get('sgd_batch_size', 64),
        'log_dir': config['logging']['log_dir'],
        'full_config': config,  # Pass full config for SNNDataset
    }


def main():
    parser = argparse.ArgumentParser(description='Train Hardware-Native Swarm LSM')
    parser.add_argument('--config', type=str, default='configs/config_swarm.yaml')
    parser.add_argument('--log-ticks', action='store_true', 
                       help='Log per-tick details during evaluation (verbose)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🧠 HARDWARE-NATIVE SWARM LSM TRAINING")
    print("="*70)
    print("Modules:")
    print("  1. Topographic Intent Map (spatial broadcast)")
    print("  2. CPG (turn-taking oscillator)")
    print("  3. Shadow Caster (predictive VETO)")
    print("  4. Chemotaxis Receptors (scent-gradient goal nudge)")
    print("="*70 + "\n")
    
    # Load config
    config = load_config(args.config)
    
    # Load datasets
    print("📁 Loading datasets...")
    train_dataset = SNNDataset(config['full_config'], 'train')
    valid_dataset = SNNDataset(config['full_config'], 'valid')
    print(f"   Train: {len(train_dataset)} episodes")
    print(f"   Valid: {len(valid_dataset)} episodes")
    
    # Create network
    print("\n🏗️  Creating swarm network...")
    network = SwarmLSM(
        num_agents=config['num_agents'],
        communication_range=config['communication_range']
    )
    print(f"   Agents: {config['num_agents']}")
    print(f"   Communication range: {config['communication_range']:.1f}")
    
    # Create trainer
    trainer = SwarmTrainer(network, device=config['device'])
    
    # Collect liquid states
    X_train, Y_train = trainer.collect_states(train_dataset, max_episodes=config['max_episodes'], num_ticks=config['num_ticks'])
    
    # Train readouts
    if config['optimizer'] == 'sgd':
        trainer.train_sgd(X_train, Y_train,
                          epochs=config['sgd_epochs'],
                          lr=config['sgd_lr'],
                          batch_size=config['sgd_batch_size'])
    else:
        trainer.train_ridge(X_train, Y_train, alpha=config['ridge_alpha'])
    
    # Create save directory
    from datetime import datetime as _dt
    timestamp = _dt.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(config['log_dir'], f'swarm_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Evaluate (use --log-ticks flag to see per-tick details)
    metrics = trainer.evaluate(
        valid_dataset, 
        num_episodes=config['test_episodes'],
        max_timesteps=50,
        num_ticks=config['num_ticks'],
        log_ticks=args.log_ticks,
        save_dir=save_dir
    )
    
    # Save
    model_path = os.path.join(save_dir, 'swarm.pt')
    trainer.save(model_path)
    
    # Save detailed summary
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write("Hardware-Native Swarm LSM - Evaluation Results\n")
        f.write("="*70 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Agents:              {config['num_agents']}\n")
        f.write(f"Train episodes:      {config['max_episodes']}\n")
        f.write(f"Valid episodes:      {config['test_episodes']}\n")
        f.write(f"Optimizer:           {config['optimizer']}\n")
        if config['optimizer'] == 'sgd':
            f.write(f"SGD epochs:          {config['sgd_epochs']}\n")
            f.write(f"SGD lr:              {config['sgd_lr']}\n")
            f.write(f"SGD batch size:      {config['sgd_batch_size']}\n")
        else:
            f.write(f"Ridge alpha:         {config['ridge_alpha']}\n")
        f.write(f"Simulation ticks:    {config['num_ticks']}\n")
        f.write(f"Comm range:          {config['communication_range']}\n")
        f.write(f"Max timesteps:       50 (2.5x dataset max)\n\n")
        
        f.write("MAPF METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Success Rate:        {metrics['success_rate']:.2%}\n")
        f.write(f"Goal Reach Rate:     {metrics['goal_reach_rate']:.2%}\n")
        f.write(f"Avg Collisions/Ep:   {metrics['avg_collisions']:.2f}\n")
        f.write(f"Total Collisions:    {metrics['total_collisions']}\n")
        f.write(f"Avg Final Distance:  {metrics['avg_final_distance']:.3f}\n")
        f.write(f"Avg Timesteps:       {metrics['avg_timesteps']:.1f}\n")
        f.write(f"Total Agents:        {metrics['total_agents']}\n")
        f.write(f"Episodes Evaluated:  {metrics['total_episodes']}\n\n")
        
        f.write("SPIKE HEALTH\n")
        f.write("-"*70 + "\n")
        f.write(f"Mean spikes/neuron:  {metrics['spike_mean']:.3f}  (healthy: {0.1*config['num_ticks']:.1f}–{0.7*config['num_ticks']:.1f})\n")
        f.write(f"Max spikes/neuron:   {metrics['spike_max']:.3f}  (cap: {config['num_ticks']})\n")
        f.write(f"Dead neurons:        {metrics['dead_neuron_frac']:.1%}\n")
        f.write(f"Saturated neurons:   {metrics['saturated_frac']:.1%}\n")
        f.write(f"Diagnostics:         spike_raster.png, spike_phase.png, neuron_spike_histogram.png\n\n")
        
        f.write("ARCHITECTURE\n")
        f.write("-"*70 + "\n")
        f.write("✓ SpikingJelly LIF neurons (hardware-native)\n")
        f.write("✓ E/I organization (80% E, 20% I)\n")
        f.write("✓ Dale's Law (E-only projections)\n")
        f.write("✓ Topographic Intent Map (144 LIF neurons)\n")
        f.write("✓ CPG oscillator (32 LIF neurons)\n")
        f.write("✓ Shadow Caster (392 LIF neurons)\n")
        f.write("✓ Observation mesh (256 LIF neurons)\n")
        f.write("✓ Readout mesh (128 LIF neurons → 102 E)\n")
    
    print(f"\n💾 Saved to {save_dir}")
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
