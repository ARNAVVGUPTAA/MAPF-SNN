"""
Quick test to verify evaluation doesn't crash
"""
import torch
import yaml
from models.swarm_lsm import SwarmLSM, SwarmTrainer
from data_loader import SNNDataset

# Load config
config = yaml.safe_load(open('configs/config_swarm.yaml'))

# Load small dataset
valid_dataset = SNNDataset(config, 'valid')
print(f"Loaded {len(valid_dataset)} episodes")

# Create network
network = SwarmLSM(num_agents=5, communication_range=3.0)
trainer = SwarmTrainer(network, device='cpu')

# Test evaluation on just 3 episodes
print("\nTesting evaluation on 3 episodes...")
try:
    metrics = trainer.evaluate(
        valid_dataset, 
        num_episodes=3,
        max_timesteps=20,
        num_ticks=10,
        log_ticks=True
    )
    print("\n✅ Evaluation completed successfully!")
    print(f"Success rate: {metrics['success_rate']:.1%}")
    print(f"Goal reach: {metrics['goal_reach_rate']:.1%}")
    print(f"Avg collisions: {metrics['avg_collisions']:.2f}")
except Exception as e:
    print(f"\n❌ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
