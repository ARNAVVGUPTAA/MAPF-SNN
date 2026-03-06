#!/usr/bin/env python3
"""
Re-process existing dataset with mistake injection.
This just re-runs the recording step on existing CBS solutions.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from record import record_env
import torch

config = {
    'device': torch.device('cpu'),
    'board_size': [9, 9],  # Dataset 5_8_28 uses 9x9 grid
    'num_agents': 5,
    'max_time': 100,
    'min_time': 0,
    'sensor_range': 3
}

print("="*70)
print("Re-processing existing dataset with mistake injection")
print("="*70)
print("This will:")
print("  1. Keep existing CBS solutions (fast!)")
print("  2. Re-generate states with 10% mistake injection")
print("  3. Inject recovery trajectories for robustness")
print()

# Re-record train set with mistake injection
print("Processing training set...")
record_env('../dataset/5_8_28/train', config, epsilon=0.10)

print("\nProcessing validation set...")
record_env('../dataset/5_8_28/valid', config, epsilon=0.10)

print("\n" + "="*70)
print("✅ DONE! Dataset re-processed with mistake injection")
print("="*70)
print("\nNext step: Train the model")
print("  cd ..")
print("  python train_lsm.py --config configs/config_lsm.yaml")
