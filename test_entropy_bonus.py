#!/usr/bin/env python3
"""
Test script to demonstrate the entropy bonus exploration schedule.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import entropy_bonus_schedule, compute_entropy_bonus
import torch

def test_entropy_bonus_schedule():
    """Test the entropy bonus schedule with the actual configuration values."""
    
    # Configuration from config_snn.yaml
    initial_entropy_weight = 0.02  # entropy_bonus_weight
    bonus_epochs = 15              # entropy_bonus_epochs
    decay_type = 'linear'          # entropy_bonus_decay_type
    
    print("Entropy Bonus Exploration Schedule")
    print("=" * 45)
    print(f"Initial entropy weight: {initial_entropy_weight}")
    print(f"Bonus epochs: {bonus_epochs}")
    print(f"Decay type: {decay_type}")
    print()
    
    print("Epoch | Entropy Weight | Status")
    print("-" * 35)
    
    for epoch in range(20):  # Test beyond bonus epochs
        weight = entropy_bonus_schedule(epoch, initial_entropy_weight, bonus_epochs, decay_type)
        
        if epoch < bonus_epochs:
            progress = epoch / bonus_epochs * 100
            status = f"Exploration ({progress:5.1f}%)"
        else:
            status = "No Bonus"
            
        print(f"{epoch:5d} | {weight:13.4f} | {status}")
    
    print()
    print("Test different decay types:")
    print("-" * 35)
    test_epoch = 7  # Middle of exploration period
    
    for decay in ['linear', 'exponential', 'cosine']:
        weight = entropy_bonus_schedule(test_epoch, initial_entropy_weight, bonus_epochs, decay)
        print(f"{decay:12s} | {weight:13.4f}")
    
    print()
    print("Key Benefits:")
    print("- Encourages exploration of diverse actions in early training")
    print("- Higher entropy = more uniform action probabilities = more exploration") 
    print("- Gradually reduces exploration bonus to allow convergence")
    print("- Prevents premature convergence to suboptimal deterministic policies")

def test_entropy_computation():
    """Test the entropy bonus computation with sample logits."""
    print("\nEntropy Bonus Computation Test")
    print("=" * 30)
    
    # Create sample logits for different scenarios
    batch_size, num_actions = 4, 5
    
    # Scenario 1: Uniform distribution (high entropy)
    uniform_logits = torch.zeros(batch_size, num_actions)
    uniform_entropy = compute_entropy_bonus(uniform_logits)
    
    # Scenario 2: Deterministic distribution (low entropy)
    deterministic_logits = torch.zeros(batch_size, num_actions)
    deterministic_logits[:, 0] = 10.0  # Strong preference for action 0
    deterministic_entropy = compute_entropy_bonus(deterministic_logits)
    
    # Scenario 3: Slightly biased distribution (medium entropy)
    biased_logits = torch.zeros(batch_size, num_actions)
    biased_logits[:, 0] = 1.0  # Slight preference for action 0
    biased_entropy = compute_entropy_bonus(biased_logits)
    
    print(f"Uniform distribution entropy: {uniform_entropy:.4f}")
    print(f"Biased distribution entropy:  {biased_entropy:.4f}")
    print(f"Deterministic entropy:        {deterministic_entropy:.4f}")
    print()
    print("Higher entropy = better exploration bonus!")
    print("The model gets rewarded for keeping its options open.")

if __name__ == "__main__":
    test_entropy_bonus_schedule()
    test_entropy_computation()
