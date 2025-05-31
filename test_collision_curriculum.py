#!/usr/bin/env python3
"""
Test script to demonstrate the collision loss curriculum learning schedule.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import collision_curriculum_schedule

def test_collision_curriculum():
    """Test the collision curriculum schedule with the actual configuration values."""
    
    # Configuration from config_snn.yaml
    final_collision_weight = 15.0  # collision_loss_weight
    curriculum_epochs = 10         # collision_curriculum_epochs
    start_ratio = 0.1              # collision_curriculum_start_ratio (10% of final weight)
    
    print("Collision Loss Curriculum Learning Schedule")
    print("=" * 50)
    print(f"Final collision weight: {final_collision_weight}")
    print(f"Curriculum epochs: {curriculum_epochs}")
    print(f"Start ratio: {start_ratio} ({start_ratio*100}% of final weight)")
    print(f"Starting weight: {start_ratio * final_collision_weight}")
    print()
    
    print("Epoch | Collision Weight | Progress")
    print("-" * 35)
    
    for epoch in range(15):  # Test beyond curriculum epochs
        weight = collision_curriculum_schedule(epoch, final_collision_weight, curriculum_epochs, start_ratio)
        
        if epoch < curriculum_epochs:
            progress = epoch / curriculum_epochs * 100
            status = f"Curriculum ({progress:5.1f}%)"
        else:
            status = "Full Weight"
            
        print(f"{epoch:5d} | {weight:15.2f} | {status}")
    
    print()
    print("Key Benefits:")
    print("- Starts with low collision penalty (1.5) to allow exploration")
    print("- Gradually increases to full penalty (15.0) over 10 epochs") 
    print("- Prevents early convergence to collision-avoiding but suboptimal paths")
    print("- Maintains full penalty after curriculum period")

if __name__ == "__main__":
    test_collision_curriculum()
