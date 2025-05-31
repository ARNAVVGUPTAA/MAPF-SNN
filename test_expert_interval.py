#!/usr/bin/env python3
"""
Test script for interval-based expert training.
"""

from expert_utils import should_apply_expert_training, compute_interval_expert_weight

def test_expert_interval():
    """Test the new interval-based expert training logic."""
    print("Testing interval-based expert training:")
    print("Expert training should activate at epochs 9, 19, 29, 39... (0-indexed)")
    print()
    
    # Test epochs 0-50 to see when expert training is applied
    for epoch in range(50):
        should_apply = should_apply_expert_training(epoch, expert_interval=10)
        expert_weight = compute_interval_expert_weight(expert_weight=0.6)
        
        if should_apply:
            print(f"Epoch {epoch}: Expert training ACTIVE - Weight: {expert_weight:.1f}")
    
    print()
    print("Configuration summary:")
    print("- Expert training applies every 10 epochs starting from epoch 10 (1-indexed)")
    print("- Expert ratio: 10% of samples per batch")
    print("- Expert weight: 0.6 (constant when active)")

if __name__ == "__main__":
    test_expert_interval()
