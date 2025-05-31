#!/usr/bin/env python3

"""
Test script to verify goal proximity reward and debug fallback usage.
"""

import torch
import sys
import os
from collision_utils import (
    load_initial_positions_from_dataset, 
    load_goal_positions_from_dataset,
    compute_goal_proximity_reward
)

def test_goal_proximity_reward():
    """Test the goal proximity reward computation"""
    print("=== Testing Goal Proximity Reward ===")
    
    # Create sample positions
    batch_size, num_agents = 2, 5
    
    # Current positions (agents scattered around)
    current_positions = torch.tensor([
        [[10, 10], [5, 5], [15, 15], [0, 0], [20, 20]],  # Batch 1
        [[12, 8], [3, 7], [18, 12], [2, 1], [25, 25]]    # Batch 2
    ], dtype=torch.float32)
    
    # Goal positions (targets for agents)
    goal_positions = torch.tensor([
        [[12, 12], [5, 6], [15, 14], [1, 1], [19, 21]],  # Batch 1 goals
        [[11, 9], [4, 6], [17, 13], [1, 2], [24, 26]]    # Batch 2 goals
    ], dtype=torch.float32)
    
    print(f"Current positions:\n{current_positions}")
    print(f"Goal positions:\n{goal_positions}")
    
    # Test different reward types
    for reward_type in ['inverse', 'exponential', 'linear']:
        reward = compute_goal_proximity_reward(current_positions, goal_positions, 
                                             reward_type=reward_type, max_distance=10.0)
        print(f"{reward_type.capitalize()} reward: {reward.item():.4f}")

def test_dataset_loading_with_fallback():
    """Test dataset loading to identify when fallback is triggered"""
    print("\n=== Testing Dataset Loading (Fallback Detection) ===")
    
    dataset_root = 'dataset/5_8_28'
    
    # Test with valid case indices
    print("Testing with valid case indices [0, 1, 2]:")
    try:
        initial_pos = load_initial_positions_from_dataset(dataset_root, [0, 1, 2], 'train')
        goal_pos = load_goal_positions_from_dataset(dataset_root, [0, 1, 2], 'train')
        print("✓ Successfully loaded positions without fallback")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test with some invalid case indices to trigger fallback
    print("\nTesting with invalid case indices [999, 1000, 1001]:")
    try:
        initial_pos = load_initial_positions_from_dataset(dataset_root, [999, 1000, 1001], 'train')
        goal_pos = load_goal_positions_from_dataset(dataset_root, [999, 1000, 1001], 'train')
        print("✓ Handled missing files (check warnings above)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test with very large case indices that definitely don't exist
    print("\nTesting with very large case indices [9999, 10000]:")
    try:
        initial_pos = load_initial_positions_from_dataset(dataset_root, [9999, 10000], 'train')
        goal_pos = load_goal_positions_from_dataset(dataset_root, [9999, 10000], 'train')
        print("✓ Handled missing files (check warnings above)")
        print(f"Final positions shape: {initial_pos.shape}, {goal_pos.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")

def check_dataset_size():
    """Check how many cases actually exist in the dataset"""
    print("\n=== Checking Dataset Size ===")
    
    dataset_path = 'dataset/5_8_28/train'
    if os.path.exists(dataset_path):
        cases = [d for d in os.listdir(dataset_path) if d.startswith('case_')]
        case_numbers = []
        for case in cases:
            try:
                case_num = int(case.split('_')[1])
                case_numbers.append(case_num)
            except:
                pass
        
        if case_numbers:
            case_numbers.sort()
            print(f"Found {len(case_numbers)} cases")
            print(f"Case range: {min(case_numbers)} to {max(case_numbers)}")
            print(f"First 10 cases: {case_numbers[:10]}")
            print(f"Last 10 cases: {case_numbers[-10:]}")
            
            # Check for missing cases in the range
            full_range = set(range(min(case_numbers), max(case_numbers) + 1))
            missing_cases = full_range - set(case_numbers)
            if missing_cases:
                print(f"Missing cases in range: {sorted(list(missing_cases))[:20]}...")  # Show first 20
        else:
            print("No valid case directories found")
    else:
        print(f"Dataset path does not exist: {dataset_path}")

if __name__ == "__main__":
    test_goal_proximity_reward()
    test_dataset_loading_with_fallback()
    check_dataset_size()
