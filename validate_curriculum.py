#!/usr/bin/env python3
"""
Quick validation test for collision curriculum learning integration.
Tests that the training code properly loads and applies curriculum settings.
"""

import yaml
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_loading():
    """Test that the config file loads correctly with curriculum parameters."""
    
    config_path = "configs/config_snn.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Config file loaded successfully")
        
        # Check curriculum parameters
        curriculum_enabled = config.get('use_collision_curriculum', False)
        curriculum_epochs = config.get('collision_curriculum_epochs', 10)
        start_ratio = config.get('collision_curriculum_start_ratio', 0.1)
        collision_weight = config.get('collision_loss_weight', 15.0)
        
        print(f"✓ Collision curriculum enabled: {curriculum_enabled}")
        print(f"✓ Curriculum epochs: {curriculum_epochs}")
        print(f"✓ Start ratio: {start_ratio}")
        print(f"✓ Final collision weight: {collision_weight}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_training_import():
    """Test that the training module imports correctly with new functions."""
    
    try:
        from train import collision_curriculum_schedule, cosine_annealing_schedule
        print("✓ Training functions imported successfully")
        
        # Test curriculum function
        weight = collision_curriculum_schedule(5, 15.0, 10, 0.1)
        expected = 1.5 + (15.0 - 1.5) * (5 / 10)  # Should be 8.25
        
        if abs(weight - expected) < 0.01:
            print(f"✓ Curriculum function working: epoch 5 → {weight:.2f}")
        else:
            print(f"✗ Curriculum function error: expected {expected:.2f}, got {weight:.2f}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Training import failed: {e}")
        return False

def test_curriculum_vs_cosine():
    """Compare curriculum and cosine annealing schedules."""
    
    print("\nComparison: Curriculum vs Cosine Annealing")
    print("=" * 50)
    print("Epoch | Curriculum | Cosine    | Difference")
    print("-" * 45)
    
    from train import collision_curriculum_schedule, cosine_annealing_schedule
    
    for epoch in range(0, 15, 2):
        curriculum = collision_curriculum_schedule(epoch, 15.0, 10, 0.1)
        cosine = cosine_annealing_schedule(epoch, 15.0, 50, 0.01, 0)
        diff = curriculum - cosine
        
        print(f"{epoch:5d} | {curriculum:10.2f} | {cosine:9.2f} | {diff:+10.2f}")

def main():
    """Run all validation tests."""
    
    print("Collision Curriculum Learning - Validation Tests")
    print("=" * 55)
    
    all_passed = True
    
    # Test 1: Config loading
    print("\n1. Testing configuration loading...")
    all_passed &= test_config_loading()
    
    # Test 2: Training imports
    print("\n2. Testing training module imports...")
    all_passed &= test_training_import()
    
    # Test 3: Schedule comparison
    print("\n3. Comparing curriculum vs cosine schedules...")
    test_curriculum_vs_cosine()
    
    # Summary
    print("\n" + "=" * 55)
    if all_passed:
        print("✓ All tests passed! Curriculum learning is ready to use.")
        print("\nTo start training with curriculum:")
        print("  python train.py configs/config_snn.yaml")
        print("\nTo disable curriculum (use cosine annealing):")
        print("  Set 'use_collision_curriculum: false' in config")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
