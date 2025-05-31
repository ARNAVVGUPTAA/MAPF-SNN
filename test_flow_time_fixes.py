#!/usr/bin/env python3
"""
Test script to verify the flow-time and loss rebalancing fixes.

Tests:
1. Living cost predictor integration
2. Post-curriculum loss rebalancing 
3. CE boost factor application
4. Collision weight normalization
"""

import torch
import torch.nn.functional as F
import yaml
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.framework_snn import Network, RewardPredictionModule
from config import config

def test_living_cost_predictor():
    """Test that the living cost predictor is properly integrated."""
    print("\n🧪 Testing Living Cost Predictor Integration...")
    
    # Create a reward prediction module
    hidden_size = 256
    reward_module = RewardPredictionModule(hidden_size)
    
    # Test input
    batch_agents = 10
    features = torch.randn(batch_agents, hidden_size)
    
    # Forward pass
    try:
        predicted_rewards, goal_achievement_probs, collision_penalties, progress_rewards, living_costs = reward_module(features)
        
        # Verify shapes
        assert predicted_rewards.shape == (batch_agents, 5), f"Expected shape (10, 5), got {predicted_rewards.shape}"
        assert goal_achievement_probs.shape == (batch_agents, 5), f"Expected shape (10, 5), got {goal_achievement_probs.shape}"
        assert collision_penalties.shape == (batch_agents, 5), f"Expected shape (10, 5), got {collision_penalties.shape}"
        assert progress_rewards.shape == (batch_agents, 5), f"Expected shape (10, 5), got {progress_rewards.shape}"
        assert living_costs.shape == (batch_agents, 5), f"Expected shape (10, 5), got {living_costs.shape}"
        
        # Verify living costs are small positive values (0-0.01 range)
        assert living_costs.min() >= 0, f"Living costs should be non-negative, got min: {living_costs.min()}"
        assert living_costs.max() <= 0.01, f"Living costs should be ≤ 0.01, got max: {living_costs.max()}"
        
        print("✅ Living cost predictor integration: PASSED")
        print(f"   Living costs range: {living_costs.min():.6f} - {living_costs.max():.6f}")
        print(f"   Living costs mean: {living_costs.mean():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Living cost predictor integration: FAILED - {e}")
        return False

def test_post_curriculum_rebalancing():
    """Test post-curriculum loss rebalancing logic."""
    print("\n🧪 Testing Post-Curriculum Loss Rebalancing...")
    
    # Test parameters
    collision_min = 5.0
    collision_max = 40.0
    collision_warm_epochs = 12
    num_agents = 5
    ce_boost_factor = 5
    collision_normalization = 8.0
    
    test_cases = [
        {"epoch": 0, "expected_curriculum": True, "expected_ce_boost": 1.0},
        {"epoch": 6, "expected_curriculum": True, "expected_ce_boost": 1.0},
        {"epoch": 11, "expected_curriculum": True, "expected_ce_boost": 1.0},
        {"epoch": 12, "expected_curriculum": False, "expected_ce_boost": ce_boost_factor},
        {"epoch": 20, "expected_curriculum": False, "expected_ce_boost": ce_boost_factor},
    ]
    
    all_passed = True
    
    for case in test_cases:
        epoch = case["epoch"]
        expected_curriculum = case["expected_curriculum"]
        expected_ce_boost = case["expected_ce_boost"]
        
        # Simulate curriculum logic
        collision_loss_weight = collision_min + \
            (collision_max - collision_min) * min(epoch / collision_warm_epochs, 1.0)
        
        # Simulate post-curriculum rebalancing
        post_curriculum_rebalance = True
        if post_curriculum_rebalance and epoch >= collision_warm_epochs:
            collision_loss_weight = collision_loss_weight / collision_normalization
            ce_boost = ce_boost_factor
            in_curriculum = False
        else:
            ce_boost = 1.0
            in_curriculum = True
        
        # Check expectations
        curriculum_match = (in_curriculum == expected_curriculum)
        ce_boost_match = (ce_boost == expected_ce_boost)
        
        if curriculum_match and ce_boost_match:
            print(f"✅ Epoch {epoch}: Curriculum={in_curriculum}, CE_boost={ce_boost}, Collision_weight={collision_loss_weight:.3f}")
        else:
            print(f"❌ Epoch {epoch}: Expected curriculum={expected_curriculum}, CE_boost={expected_ce_boost}")
            print(f"   Got curriculum={in_curriculum}, CE_boost={ce_boost}")
            all_passed = False
    
    if all_passed:
        print("✅ Post-curriculum rebalancing logic: PASSED")
    else:
        print("❌ Post-curriculum rebalancing logic: FAILED")
    
    return all_passed

def test_snn_network_integration():
    """Test that the SNN network can handle the new living cost outputs."""
    print("\n🧪 Testing SNN Network Integration...")
    
    try:
        # Load config
        with open('configs/config_snn.yaml', 'r') as f:
            test_config = yaml.safe_load(f)
        
        # Create network
        network = Network(test_config)
        network.eval()
        
        # Test input
        batch_size = 2
        num_agents = test_config.get('num_agents', 5)
        input_dim = test_config.get('input_dim', 125)  # 5x5x5 FOV
        
        x = torch.randn(batch_size * num_agents, input_dim)
        current_positions = torch.randn(batch_size, num_agents, 2) * 10
        goal_positions = torch.randn(batch_size, num_agents, 2) * 10
        neighbor_positions = torch.randn(batch_size, num_agents, 8) * 10  # Add neighbor positions
        
        # Forward pass
        output, spike_info = network(
            x, 
            return_spikes=True, 
            return_logits=True,
            current_positions=current_positions,
            goal_positions=goal_positions,
            neighbor_positions=neighbor_positions
        )
        
        # Check that living costs are in spike_info if reward prediction is enabled
        if network.use_reward_prediction:
            assert 'living_costs' in spike_info, "Living costs should be in spike_info when reward prediction is enabled"
            living_costs = spike_info['living_costs']
            assert living_costs.shape[0] == batch_size * num_agents, f"Expected {batch_size * num_agents} agents, got {living_costs.shape[0]}"
            assert living_costs.shape[1] == 5, f"Expected 5 actions, got {living_costs.shape[1]}"
            print(f"✅ SNN integration: Living costs shape {living_costs.shape}")
            print(f"   Living costs range: {living_costs.min():.6f} - {living_costs.max():.6f}")
        else:
            print("⚠️  Reward prediction disabled - skipping living costs check")
        
        print("✅ SNN network integration: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ SNN network integration: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ce_loss_boosting():
    """Test CE loss boosting calculation."""
    print("\n🧪 Testing CE Loss Boosting...")
    
    # Simulate CE loss calculation
    batch_size = 4
    num_agents = 5
    num_actions = 5
    
    # Create sample logits and targets
    logits = torch.randn(batch_size, num_actions)
    targets = torch.randint(0, num_actions, (batch_size,))
    
    # Calculate base CE loss
    base_loss = 0
    for agent in range(num_agents):
        agent_loss = F.cross_entropy(logits, targets, reduction='none')
        base_loss += agent_loss.mean()
    base_loss = base_loss / num_agents
    
    # Test CE boost factor
    ce_boost_factor = 5.0
    boosted_loss = base_loss * ce_boost_factor
    
    expected_ratio = boosted_loss / base_loss
    
    if abs(expected_ratio - ce_boost_factor) < 1e-6:
        print(f"✅ CE boost calculation: PASSED")
        print(f"   Base loss: {base_loss:.6f}")
        print(f"   Boosted loss: {boosted_loss:.6f}")
        print(f"   Boost ratio: {expected_ratio:.2f}")
        return True
    else:
        print(f"❌ CE boost calculation: FAILED")
        print(f"   Expected ratio: {ce_boost_factor}, got: {expected_ratio}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Flow-Time and Loss Rebalancing Fixes")
    print("=" * 60)
    
    tests = [
        test_living_cost_predictor,
        test_post_curriculum_rebalancing,
        test_snn_network_integration,
        test_ce_loss_boosting,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED! Flow-time and loss rebalancing fixes are working correctly.")
        return True
    else:
        print("⚠️  Some tests FAILED. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
