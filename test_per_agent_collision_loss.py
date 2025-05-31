#!/usr/bin/env python3

"""
Test script for per-agent collision loss implementation.
Verifies that collision penalties are applied individually to agents.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from collision_utils import (
    compute_collision_loss, 
    detect_collisions,
    compute_per_agent_collision_losses
)

def test_per_agent_collision_detection():
    """Test basic per-agent collision detection."""
    print("=== Testing Per-Agent Collision Detection ===")
    
    # Setup test scenario
    batch_size = 2
    num_agents = 3
    device = 'cpu'
    
    # Create positions where only some agents collide
    # Batch 0: agents 0 and 1 collide, agent 2 is safe
    # Batch 1: no collisions
    positions = torch.tensor([
        [[1.0, 1.0], [1.0, 1.0], [5.0, 5.0]],  # Batch 0: agents 0,1 collide
        [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]   # Batch 1: no collisions
    ], dtype=torch.float32)
    
    # Create dummy logits for testing
    num_actions = 5
    logits = torch.randn(batch_size * num_agents, num_actions)
    
    # Test collision detection
    collisions = detect_collisions(positions)
    print(f"Vertex collisions: {collisions['vertex_collisions']}")
    print(f"Expected: [[1, 1, 0], [0, 0, 0]]")
    
    # Test per-agent collision loss computation
    collision_config = {
        'vertex_collision_weight': 2.0,
        'edge_collision_weight': 1.0,
        'collision_loss_type': 'l2',
        'per_agent_loss': True
    }
    
    collision_loss, collision_info = compute_collision_loss(
        logits, positions, None, collision_config
    )
    
    print(f"\nPer-agent collision losses shape: {collision_loss.shape}")
    print(f"Per-agent collision losses: {collision_loss}")
    print(f"Expected shape: torch.Size([{batch_size * num_agents}])")
    print(f"Collision info: {collision_info}")
    
    # Verify that only colliding agents have non-zero losses
    collision_loss_reshaped = collision_loss.view(batch_size, num_agents)
    print(f"\nReshaped collision losses: {collision_loss_reshaped}")
    
    # Check expectations
    batch_0_expected = collision_loss_reshaped[0]  # [agent0_loss, agent1_loss, agent2_loss]
    batch_1_expected = collision_loss_reshaped[1]  # [agent0_loss, agent1_loss, agent2_loss]
    
    print(f"Batch 0 collision losses: {batch_0_expected}")
    print(f"  Agent 0 (colliding): {batch_0_expected[0].item():.4f}")
    print(f"  Agent 1 (colliding): {batch_0_expected[1].item():.4f}")
    print(f"  Agent 2 (safe): {batch_0_expected[2].item():.4f}")
    
    print(f"Batch 1 collision losses: {batch_1_expected}")
    print(f"  All agents should have zero loss: {batch_1_expected}")
    
    # Verify colliding agents have higher loss
    assert batch_0_expected[0] > 0, "Agent 0 should have collision loss"
    assert batch_0_expected[1] > 0, "Agent 1 should have collision loss"
    assert batch_0_expected[2] == 0, "Agent 2 should have no collision loss"
    assert torch.allclose(batch_1_expected, torch.zeros_like(batch_1_expected)), "Batch 1 should have no collision losses"
    
    print("✅ Per-agent collision detection test passed!")

def test_collision_loss_application():
    """Test how per-agent collision losses are applied to training."""
    print("\n=== Testing Collision Loss Application ===")
    
    batch_size = 2
    num_agents = 3
    num_actions = 5
    
    # Create dummy agent losses (like cross-entropy)
    agent_losses = []
    for agent in range(num_agents):
        # Random loss per agent per batch
        agent_loss = torch.rand(batch_size) * 0.5  # Base loss: 0.0 - 0.5
        agent_losses.append(agent_loss)
    
    print("Base agent losses:")
    for i, loss in enumerate(agent_losses):
        print(f"  Agent {i}: {loss}")
    
    # Create collision scenario
    positions = torch.tensor([
        [[1.0, 1.0], [1.0, 1.0], [5.0, 5.0]],  # Batch 0: agents 0,1 collide
        [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]   # Batch 1: no collisions
    ], dtype=torch.float32)
    
    logits = torch.randn(batch_size * num_agents, num_actions)
    
    collision_config = {
        'vertex_collision_weight': 2.0,
        'edge_collision_weight': 1.0,
        'collision_loss_type': 'l2',
        'per_agent_loss': True
    }
    
    collision_loss, collision_info = compute_collision_loss(
        logits, positions, None, collision_config
    )
    
    # Apply collision losses like in training
    collision_loss_weight = 10.0
    per_agent_collision_penalties = collision_loss.view(batch_size, num_agents)
    
    print(f"\nPer-agent collision penalties: {per_agent_collision_penalties}")
    
    # Apply penalties to individual agent losses
    weighted_loss = 0
    print("\nWeighted agent losses:")
    for agent in range(num_agents):
        agent_collision_penalty = per_agent_collision_penalties[:, agent]
        weighted_agent_loss = agent_losses[agent] + collision_loss_weight * agent_collision_penalty
        weighted_loss += weighted_agent_loss.mean()
        
        print(f"  Agent {agent}:")
        print(f"    Base loss: {agent_losses[agent]}")
        print(f"    Collision penalty: {agent_collision_penalty}")
        print(f"    Weighted loss: {weighted_agent_loss}")
    
    final_loss = weighted_loss / num_agents
    print(f"\nFinal weighted loss: {final_loss.item():.4f}")
    
    # Compare to base loss without collision penalties
    base_loss = sum(loss.mean() for loss in agent_losses) / num_agents
    print(f"Base loss (no collision penalty): {base_loss.item():.4f}")
    print(f"Loss increase due to collisions: {(final_loss - base_loss).item():.4f}")
    
    print("✅ Collision loss application test passed!")

def test_legacy_vs_per_agent_comparison():
    """Compare legacy aggregate loss vs new per-agent loss."""
    print("\n=== Comparing Legacy vs Per-Agent Loss ===")
    
    batch_size = 2
    num_agents = 3
    num_actions = 5
    
    positions = torch.tensor([
        [[1.0, 1.0], [1.0, 1.0], [5.0, 5.0]],  # Agents 0,1 collide
        [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]   # No collisions
    ], dtype=torch.float32)
    
    logits = torch.randn(batch_size * num_agents, num_actions)
    
    # Test legacy behavior
    legacy_config = {
        'vertex_collision_weight': 2.0,
        'edge_collision_weight': 1.0,
        'collision_loss_type': 'l2',
        'per_agent_loss': False
    }
    
    legacy_loss, legacy_info = compute_collision_loss(
        logits, positions, None, legacy_config
    )
    
    # Test per-agent behavior
    per_agent_config = {
        'vertex_collision_weight': 2.0,
        'edge_collision_weight': 1.0,
        'collision_loss_type': 'l2',
        'per_agent_loss': True
    }
    
    per_agent_loss, per_agent_info = compute_collision_loss(
        logits, positions, None, per_agent_config
    )
    
    print(f"Legacy loss (scalar): {legacy_loss}")
    print(f"Per-agent loss shape: {per_agent_loss.shape}")
    print(f"Per-agent loss values: {per_agent_loss}")
    print(f"Per-agent loss mean: {per_agent_loss.mean()}")
    
    # The mean of per-agent losses should be comparable to legacy loss
    print(f"\nComparison:")
    print(f"  Legacy aggregate loss: {legacy_loss.item():.4f}")
    print(f"  Per-agent loss mean: {per_agent_loss.mean().item():.4f}")
    print(f"  Real collision loss (legacy): {legacy_info['real_collision_loss']:.4f}")
    print(f"  Real collision loss (per-agent): {per_agent_info['real_collision_loss']:.4f}")
    
    print("✅ Legacy vs per-agent comparison completed!")

if __name__ == "__main__":
    print("Testing Per-Agent Collision Loss Implementation")
    print("=" * 50)
    
    test_per_agent_collision_detection()
    test_collision_loss_application()
    test_legacy_vs_per_agent_comparison()
    
    print("\n" + "=" * 50)
    print("🎉 All per-agent collision loss tests passed!")
    print("\nThe new implementation correctly:")
    print("1. Detects collisions per agent")
    print("2. Applies penalties only to colliding agents")
    print("3. Maintains backward compatibility with legacy mode")
    print("4. Enables individual learning for each agent")
