#!/usr/bin/env python3
"""
Debug script that mimics exactly what happens during training to find the collision discrepancy.
"""

import torch
import numpy as np
from collision_utils import detect_collisions, compute_collision_loss, reconstruct_positions_from_trajectories, load_initial_positions_from_dataset
from data_loader import SNNDataLoader
from config import config
import yaml
import os

def debug_training_collisions():
    """Debug collisions exactly as they happen during training"""
    
    print("=== DEBUGGING TRAINING COLLISION LOGIC ===")
    
    # Use exact config from training
    data_loader = SNNDataLoader(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use EXACT collision config from training
    collision_config = {
        'vertex_collision_weight': config.get('vertex_collision_weight', 3.0),
        'edge_collision_weight': config.get('edge_collision_weight', 3.0),
        'collision_loss_type': config.get('collision_loss_type', 'exponential'),
        'use_future_collision_penalty': config.get('use_future_collision_penalty', True),
        'future_collision_steps': config.get('future_collision_steps', 2),
        'future_collision_weight': config.get('future_collision_weight', 1.4),
        'future_step_decay': config.get('future_step_decay', 0.7),
        'separate_collision_types': config.get('separate_collision_types', False)
    }
    
    print(f"Collision config: {collision_config}")
    
    total_real_collisions_across_batches = 0
    batches_processed = 0
    
    for i, (states, trajectories, _, case_indices) in enumerate(data_loader.train_loader):
        if i >= 3:  # Check first 3 batches
            break
            
        print(f"\n--- TRAINING BATCH {i} ---")
        
        states = states.to(device)
        trajectories = trajectories.to(device)
        
        batch_size = states.shape[0]
        T = states.shape[1]
        num_agents = trajectories.shape[2]
        
        print(f"Batch size: {batch_size}, Time steps: {T}, Agents: {num_agents}")
        
        # Load initial positions EXACTLY as in training
        try:
            batch_start_idx = i * batch_size
            case_indices_list = list(range(batch_start_idx, batch_start_idx + batch_size))
            
            # Cycle case indices to stay within dataset bounds (0-999)
            max_cases = 1000  # Assuming 1000 cases in dataset
            case_indices_list = [idx % max_cases for idx in case_indices_list]
            
            initial_positions = load_initial_positions_from_dataset(
                config['train']['root_dir'], case_indices_list, mode='train'
            )
            initial_positions = initial_positions.to(device)
            
            # Handle batch size mismatch (as in training)
            if initial_positions.shape[0] > batch_size:
                initial_positions = initial_positions[:batch_size]
            elif initial_positions.shape[0] < batch_size:
                print(f"WARNING: Batch size mismatch! Expected {batch_size}, got {initial_positions.shape[0]}")
                # This would trigger fallback in training
                
        except Exception as e:
            print(f"Error loading initial positions: {e}")
            continue
        
        # Simulate the EXACT training loop collision detection
        batch_real_collisions = 0
        prev_positions = None
        
        for t in range(T):
            print(f"  Time step {t}:")
            
            # Reconstruct positions EXACTLY as in training
            current_positions = reconstruct_positions_from_trajectories(
                initial_positions, trajectories, current_time=t, 
                board_size=config['board_size'][0]
            )
            
            # Create dummy logits (since we don't have the model here)
            dummy_logits = torch.randn(batch_size * num_agents, 5, device=device)
            
            # Call collision loss function EXACTLY as in training
            collision_loss, collision_info = compute_collision_loss(
                dummy_logits, current_positions, prev_positions, collision_config
            )
            
            step_real_collisions = collision_info['total_real_collisions']
            batch_real_collisions += step_real_collisions
            
            print(f"    Step real collisions: {step_real_collisions}")
            print(f"    Vertex collisions: {collision_info['vertex_collisions']}")
            print(f"    Edge collisions: {collision_info['edge_collisions']}")
            print(f"    Real collision rate: {collision_info['real_collision_rate']:.4f}")
            
            # Check if we can see the actual positions causing collisions
            if step_real_collisions > 0:
                print(f"    COLLISION DETECTED! Let's investigate:")
                
                # Manual collision detection to see what's happening
                manual_collisions = detect_collisions(current_positions, prev_positions)
                vertex_collisions = manual_collisions['vertex_collisions']
                edge_collisions = manual_collisions['edge_collisions']
                
                print(f"    Manual vertex collision count: {torch.sum(vertex_collisions > 0).item()}")
                print(f"    Manual edge collision count: {torch.sum(edge_collisions > 0).item()}")
                
                # Find which agents are colliding
                for b in range(batch_size):
                    for agent in range(num_agents):
                        if vertex_collisions[b, agent] > 0:
                            pos = current_positions[b, agent]
                            print(f"      VERTEX: Batch {b}, Agent {agent} at ({pos[0]:.1f}, {pos[1]:.1f})")
                            
                            # Find other agent at same position
                            for other_agent in range(num_agents):
                                if other_agent != agent:
                                    other_pos = current_positions[b, other_agent]
                                    if torch.allclose(pos, other_pos, atol=1e-6):
                                        print(f"               Collides with Agent {other_agent} at ({other_pos[0]:.1f}, {other_pos[1]:.1f})")
                        
                        if edge_collisions[b, agent] > 0 and prev_positions is not None:
                            pos = current_positions[b, agent]
                            prev_pos = prev_positions[b, agent]
                            print(f"      EDGE: Batch {b}, Agent {agent} moved ({prev_pos[0]:.1f}, {prev_pos[1]:.1f}) -> ({pos[0]:.1f}, {pos[1]:.1f})")
            
            prev_positions = current_positions.clone()
        
        print(f"  Batch {i} total real collisions: {batch_real_collisions}")
        total_real_collisions_across_batches += batch_real_collisions
        batches_processed += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Total real collisions across {batches_processed} batches: {total_real_collisions_across_batches}")
    print(f"Average real collisions per batch: {total_real_collisions_across_batches / batches_processed:.2f}")
    
    # Compare with what training reports
    print(f"\nTraining reported real collisions: 126, 137, 88")
    print(f"Our debug found: {total_real_collisions_across_batches}")
    
    if total_real_collisions_across_batches == 0:
        print("\n🤔 MYSTERY: Training reports collisions but debug finds none!")
        print("Possible causes:")
        print("1. Different collision config parameters")
        print("2. Different data batches")
        print("3. Bug in collision counting logic")
        print("4. Model predictions causing collisions (we used dummy logits)")
    else:
        print(f"\n✅ Found the collisions! They're real.")

if __name__ == "__main__":
    debug_training_collisions()
