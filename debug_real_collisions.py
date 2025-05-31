#!/usr/bin/env python3
"""
Debug script to check if the "real collisions" are actually real collisions.
This will help us understand what the collision metric actually represents.
"""

import torch
import numpy as np
from collision_utils import detect_collisions, compute_collision_loss, reconstruct_positions_from_trajectories, load_initial_positions_from_dataset
from data_loader import SNNDataLoader
from config import config
import yaml
import os

def debug_real_collisions():
    """Debug what the 'real collisions' metric actually represents"""
    
    print("=== DEBUGGING REAL COLLISIONS ===")
    
    # Create a debug config with smaller batch size
    debug_config = config.copy()
    debug_config['train']['batch_size'] = 4  # Small batch for debugging
    
    # Load a small sample of data
    data_loader = SNNDataLoader(debug_config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i, (states, trajectories, _, case_indices) in enumerate(data_loader.train_loader):
        if i > 0:  # Only check first batch
            break
            
        print(f"\n--- BATCH {i} ---")
        print(f"States shape: {states.shape}")
        print(f"Trajectories shape: {trajectories.shape}")
        print(f"Case indices: {case_indices}")
        
        states = states.to(device)
        trajectories = trajectories.to(device)
        
        batch_size = states.shape[0]
        T = states.shape[1]  # Number of time steps
        num_agents = trajectories.shape[2]
        
        print(f"Batch size: {batch_size}, Time steps: {T}, Agents: {num_agents}")
        
        # Load initial positions
        try:
            initial_positions = load_initial_positions_from_dataset(
                config['train']['root_dir'], case_indices.tolist(), mode='train'
            )
            initial_positions = initial_positions.to(device)
            print(f"Initial positions shape: {initial_positions.shape}")
            print(f"Initial positions:\n{initial_positions}")
            
        except Exception as e:
            print(f"Error loading initial positions: {e}")
            continue
        
        # Check collisions at each time step
        total_real_collisions = 0
        prev_positions = None
        
        for t in range(T):
            print(f"\n  TIME STEP {t}:")
            
            # Reconstruct current positions
            current_positions = reconstruct_positions_from_trajectories(
                initial_positions, trajectories, current_time=t, 
                board_size=28
            )
            
            print(f"    Current positions:\n{current_positions}")
            
            # Detect collisions manually
            collisions = detect_collisions(current_positions, prev_positions)
            
            vertex_collisions = collisions['vertex_collisions']
            edge_collisions = collisions['edge_collisions']
            
            # Count collisions
            vertex_count = torch.sum(vertex_collisions > 0).item()
            edge_count = torch.sum(edge_collisions > 0).item()
            total_collisions_this_step = vertex_count + edge_count
            
            print(f"    Vertex collisions: {vertex_count}")
            print(f"    Edge collisions: {edge_count}")
            print(f"    Total collisions this step: {total_collisions_this_step}")
            
            if vertex_count > 0:
                print(f"    Vertex collision matrix:\n{vertex_collisions}")
                # Find which agents are colliding
                for b in range(batch_size):
                    for agent in range(num_agents):
                        if vertex_collisions[b, agent] > 0:
                            pos = current_positions[b, agent]
                            print(f"      Batch {b}, Agent {agent} at position ({pos[0]:.1f}, {pos[1]:.1f}) has vertex collision")
            
            if edge_count > 0:
                print(f"    Edge collision matrix:\n{edge_collisions}")
                # Find which agents are colliding
                for b in range(batch_size):
                    for agent in range(num_agents):
                        if edge_collisions[b, agent] > 0:
                            pos = current_positions[b, agent]
                            prev_pos = prev_positions[b, agent] if prev_positions is not None else "N/A"
                            print(f"      Batch {b}, Agent {agent} at position ({pos[0]:.1f}, {pos[1]:.1f}) has edge collision (prev: {prev_pos})")
            
            total_real_collisions += total_collisions_this_step
            prev_positions = current_positions.clone()
        
        print(f"\n  BATCH SUMMARY:")
        print(f"    Total real collisions across all time steps: {total_real_collisions}")
        print(f"    Average collisions per time step: {total_real_collisions / T:.2f}")
        
        # Now test what the collision loss function reports
        print(f"\n  COLLISION LOSS FUNCTION TEST:")
        
        # Use dummy logits for collision loss computation
        dummy_logits = torch.randn(batch_size * num_agents, 5, device=device)  # 5 actions
        
        collision_config = {
            'vertex_collision_weight': 3.0,
            'edge_collision_weight': 3.0,
            'collision_loss_type': 'exponential',
            'use_future_collision_penalty': True,
            'future_collision_steps': 2,
            'future_collision_weight': 1.4,
            'future_step_decay': 0.7,
            'separate_collision_types': False
        }
        
        # Test collision loss at final time step
        final_positions = reconstruct_positions_from_trajectories(
            initial_positions, trajectories, current_time=T-1, board_size=28
        )
        prev_final_positions = reconstruct_positions_from_trajectories(
            initial_positions, trajectories, current_time=T-2, board_size=28
        ) if T > 1 else None
        
        collision_loss, collision_info = compute_collision_loss(
            dummy_logits, final_positions, prev_final_positions, collision_config
        )
        
        print(f"    Collision loss function reports:")
        print(f"      Total real collisions: {collision_info['total_real_collisions']}")
        print(f"      Vertex collisions: {collision_info['vertex_collisions']}")
        print(f"      Edge collisions: {collision_info['edge_collisions']}")
        print(f"      Real collision rate: {collision_info['real_collision_rate']:.4f}")
        print(f"      Average real collision penalty: {collision_info['avg_real_collision_penalty']:.4f}")
        
        break  # Only process first batch
    
    print("\n=== ANALYSIS ===")
    print("If 'real collisions' numbers are high but you don't see actual position overlaps,")
    print("then the collision detection might be too sensitive or there's a bug in the logic.")
    print("Check the collision detection radius and vertex/edge collision definitions.")

if __name__ == "__main__":
    debug_real_collisions()
