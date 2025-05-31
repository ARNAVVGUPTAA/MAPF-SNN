#!/usr/bin/env python3
"""
Performance benchmark for different problem sizes to demonstrate 
the vectorization benefits at scale.
"""

import torch
import time
import numpy as np

def benchmark_performance():
    """Benchmark performance for different problem sizes."""
    
    print("Dynamic GraphSNN Vectorization Performance Benchmark")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test different problem sizes
    test_configs = [
        (4, 10, 64),   # Small: 4 batches, 10 nodes, 64 hidden
        (8, 20, 128),  # Medium: 8 batches, 20 nodes, 128 hidden  
        (16, 50, 256), # Large: 16 batches, 50 nodes, 256 hidden
        (32, 100, 512) # Very Large: 32 batches, 100 nodes, 512 hidden
    ]
    
    def original_implementation(features, num_nodes):
        batch_size, _, hidden_dim = features.shape
        edge_features_orig = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    pair_features = torch.cat([features[:, i, :], features[:, j, :]], dim=-1)
                    edge_features_orig.append(pair_features)
        
        if edge_features_orig:
            edge_features_orig = torch.stack(edge_features_orig, dim=1)
        else:
            edge_features_orig = torch.empty(batch_size, 0, hidden_dim * 2, device=device)
        
        return edge_features_orig
    
    def vectorized_implementation(features, num_nodes):
        # Create all pairwise combinations using broadcasting
        node_i = features.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        node_j = features.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        
        # Concatenate all pairs
        all_edge_features = torch.cat([node_i, node_j], dim=-1)
        
        # Mask out self-connections
        mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device)
        edge_features_vec = all_edge_features[:, mask, :]
        
        return edge_features_vec
    
    print(f"{'Config':<20} {'Original (ms)':<15} {'Vectorized (ms)':<17} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, num_nodes, hidden_dim in test_configs:
        # Create test data
        combined_features = torch.randn(batch_size, num_nodes, hidden_dim, device=device)
        
        # Warm up GPU if using CUDA
        if device.type == 'cuda':
            _ = original_implementation(combined_features, num_nodes)
            _ = vectorized_implementation(combined_features, num_nodes)
            torch.cuda.synchronize()
        
        # Benchmark original implementation
        start_time = time.time()
        for _ in range(10):  # Average over multiple runs
            result_orig = original_implementation(combined_features, num_nodes)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        original_time = (time.time() - start_time) / 10 * 1000  # Convert to milliseconds
        
        # Benchmark vectorized implementation
        start_time = time.time()
        for _ in range(10):  # Average over multiple runs
            result_vec = vectorized_implementation(combined_features, num_nodes)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        vectorized_time = (time.time() - start_time) / 10 * 1000  # Convert to milliseconds
        
        # Calculate speedup
        speedup = original_time / vectorized_time if vectorized_time > 0 else 0
        
        # Format configuration string
        config_str = f"B{batch_size}_N{num_nodes}_H{hidden_dim}"
        
        print(f"{config_str:<20} {original_time:<15.3f} {vectorized_time:<17.3f} {speedup:<10.2f}x")
        
        # Verify correctness for each config
        are_close = torch.allclose(result_orig, result_vec, rtol=1e-5, atol=1e-6)
        if not are_close:
            print(f"  ⚠️  Warning: Results differ for {config_str}")
    
    print(f"\nNote: Performance benefits are more significant with larger problem sizes")
    print(f"and when running on GPU with higher parallelization capabilities.")

if __name__ == "__main__":
    benchmark_performance()
