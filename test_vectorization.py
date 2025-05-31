#!/usr/bin/env python3
"""
Test script to verify that the vectorized edge feature construction 
produces the same results as the original O(N²) implementation.
"""

import torch
import time
import numpy as np

def test_vectorization_correctness():
    """Test that vectorized implementation produces same results as original."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test parameters
    batch_size = 4
    num_nodes = 8
    hidden_dim = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample input data
    combined_features = torch.randn(batch_size, num_nodes, hidden_dim, device=device)
    
    print(f"Testing vectorization correctness...")
    print(f"Batch size: {batch_size}, Nodes: {num_nodes}, Hidden dim: {hidden_dim}")
    print(f"Device: {device}")
    
    # Original O(N²) implementation
    def original_implementation(features):
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
    
    # Vectorized implementation
    def vectorized_implementation(features):
        # Create all pairwise combinations using broadcasting
        node_i = features.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        node_j = features.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        
        # Concatenate all pairs
        all_edge_features = torch.cat([node_i, node_j], dim=-1)
        
        # Mask out self-connections
        mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device)
        edge_features_vec = all_edge_features[:, mask, :]
        
        return edge_features_vec
    
    # Test correctness
    start_time = time.time()
    result_original = original_implementation(combined_features)
    original_time = time.time() - start_time
    
    start_time = time.time()
    result_vectorized = vectorized_implementation(combined_features)
    vectorized_time = time.time() - start_time
    
    # Check if results are equal
    print(f"\nShape comparison:")
    print(f"Original shape: {result_original.shape}")
    print(f"Vectorized shape: {result_vectorized.shape}")
    
    # Check if tensors are close (allowing for floating point precision)
    are_close = torch.allclose(result_original, result_vectorized, rtol=1e-6, atol=1e-8)
    print(f"\nResults are equal: {are_close}")
    
    if are_close:
        print("✅ Vectorization is correct!")
    else:
        print("❌ Vectorization produces different results!")
        # Show max difference
        max_diff = torch.max(torch.abs(result_original - result_vectorized)).item()
        print(f"Maximum difference: {max_diff}")
    
    # Performance comparison
    print(f"\nPerformance comparison:")
    print(f"Original time: {original_time:.6f} seconds")
    print(f"Vectorized time: {vectorized_time:.6f} seconds")
    
    if vectorized_time > 0:
        speedup = original_time / vectorized_time
        print(f"Speedup: {speedup:.2f}x")
    
    return are_close

def test_edge_weight_assignment():
    """Test the edge weight assignment back to matrix format."""
    
    print(f"\n{'='*60}")
    print("Testing edge weight assignment to matrix format...")
    
    batch_size = 2
    num_nodes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample predicted weights for all node pairs (excluding self-connections)
    num_pairs = num_nodes * (num_nodes - 1)
    predicted_weights = torch.randn(batch_size, num_pairs, device=device)
    
    # Original implementation
    def original_assignment(weights):
        edge_weights = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
        idx = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_weights[:, i, j] = weights[:, idx]
                    idx += 1
        return edge_weights
    
    # Vectorized implementation  
    def vectorized_assignment(weights):
        edge_weights = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
        mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device)
        edge_weights[:, mask] = weights
        return edge_weights
    
    # Test both implementations
    result_original = original_assignment(predicted_weights)
    result_vectorized = vectorized_assignment(predicted_weights)
    
    # Check if results are equal
    are_close = torch.allclose(result_original, result_vectorized, rtol=1e-6, atol=1e-8)
    print(f"Edge weight assignment results are equal: {are_close}")
    
    if are_close:
        print("✅ Edge weight assignment is correct!")
    else:
        print("❌ Edge weight assignment produces different results!")
        max_diff = torch.max(torch.abs(result_original - result_vectorized)).item()
        print(f"Maximum difference: {max_diff}")
    
    return are_close

if __name__ == "__main__":
    print("Testing Dynamic GraphSNN Vectorization")
    print("="*60)
    
    # Test edge feature construction
    test1_passed = test_vectorization_correctness()
    
    # Test edge weight assignment
    test2_passed = test_edge_weight_assignment()
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS:")
    print(f"Edge feature construction test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Edge weight assignment test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Vectorization is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
