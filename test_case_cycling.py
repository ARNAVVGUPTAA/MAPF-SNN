#!/usr/bin/env python3

"""
Test the fixed case index cycling logic
"""

def test_case_index_cycling():
    """Test the case index cycling logic to ensure no invalid indices"""
    dataset_size = 1000
    batch_size = 32
    
    print("Testing case index cycling logic:")
    print(f"Dataset size: {dataset_size} (cases 0-999)")
    print(f"Batch size: {batch_size}")
    print()
    
    # Test several batches around the boundary
    test_batches = [0, 1, 30, 31, 32, 33, 62, 63, 64]
    
    for i in test_batches:
        batch_start_idx = i * batch_size
        case_indices = [(batch_start_idx + j) % dataset_size for j in range(batch_size)]
        
        min_case = min(case_indices)
        max_case = max(case_indices)
        
        print(f"Batch {i:2d}: start_idx={batch_start_idx:4d}, cases=[{min_case:3d}..{max_case:3d}], "
              f"valid={all(0 <= idx < dataset_size for idx in case_indices)}")
        
        # Show case cycling around boundary
        if batch_start_idx >= dataset_size - batch_size:
            print(f"         Cases: {case_indices[:5]}...{case_indices[-5:]}")

if __name__ == "__main__":
    test_case_index_cycling()
