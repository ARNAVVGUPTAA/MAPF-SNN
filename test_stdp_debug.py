#!/usr/bin/env python3

import torch
import sys
import os

# Add the models directory to the path
sys.path.append('/home/arnav/dev/summer25/MAPF-GNN')

from models.framework_snn import STDPSpatialDangerDetector

def test_stdp_detector():
    """Test the STDP spatial danger detector to debug the issue."""
    
    print("Testing STDP Spatial Danger Detector...")
    
    # Create test configuration
    config = {
        'stdp_feature_channels': 32,
        'stdp_kernel_size': 3,
        'enable_stdp_learning': True,
        'stdp_tau_pre': 20.0,
        'stdp_tau_post': 20.0,
        'stdp_lr_pre': 0.01,
        'stdp_lr_post': 0.01,
        'danger_threshold': 0.5
    }
    
    # Create detector instance
    try:
        detector = STDPSpatialDangerDetector(
            input_channels=3,  # RGB input
            feature_channels=config['stdp_feature_channels'],
            kernel_size=config['stdp_kernel_size'],
            cfg=config  # Pass the entire config dictionary
        )
        print("✓ STDP detector created successfully")
    except Exception as e:
        print(f"✗ Error creating STDP detector: {e}")
        return False
    
    # Create test input
    batch_size = 2
    height, width = 32, 32
    test_input = torch.randn(batch_size, 3, height, width)
    print(f"✓ Created test input: {test_input.shape}")
    
    # Test forward pass
    try:
        detector.train()  # Set to training mode
        output = detector(test_input)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        # Test multiple forward passes to check STDP learning
        for i in range(3):
            output = detector(test_input)
            print(f"✓ Forward pass {i+1} successful, output shape: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_stdp_detector()
    if success:
        print("\n🎉 STDP Spatial Danger Detector test passed!")
    else:
        print("\n❌ STDP Spatial Danger Detector test failed!")
