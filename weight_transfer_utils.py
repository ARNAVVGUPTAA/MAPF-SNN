#!/usr/bin/env python3
"""
Weight Transfer Utilities
========================

Utilities to verify that pretrained weights are properly transferred to main training.
"""

import torch
import os
from typing import Dict, Any, Optional, Tuple


def verify_weight_transfer(model: torch.nn.Module, 
                         pretrained_checkpoint_path: str,
                         verbose: bool = True) -> bool:
    """
    Verify that model weights match a pretrained checkpoint.
    
    Args:
        model: Current model to check
        pretrained_checkpoint_path: Path to pretrained checkpoint
        verbose: Whether to print detailed verification info
        
    Returns:
        bool: True if weights match, False otherwise
    """
    if not os.path.exists(pretrained_checkpoint_path):
        if verbose:
            print(f"❌ Pretrained checkpoint not found: {pretrained_checkpoint_path}")
        return False
    
    try:
        # Load pretrained checkpoint
        checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu')
        pretrained_state = checkpoint['model_state_dict']
        
        # Get current model state
        current_state = model.state_dict()
        
        # Compare states
        matches = 0
        total_params = 0
        max_diff = 0.0
        
        for name, pretrained_param in pretrained_state.items():
            if name in current_state:
                current_param = current_state[name]
                param_diff = torch.abs(current_param - pretrained_param).max().item()
                max_diff = max(max_diff, param_diff)
                
                if torch.allclose(current_param, pretrained_param, atol=1e-6):
                    matches += 1
                total_params += 1
                
                if verbose and 'global_features' in name:
                    print(f"   🧠 {name}: max_diff={param_diff:.8f}")
        
        success_rate = matches / total_params if total_params > 0 else 0.0
        
        if verbose:
            print(f"🔍 Weight transfer verification:")
            print(f"   📊 Matching parameters: {matches}/{total_params} ({success_rate:.3f})")
            print(f"   📊 Maximum difference: {max_diff:.8f}")
        
        # Consider successful if > 95% of parameters match closely
        return success_rate > 0.95 and max_diff < 1e-5
        
    except Exception as e:
        if verbose:
            print(f"❌ Error verifying weight transfer: {e}")
        return False


def save_weight_snapshot(model: torch.nn.Module, 
                        snapshot_path: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a snapshot of current model weights for comparison.
    
    Args:
        model: Model to snapshot
        snapshot_path: Path to save snapshot
        metadata: Optional metadata to include
    """
    snapshot_data = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {},
        'timestamp': torch.tensor(torch.cuda.Event().query() if torch.cuda.is_available() else 0.0)
    }
    
    torch.save(snapshot_data, snapshot_path)
    print(f"📸 Model weight snapshot saved: {snapshot_path}")


def compare_weight_snapshots(snapshot1_path: str, 
                           snapshot2_path: str,
                           verbose: bool = True) -> Dict[str, float]:
    """
    Compare two weight snapshots and return difference metrics.
    
    Args:
        snapshot1_path: First snapshot path
        snapshot2_path: Second snapshot path
        verbose: Whether to print detailed comparison info
        
    Returns:
        Dict with comparison metrics
    """
    try:
        snap1 = torch.load(snapshot1_path, map_location='cpu')
        snap2 = torch.load(snapshot2_path, map_location='cpu')
        
        state1 = snap1['model_state_dict']
        state2 = snap2['model_state_dict']
        
        total_change = 0.0
        max_change = 0.0
        param_count = 0
        layer_changes = {}
        
        for name, param1 in state1.items():
            if name in state2:
                param2 = state2[name]
                change = torch.abs(param2 - param1).mean().item()
                max_layer_change = torch.abs(param2 - param1).max().item()
                
                total_change += change
                max_change = max(max_change, max_layer_change)
                param_count += 1
                layer_changes[name] = change
                
                if verbose and ('global_features' in name or 'snn' in name):
                    print(f"   🧠 {name}: avg_change={change:.8f}, max_change={max_layer_change:.8f}")
        
        avg_change = total_change / param_count if param_count > 0 else 0.0
        
        metrics = {
            'average_change': avg_change,
            'maximum_change': max_change,
            'total_parameters': param_count,
            'layer_changes': layer_changes
        }
        
        if verbose:
            print(f"📊 Weight comparison metrics:")
            print(f"   📈 Average change: {avg_change:.8f}")
            print(f"   📈 Maximum change: {max_change:.8f}")
            print(f"   📊 Parameters compared: {param_count}")
        
        return metrics
        
    except Exception as e:
        if verbose:
            print(f"❌ Error comparing snapshots: {e}")
        return {}


def analyze_weight_magnitudes(model: torch.nn.Module, 
                            layer_filter: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
    """
    Analyze weight magnitudes in model layers.
    
    Args:
        model: Model to analyze
        layer_filter: Optional string to filter layer names
        
    Returns:
        Dict mapping layer names to (mean_magnitude, std_magnitude) tuples
    """
    results = {}
    
    for name, param in model.named_parameters():
        if layer_filter is None or layer_filter in name:
            if param.requires_grad and param.numel() > 0:
                mean_mag = param.abs().mean().item()
                std_mag = param.std().item()
                results[name] = (mean_mag, std_mag)
    
    return results


def log_weight_statistics(model: torch.nn.Module, 
                         stage: str = "current",
                         layer_filter: str = "global_features") -> None:
    """
    Log detailed weight statistics for debugging.
    
    Args:
        model: Model to analyze
        stage: Stage name for logging (e.g., "pretraining", "main_training")
        layer_filter: Filter for layer names to analyze
    """
    print(f"🔍 Weight statistics at {stage}:")
    
    magnitudes = analyze_weight_magnitudes(model, layer_filter)
    
    for name, (mean_mag, std_mag) in magnitudes.items():
        print(f"   🧠 {name}: mean={mean_mag:.8f}, std={std_mag:.8f}")
        
        # Additional analysis
        param = dict(model.named_parameters())[name]
        zero_count = (param == 0).sum().item()
        total_count = param.numel()
        nonzero_ratio = (total_count - zero_count) / total_count
        
        print(f"      📊 Non-zero ratio: {nonzero_ratio:.4f} ({total_count - zero_count}/{total_count})")
        print(f"      📊 Range: [{param.min().item():.8f}, {param.max().item():.8f}]")
