"""
Debug utilities for MAPF-GNN training system.
Contains essential debugging functions for monitoring training progress,
collision detection, and agent behavior analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Global debug mode state
_debug_enabled = False

def set_debug_mode(enabled: bool):
    """Set global debug mode state."""
    global _debug_enabled
    _debug_enabled = enabled

def debug_print(message: str, force: bool = False):
    """Simple debug print function that respects global debug mode."""
    global _debug_enabled
    if _debug_enabled or force:
        print(f"[DEBUG] {message}")

def print_tensor_info(tensor: torch.Tensor, name: str = "tensor", detailed: bool = False):
    """Print comprehensive information about a tensor."""
    if tensor is None:
        print(f"{name}: None")
        return
    
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Requires grad: {tensor.requires_grad}")
    
    if detailed and tensor.numel() > 0:
        print(f"  Min: {tensor.min().item():.4f}")
        print(f"  Max: {tensor.max().item():.4f}")
        print(f"  Mean: {tensor.mean().item():.4f}")
        print(f"  Std: {tensor.std().item():.4f}")
        print(f"  NaN count: {torch.isnan(tensor).sum().item()}")
        print(f"  Inf count: {torch.isinf(tensor).sum().item()}")

def check_gradient_flow(model: torch.nn.Module, print_grads: bool = False):
    """Check if gradients are flowing properly through the model."""
    total_params = 0
    trainable_params = 0
    params_with_grad = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if param.grad is not None:
                params_with_grad += param.numel()
                if print_grads:
                    grad_norm = param.grad.norm().item()
                    print(f"{name}: grad_norm = {grad_norm:.6f}")
    
    print(f"Gradient Flow Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameters with gradients: {params_with_grad:,}")
    print(f"  Gradient coverage: {params_with_grad/trainable_params*100:.1f}%")

def log_training_step(epoch: int, step: int, loss: float, metrics: Dict[str, float], 
                     log_freq: int = 10):
    """Log training step information."""
    if step % log_freq == 0:
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Epoch {epoch}, Step {step} | Loss: {loss:.4f} | {metrics_str}")

def analyze_agent_positions(positions: torch.Tensor, grid_size: int = 20) -> Dict[str, Any]:
    """Analyze agent position distribution and statistics."""
    if positions.dim() == 3:  # [batch, agents, 2]
        positions = positions.reshape(-1, 2)
    
    x_coords = positions[:, 0].cpu().numpy()
    y_coords = positions[:, 1].cpu().numpy()
    
    analysis = {
        'num_positions': len(positions),
        'x_range': (x_coords.min(), x_coords.max()),
        'y_range': (y_coords.min(), y_coords.max()),
        'x_mean': x_coords.mean(),
        'y_mean': y_coords.mean(),
        'x_std': x_coords.std(),
        'y_std': y_coords.std(),
        'out_of_bounds': ((x_coords < 0) | (x_coords >= grid_size) | 
                         (y_coords < 0) | (y_coords >= grid_size)).sum()
    }
    
    return analysis

def detect_position_anomalies(positions: torch.Tensor, grid_size: int = 20) -> List[str]:
    """Detect anomalies in agent positions."""
    anomalies = []
    
    if positions.dim() == 3:  # [batch, agents, 2]
        batch_size, num_agents = positions.shape[:2]
        positions_flat = positions.reshape(-1, 2)
    else:
        positions_flat = positions
    
    # Check for out-of-bounds positions
    x_coords = positions_flat[:, 0]
    y_coords = positions_flat[:, 1]
    oob_mask = (x_coords < 0) | (x_coords >= grid_size) | (y_coords < 0) | (y_coords >= grid_size)
    if oob_mask.any():
        anomalies.append(f"Out-of-bounds positions: {oob_mask.sum().item()}")
    
    # Check for duplicate positions (collisions)
    if positions.dim() == 3:
        for b in range(batch_size):
            batch_positions = positions[b]  # [agents, 2]
            unique_positions = torch.unique(batch_positions, dim=0)
            if len(unique_positions) < num_agents:
                anomalies.append(f"Batch {b}: {num_agents - len(unique_positions)} collisions detected")
    
    # Check for NaN or Inf values
    if torch.isnan(positions_flat).any():
        anomalies.append("NaN values in positions")
    if torch.isinf(positions_flat).any():
        anomalies.append("Inf values in positions")
    
    return anomalies

def monitor_loss_components(loss_dict: Dict[str, float], step: int, 
                          window_size: int = 100) -> Dict[str, float]:
    """Monitor and analyze loss components over time."""
    if not hasattr(monitor_loss_components, 'history'):
        monitor_loss_components.history = {}
    
    # Update history
    for key, value in loss_dict.items():
        if key not in monitor_loss_components.history:
            monitor_loss_components.history[key] = []
        monitor_loss_components.history[key].append(value)
        
        # Keep only recent history
        if len(monitor_loss_components.history[key]) > window_size:
            monitor_loss_components.history[key] = monitor_loss_components.history[key][-window_size:]
    
    # Compute moving averages
    moving_averages = {}
    for key, values in monitor_loss_components.history.items():
        moving_averages[f"{key}_avg"] = np.mean(values)
        if len(values) > 1:
            moving_averages[f"{key}_std"] = np.std(values)
    
    return moving_averages

def validate_model_output(output: torch.Tensor, expected_shape: Tuple[int, ...], 
                         name: str = "model_output") -> bool:
    """Validate model output shape and values."""
    issues = []
    
    # Check shape
    if output.shape != expected_shape:
        issues.append(f"Shape mismatch: got {output.shape}, expected {expected_shape}")
    
    # Check for NaN/Inf
    if torch.isnan(output).any():
        issues.append("Contains NaN values")
    if torch.isinf(output).any():
        issues.append("Contains Inf values")
    
    # Check value ranges for logits/probabilities
    if "logit" in name.lower() or "prob" in name.lower():
        if output.min() < -50 or output.max() > 50:
            issues.append(f"Extreme values: min={output.min().item():.2f}, max={output.max().item():.2f}")
    
    if issues:
        logger.warning(f"{name} validation issues: {'; '.join(issues)}")
        return False
    
    return True

def create_position_heatmap(positions: torch.Tensor, grid_size: int = 20, 
                          title: str = "Agent Position Heatmap"):
    """Create a heatmap showing agent position density."""
    if positions.dim() == 3:
        positions = positions.reshape(-1, 2)
    
    x_coords = positions[:, 0].cpu().numpy().astype(int)
    y_coords = positions[:, 1].cpu().numpy().astype(int)
    
    # Filter valid coordinates
    valid_mask = ((x_coords >= 0) & (x_coords < grid_size) & 
                  (y_coords >= 0) & (y_coords < grid_size))
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    
    # Create heatmap
    heatmap = np.zeros((grid_size, grid_size))
    for x, y in zip(x_coords, y_coords):
        heatmap[y, x] += 1
    
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Agent Count')
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()
    
    return heatmap

def check_collision_consistency(positions: torch.Tensor, collision_mask: torch.Tensor) -> Dict[str, Any]:
    """Check consistency between positions and collision detection."""
    if positions.dim() != 3 or collision_mask.dim() != 2:
        return {"error": "Invalid tensor dimensions"}
    
    batch_size, num_agents = positions.shape[:2]
    results = {
        "total_batches": batch_size,
        "agents_per_batch": num_agents,
        "collision_rate": collision_mask.float().mean().item(),
        "inconsistencies": 0
    }
    
    for b in range(batch_size):
        batch_positions = positions[b]  # [agents, 2]
        batch_collisions = collision_mask[b]  # [agents]
        
        # Find actual collisions by checking for duplicate positions
        unique_positions, inverse_indices = torch.unique(batch_positions, dim=0, return_inverse=True)
        actual_collisions = torch.zeros(num_agents, dtype=torch.bool, device=positions.device)
        
        for i in range(len(unique_positions)):
            agents_at_position = (inverse_indices == i).nonzero(as_tuple=True)[0]
            if len(agents_at_position) > 1:
                actual_collisions[agents_at_position] = True
        
        # Check consistency
        if not torch.equal(actual_collisions, batch_collisions):
            results["inconsistencies"] += 1
    
    return results

def summarize_training_metrics(metrics_history: Dict[str, List[float]], 
                             recent_steps: int = 100) -> Dict[str, float]:
    """Summarize training metrics over recent steps."""
    summary = {}
    
    for metric_name, values in metrics_history.items():
        if not values:
            continue
            
        recent_values = values[-recent_steps:] if len(values) > recent_steps else values
        
        summary[f"{metric_name}_recent_mean"] = np.mean(recent_values)
        summary[f"{metric_name}_recent_std"] = np.std(recent_values)
        summary[f"{metric_name}_trend"] = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        
        if len(values) > 1:
            summary[f"{metric_name}_total_trend"] = np.polyfit(range(len(values)), values, 1)[0]
    
    return summary

def log_system_resources():
    """Log current system resource usage."""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        logger.info(f"System Resources - CPU: {cpu_percent:.1f}% | "
                   f"Memory: {memory.percent:.1f}% ({memory.used // (1024**3):.1f}GB / "
                   f"{memory.total // (1024**3):.1f}GB)")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                logger.info(f"GPU {i} Memory - Allocated: {allocated:.1f}GB | Cached: {cached:.1f}GB")
                
    except ImportError:
        logger.warning("psutil not available, skipping system resource logging")
