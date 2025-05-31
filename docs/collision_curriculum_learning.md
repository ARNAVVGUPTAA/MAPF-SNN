# Collision Loss Curriculum Learning

## Overview

This document explains the collision loss curriculum learning implementation for the MAPF-GNN project. The curriculum gradually increases the collision penalty weight over the first 10 epochs of training to improve learning dynamics.

## Problem Statement

In multi-agent pathfinding, collision avoidance is crucial but can create training challenges:

1. **High initial collision penalty**: Can cause premature convergence to overly conservative paths
2. **Exploration vs exploitation**: Agents need to explore initially but avoid collisions in final policies
3. **Learning dynamics**: Sudden high collision penalties can destabilize early training

## Solution: Curriculum Learning

The collision curriculum gradually increases the collision loss weight from 10% to 100% of the final value over 10 epochs:

- **Epoch 0**: 1.5 (10% of 15.0)
- **Epoch 5**: 8.25 (55% of 15.0) 
- **Epoch 9**: 13.65 (91% of 15.0)
- **Epoch 10+**: 15.0 (100% - full weight)

## Configuration

Add these parameters to your config file:

```yaml
# Collision Loss Curriculum Learning
use_collision_curriculum: true        # Enable curriculum learning for collision loss
collision_curriculum_epochs: 10       # Number of epochs for curriculum (gradual increase)
collision_curriculum_start_ratio: 0.1 # Starting ratio of full collision loss weight (10% of final weight)
```

## Implementation Details

### 1. Curriculum Schedule Function

```python
def collision_curriculum_schedule(epoch, final_value, curriculum_epochs=10, start_ratio=0.1):
    """
    Curriculum learning schedule for collision loss weight.
    Gradually increases from start_ratio * final_value to final_value over curriculum_epochs.
    """
    if epoch >= curriculum_epochs:
        return final_value
    
    # Linear increase from start_ratio * final_value to final_value
    start_value = start_ratio * final_value
    progress = epoch / curriculum_epochs
    value = start_value + (final_value - start_value) * progress
    
    return value
```

### 2. Training Integration

The curriculum is integrated into the training loop with priority over cosine annealing:

1. **Curriculum enabled**: Uses curriculum schedule for collision loss, other schedules remain unchanged
2. **Curriculum disabled**: Falls back to cosine annealing or fixed weights
3. **After curriculum period**: Uses full collision loss weight

### 3. Logging

Training output includes curriculum progress:

```
Epoch 5 | Loss: 0.1234 | CollisionWeight: 8.25e+00 | Curriculum: 6/10 | ...
```

## Benefits

1. **Improved exploration**: Low initial collision penalty allows agents to explore diverse paths
2. **Stable convergence**: Gradual increase prevents training instability
3. **Better final performance**: Full collision penalty ensures collision-free final policies
4. **Maintained flexibility**: Other regularizers (spike, future collision) use existing schedules

## Usage

1. **Enable curriculum**: Set `use_collision_curriculum: true` in config
2. **Adjust duration**: Modify `collision_curriculum_epochs` (default: 10)
3. **Control starting point**: Adjust `collision_curriculum_start_ratio` (default: 0.1 = 10%)
4. **Monitor progress**: Check training logs for curriculum information

## Comparison with Cosine Annealing

| Method | Collision Weight Behavior | Use Case |
|--------|---------------------------|----------|
| **Curriculum** | Linear increase 1.5→15.0 over 10 epochs, then constant | Improved exploration early, strict collision avoidance later |
| **Cosine Annealing** | Oscillating pattern based on cosine function | Regularization scheduling throughout training |
| **Fixed** | Constant weight throughout training | Simple baseline approach |

## Testing

Use the test script to visualize the schedule:

```bash
python test_collision_curriculum.py
```

This shows the collision weight progression and helps tune curriculum parameters.

## Compatibility

- **Backward compatible**: Existing configs work unchanged (curriculum disabled by default)
- **Flexible integration**: Works with existing cosine annealing for other parameters
- **Easy tuning**: Simple parameters to adjust curriculum behavior
