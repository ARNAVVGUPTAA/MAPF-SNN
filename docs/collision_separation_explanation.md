# Collision Loss Separation: Real vs Future Collisions

## Problem
Previously, the collision loss system had a logical issue where agents could be penalized twice:
1. **Real collisions**: When agents actually collide (same position or swap positions)
2. **Future collisions**: When agents might collide in future steps based on their current action tendencies

This double penalization doesn't make logical sense - if an agent is already colliding, why should we also penalize it for potential future collisions?

## Solution
I've implemented a new system that separates real and future collision handling:

### Key Changes

1. **Separate Tracking**: The training loop now tracks real collisions and future collision penalties separately:
   - `real_collision_reg`: Accumulates real collision losses
   - `future_collision_reg`: Accumulates future collision losses
   - `total_real_collisions`: Count of actual collisions
   - `total_future_collisions`: Count of timesteps where future penalty was applied

2. **Logical Collision Loss**: The collision loss function now uses an either/or approach:
   - **If real collisions exist**: Only apply real collision penalty
   - **If no real collisions**: Apply future collision penalty for collision avoidance
   - This prevents double penalization

3. **Collision Mask**: Added a mask system that only predicts future collisions for agents that are NOT currently colliding

### Configuration
- **`separate_collision_types: true`** (default): Use the new logical approach
- **`separate_collision_types: false`**: Use the old approach (for comparison)

### Benefits
1. **Logical consistency**: No double penalization
2. **Better training signal**: Clear distinction between actual problems (real collisions) and preventive measures (future collision avoidance)
3. **Debugging visibility**: Separate metrics help understand what's happening during training

### Training Output
The training now shows separate metrics:
```
Epoch X | Loss: 0.1234 | Spikes: 5000 | SpikeReg: 0.05 | RealCollisionReg: 0.02 | FutureCollisionReg: 0.01 | RealCollisions: 3 | FutureCollisionSteps: 8
```

Where:
- `RealCollisionReg`: Average real collision loss per timestep
- `FutureCollisionReg`: Average future collision loss per timestep  
- `RealCollisions`: Total number of actual collisions across all agents/batches
- `FutureCollisionSteps`: Number of timesteps where future collision penalty was applied

This gives you clear visibility into whether your model is actually colliding vs. just being cautious about potential future collisions.
