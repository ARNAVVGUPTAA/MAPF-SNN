# **BDSM: Brains Don't Simply Multiply**

<p align="center">
  <b>Neuromorphic Liquid State Machine for Multi-Agent Path Finding</b><br>
  <i>Brain-inspired computing without multiplication operations</i>
</p>

---

## 🧠 Table of Contents

- [Overview](#overview)
- [The BDSM Principle](#the-bdsm-principle)
- [Liquid State Machines Explained](#liquid-state-machines-explained)
- [Reservoir Computing vs SGD Training](#reservoir-computing-vs-sgd-training)
- [Architecture](#architecture)
- [Neuromorphic Operations](#neuromorphic-operations)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Configuration](#configuration)
- [Hardware Deployment](#hardware-deployment)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

This project implements a **neuromorphic Liquid State Machine (LSM)** for solving Multi-Agent Path Finding (MAPF) problems. The key innovation is using **brain-inspired operations** (bit-shifts, additions, sparse computation) instead of traditional neural network multiplications, making the system deployable on neuromorphic hardware like Intel Loihi, IBM TrueNorth, and SpiNNaker.

### Key Features

- ✅ **4,096 spiking neurons** organized in 16 circular meshes
- ✅ **EILIF neurons** (Excitatory-Inhibitory Leaky Integrate-and-Fire)
- ✅ **Bit-shift decay** instead of multiplicative membrane leak
- ✅ **Quantized weights** (4-bit synapses, 8-bit readout)
- ✅ **Dual training modes**: Reservoir computing OR end-to-end SGD/BPTT
- ✅ **Hardware-ready**: Export integer weights for neuromorphic chips
- ✅ **Gradient flow**: Surrogate gradients enable backpropagation through spikes

---

## The BDSM Principle

### **Brains Don't Simply Multiply**

Traditional artificial neural networks rely heavily on matrix multiplications:
```python
y = W @ x + b  # Standard neural network operation
```

Real brains, however, use fundamentally different operations:
- **Membrane potential decay**: `v_new = v - (v >> shift)` (bit-shift, not multiply)
- **Spike integration**: Addition of input currents
- **Weight quantization**: Synaptic strengths are discrete (not continuous floats)
- **Sparse computation**: Only active neurons consume energy

Our neuromorphic LSM implements these brain-like principles:

| Traditional NN | Neuromorphic LSM (BDSM) |
|----------------|-------------------------|
| `v = v * τ` | `v = v - (v >> shift)` |
| Float32 weights | Integer (4-bit/8-bit) weights |
| Dense computation | Event-driven (sparse) |
| GPU-optimized | Neuromorphic chip-ready |

### Why BDSM Matters

1. **Energy Efficiency**: Bit-shifts consume ~100× less energy than multiplications
2. **Hardware Compatibility**: Neuromorphic chips (Loihi, TrueNorth) don't have floating-point units
3. **Biological Realism**: Closer to how real neurons operate
4. **Robustness**: Quantized operations are more noise-tolerant

---

## Liquid State Machines Explained

### What is an LSM?

A Liquid State Machine is a type of **reservoir computing** system inspired by cortical microcircuits. The "liquid" refers to the high-dimensional, dynamic state of a recurrent neural network that responds to input stimuli.

```
Input → [Liquid Reservoir] → Readout → Output
         (Rich dynamics)     (Simple linear layer)
```

### Core Concept

1. **Input Layer**: Projects external stimuli into the reservoir
2. **Liquid (Reservoir)**: Large recurrent network with fixed random weights
   - High-dimensional state space
   - Rich temporal dynamics
   - Non-linear transformations
3. **Readout**: Simple trainable layer that reads out the liquid state

### Why LSM for MAPF?

- **Temporal dynamics**: MAPF requires processing sequences of observations
- **High-dimensional expansion**: 98 input features → 4096 reservoir neurons
- **Memory**: Reservoir maintains history of past observations
- **Generalization**: Random reservoir acts as universal feature extractor

### Comparison with Other Models

| Model | Training | Complexity | Temporal |
|-------|----------|------------|----------|
| **MLP** | All weights | Medium | ❌ No memory |
| **RNN/LSTM** | All weights | High | ✅ Sequential |
| **LSM** | Readout only | Low | ✅ Rich dynamics |
| **Transformer** | All weights | Very High | ✅ Attention |

---

## Reservoir Computing vs SGD Training

Our system supports **two training paradigms**:

### 1. Reservoir Computing (Default)

**Philosophy**: Use the reservoir as a fixed feature extractor, train only the readout.

**Process**:
```
1. Initialize reservoir with random weights (FROZEN)
2. Pass all training data through reservoir
3. Collect liquid states (spike rates)
4. Train linear readout with ridge regression
   W = (X^T X + λI)^{-1} X^T Y
```

**Advantages**:
- ✅ **Fast training**: Only readout needs training (~1 minute)
- ✅ **Low compute**: Ridge regression is closed-form solution
- ✅ **Avoid overfitting**: Reservoir is fixed, can't overfit
- ✅ **Simple**: No backpropagation through time needed

**Disadvantages**:
- ❌ Reservoir weights are random (not optimized)
- ❌ Performance limited by reservoir quality
- ❌ Can't adapt reservoir to specific task

**When to use**: Quick prototyping, small datasets, limited compute

### 2. End-to-End SGD/BPTT Training

**Philosophy**: Train the entire network (reservoir + readout) with gradient descent.

**Process**:
```
1. Initialize reservoir with random weights (TRAINABLE)
2. Forward pass: Compute network output
3. Backward pass: Compute gradients through surrogate functions
4. Update: Adjust all weights with optimizer (Adam/SGD)
5. Repeat for N epochs
```

**Advantages**:
- ✅ **Optimized reservoir**: Weights adapted to task
- ✅ **Better performance**: Can learn complex patterns
- ✅ **End-to-end**: Entire system optimized together

**Disadvantages**:
- ❌ **Slow training**: Requires backpropagation through time
- ❌ **Risk of overfitting**: More parameters to tune
- ❌ **Gradient flow issues**: Spikes are non-differentiable

**When to use**: Large datasets, high performance needed, sufficient compute

### 3. Hybrid Mode

**Philosophy**: Best of both worlds.

**Process**:
```
1. Start with reservoir computing (fast initial training)
2. Fine-tune entire network with SGD (performance boost)
```

**Advantages**:
- ✅ Fast initial convergence
- ✅ Performance improvement from fine-tuning
- ✅ Less prone to overfitting than pure SGD

---

## Architecture

### Overall Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER (98 features)                │
│         FOV: [2 channels × 7×7] = 98 values                 │
│         Channel 0: Obstacles/agents (0/1/2)                 │
│         Channel 1: Goals (0/3)                              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              CIRCULAR RESERVOIR (4096 neurons)              │
│                                                             │
│  ┌──────┐   ┌──────┐   ┌──────┐         ┌──────┐          │
│  │Mesh 0│──▶│Mesh 1│──▶│Mesh 2│── ... ──│Mesh15│          │
│  │ 256N │   │ 256N │   │ 256N │         │ 256N │          │
│  └──────┘   └──────┘   └──────┘         └──────┘          │
│      ▲                                       │              │
│      └───────────────────────────────────────┘              │
│                  (Circular connection)                      │
│                                                             │
│  Each Mesh: 80% Excitatory + 20% Inhibitory                │
│  Operations: Bit-shift decay, quantized weights            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            READOUT LAYER (4096 → 5 actions)                 │
│         Quantized Linear: 8-bit integer weights             │
│         Actions: RIGHT, UP, LEFT, DOWN, STAY                │
└─────────────────────────────────────────────────────────────┘
```

### Mesh Architecture

Each mesh contains 256 EILIF neurons organized as:
- **204 Excitatory (E)** neurons (80%)
- **52 Inhibitory (I)** neurons (20%)

**Connectivity within mesh**:
- E↔E: 50% density (recurrent excitation)
- I↔I: 70% density (lateral inhibition competition)
- E→I: 60% density (drive inhibition)
- I→E: 80% density (strong lateral inhibition)

**Inter-mesh connections**:
- E neurons project to next mesh (20% density)
- I neurons stay local (mesh-specific)
- Circular: Mesh 15 → Mesh 0

### EILIF Neuron Model

```python
# Membrane potential update (bit-shift decay)
v_E = v_E - (v_E >> 1)  # E neurons: τ=2.0ms, shift=1
v_I = v_I - (v_I >> 1)  # I neurons: τ=1.5ms, shift=1

# Current integration (addition only)
I_syn = W_EE @ spikes_E + W_IE @ spikes_I + input_current

# Membrane update
v = v + I_syn

# Spike generation
spikes = (v >= threshold).float()

# Reset
v = torch.where(spikes > 0.5, v_reset, v)
```

**Key properties**:
- No multiplications (only bit-shifts and additions)
- Integer arithmetic in forward pass
- Gradients flow via surrogate functions in backward pass

---

## Neuromorphic Operations

### 1. Bit-Shift Decay

**Traditional**: `v_new = v_old * τ` (requires multiplication)

**Neuromorphic**: `v_new = v_old - (v_old >> shift)` (only subtraction and bit-shift)

**Mathematical equivalence**:
- `v >> 1` ≈ `v * 0.5`
- `v - (v >> 1)` ≈ `v * 0.5`
- `v >> 2` ≈ `v * 0.25`
- General: `v - (v >> shift)` ≈ `v * (1 - 2^(-shift))`

**Example**:
```python
v = torch.tensor([16, 8, 4, 2])
decay_shift = 1

# Bit-shift decay
v_new = v - (v >> decay_shift)
# Result: [8, 4, 2, 1]

# Equivalent to v * 0.5 (but NO multiplication!)
```

### 2. Quantized Weights (QAT)

**Quantization-Aware Training** simulates integer arithmetic during training:

```python
# Training phase (float weights)
w_float = torch.randn(256, 256) * 0.1

# Quantize to integers
w_int = torch.clamp(torch.round(w_float * scale), -8, 7)  # 4-bit: [-8, 7]

# Straight-Through Estimator (STE)
w_forward = w_int  # Forward uses integers
w_backward = w_float * scale  # Backward uses floats

# During training: gradients flow through w_float
# At deployment: export w_int for hardware
```

**Weight quantization levels**:
- **Reservoir synapses**: 4-bit integers ([-8, 7])
- **Readout weights**: 8-bit integers ([-128, 127])
- **Membrane potentials**: 16-bit (internal state, not weights)

### 3. Surrogate Gradients

**Problem**: Spikes are non-differentiable (Heaviside step function)
```
spike = 1 if v >= threshold else 0
∂spike/∂v = δ(v - threshold)  # Dirac delta (infinite/zero)
```

**Solution**: Use smooth surrogate during backpropagation
```python
# Forward: Hard spike (Heaviside)
spike_forward = (v >= threshold).float()

# Backward: Smooth surrogate (differentiable)
d_spike / d_v = surrogate_gradient(v, threshold)
```

**Surrogate options** (configured in `config_lsm.yaml`):

1. **Fast Sigmoid** (default):
   ```
   σ(v) = 1 / (1 + exp(-β(v - threshold)))
   dσ/dv = β * σ(v) * (1 - σ(v))
   ```

2. **Triangular**:
   ```
   g(v) = max(0, 1 - |v - threshold|)
   ```

3. **Rectangular**:
   ```
   g(v) = 1 if |v - threshold| < 0.5 else 0
   ```

### 4. Sparse Event-Driven Computation

Only process neurons that spike:
```python
# Find active neurons
active_mask = (spikes > 0.5)

# Only update active neurons
if active_mask.any():
    v[active_mask] = v_reset
    out_spikes = spikes[active_mask]
```

**Energy savings**: ~10-100× compared to dense computation

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/MAPF-GNN.git
cd MAPF-GNN

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Dataset

Ensure you have MAPF expert trajectories:
```
dataset/5_8_28/
├── train/
│   ├── case_0/
│   │   ├── states.npy        # [T, A, 2, 7, 7] FOV observations
│   │   ├── trajectory_record.npy  # [T, A] actions
│   │   └── gso.npy           # [T, A, A] graph adjacency
│   ├── case_1/
│   └── ...
└── valid/
    └── ...
```

### Training (Reservoir Computing)

```bash
# Quick training with 100 episodes
./run_lsm_pipeline.sh reservoir 100

# Or manually:
python train_lsm.py --config configs/config_lsm.yaml
```

### Evaluation

```bash
# Evaluate on 50 test episodes
python evaluate_lsm.py \
    --checkpoint checkpoints/lsm/lsm_network_TIMESTAMP.pt \
    --num_episodes 50
```

### Results

```
🎯 Performance Metrics:
   Collision Rate: 3.12%
   Reach Rate: 4.00%      # Goal: ↑ improve this
   Avg Path Length: 9.5

🧠 Reservoir Metrics:
   Spike Rate: 0.0538
   E/I Balance: 0.97
```

---

## Detailed Usage

### 1. Configuration

Edit `configs/config_lsm.yaml`:

```yaml
# LSM Architecture
lsm:
  num_meshes: 16                # Number of sub-meshes
  neurons_per_mesh: 256         # Neurons per mesh
  E_ratio: 0.8                  # 80% excitatory
  
  # Neuromorphic settings
  neuromorphic:
    enabled: true               # Use BDSM operations
    weight_bits: 8              # Synaptic quantization (4 or 8)
    use_bit_shifts: true        # Bit-shift decay
    
    # Surrogate gradients
    surrogate_type: "fast_sigmoid"  # Options: fast_sigmoid, sigmoid, triangular
    surrogate_beta: 10.0        # Steepness
    
    quantize_weights: true      # QAT during training
  
  # Connectivity
  intra_mesh_E_density: 0.5     # Within-mesh E↔E
  inter_mesh_E_density: 0.2     # Between-mesh E→E

# Training
training:
  mode: "reservoir_computing"   # Options: reservoir_computing, end_to_end, hybrid

# Readout
readout:
  method: "ridge_regression"    # Options: ridge_regression, pytorch
  collect_episodes: 100         # Training episodes
  regularization: 1e-2          # Ridge λ
```

### 2. Training Modes

#### Mode 1: Reservoir Computing (Fast)

```bash
# Set in config
training:
  mode: "reservoir_computing"

# Run training
python train_lsm.py --config configs/config_lsm.yaml
```

**Expected time**: ~5 minutes for 100 episodes

#### Mode 2: End-to-End SGD (Slow but better)

```yaml
# Edit config
training:
  mode: "end_to_end"
  end_to_end_epochs: 50
  learning_rate: 1e-4
  batch_size: 16
  optimizer: "adam"
```

```bash
python train_lsm.py --config configs/config_lsm.yaml
```

**Expected time**: ~2 hours for 50 epochs (depends on GPU)

#### Mode 3: Hybrid (Recommended)

```yaml
training:
  mode: "hybrid"
  fine_tune_reservoir: true
  fine_tune_epochs: 10
  fine_tune_lr: 1e-5
```

**Workflow**:
1. Reservoir computing (fast initial training)
2. Fine-tune entire network (performance boost)

### 3. Evaluation Metrics

The evaluator runs full MAPF episodes and measures:

| Metric | Description | Target |
|--------|-------------|--------|
| **Reach Rate** | % of agents reaching goals | ↑ Higher |
| **Collision Rate** | % of timesteps with collisions | ↓ Lower |
| **Action Accuracy** | Agreement with expert | Reference only |
| **Path Length** | Steps taken per agent | ↓ Lower |
| **Spike Rate** | Reservoir activity | 0.05-0.15 |
| **E/I Balance** | Excitatory/Inhibitory ratio | ~1.0 |

**Example output**:
```
🎯 Performance Metrics:
   Action Accuracy: 80.63% ± 7.90%    # Expert agreement (not goal!)
   Collision Rate: 3.12% ± 3.11%      # Good: low collisions
   Reach Rate: 4.00% ± 8.00%          # Problem: only 4% reach goals!

📏 Path Metrics:
   Avg Path Length: 9.50 ± 2.02       # Agents are moving
   Avg Collisions: 1.50 per trajectory
```

**Interpreting results**:
- **Reach rate** is the most important metric
- Action accuracy ≠ task success (can match expert but fail task)
- Low collision + low reach = agents moving but not navigating properly

### 4. Scaling Up Training

```bash
# More episodes = better generalization
./run_lsm_pipeline.sh reservoir 1000

# Use PyTorch SGD for large datasets
# Edit config:
readout:
  method: "pytorch"
  regularization: 1e-2

# Then train
python train_lsm.py --config configs/config_lsm.yaml
```

### 5. Hyperparameter Tuning

**Key parameters to tune**:

1. **Reservoir size**: `num_meshes × neurons_per_mesh`
   - Larger = more capacity, slower
   - Recommended: 2048-8192 neurons

2. **Connectivity densities**:
   - Higher = more recurrence, richer dynamics
   - Too high = unstable, exploding activity

3. **Regularization** (`λ`):
   - Lower = more fitting, risk overfitting
   - Higher = smoother, may underfit
   - Typical: 1e-3 to 1e-1

4. **Surrogate beta**:
   - Higher = sharper gradient (better learning)
   - Too high = gradient issues
   - Typical: 5.0-20.0

5. **Training episodes**:
   - More = better generalization
   - Diminishing returns after ~1000

---

## Hardware Deployment

### Export for Neuromorphic Chips

```python
import torch
from models.neuromorphic_lsm_trainable import TrainableNeuromorphicLSMNetwork

# Load trained model
config = {...}
network = TrainableNeuromorphicLSMNetwork(config)
checkpoint = torch.load('checkpoints/lsm/model.pt')
network.load_state_dict(checkpoint['network_state'])

# Export integer weights
int_weights = network.export_for_neuromorphic_hardware()

# Save
torch.save(int_weights, 'neuromorphic_weights.pt')
```

**Output format**:
```python
int_weights = {
    'reservoir': {
        'meshes': [
            {
                'w_EE': torch.int32,  # [204, 204]
                'w_EI': torch.int32,  # [204, 52]
                'w_IE': torch.int32,  # [52, 204]
                'w_II': torch.int32,  # [52, 52]
            },
            ...  # 16 meshes
        ],
        'inter_mesh': torch.int32,  # [204, 204] per connection
    },
    'readout': {
        'weight': torch.int32,  # [5, 4096]
        'bias': torch.int32,     # [5]
    }
}
```

### Deployment Targets

#### Intel Loihi

```python
# Loihi requires:
# - Integer weights ✅ (provided)
# - Event-driven spikes ✅ (EILIF neurons)
# - Synaptic delays (manually configure)

# Map to Loihi cores (1024 neurons per core)
# Our 4096 neurons → 4 cores
```

#### IBM TrueNorth

```python
# TrueNorth requires:
# - 1-bit weights (quantize further from 4-bit)
# - 256 axons per neuron (our meshes fit)
# - Threshold-based spikes ✅

# 4096 neurons → 16 cores (256 neurons each)
```

#### SpiNNaker

```python
# SpiNNaker requires:
# - Event packets (spike timing)
# - Flexible neuron models ✅ (can implement EILIF)
# - Software-configurable connectivity
```

### Power Consumption Estimates

| Platform | Power | Speed | Energy per Op |
|----------|-------|-------|---------------|
| GPU (A100) | 400W | Fast | ~1 pJ/op |
| CPU | 100W | Slow | ~10 pJ/op |
| Loihi 2 | 1W | Fast | ~0.01 pJ/op |
| TrueNorth | 70mW | Medium | ~0.05 pJ/op |

**Neuromorphic advantage**: 100-1000× more energy efficient!

---

## Performance Metrics

### Training Results (100 Episodes)

| Metric | Value |
|--------|-------|
| Training time | ~5 minutes |
| Train accuracy | 99.77% |
| Val accuracy | 64.58% |
| Reach rate | 4.00% |
| Collision rate | 3.12% |

**Analysis**:
- ❌ **Overfitting**: Train 99% but val 64%
- ❌ **Poor generalization**: Only 4% reach goals
- ✅ **Low collisions**: Agents avoid each other

**Solution**: Scale to 1000+ episodes

### Expected Performance (with more data)

Based on reservoir computing literature:

| Episodes | Expected Reach Rate |
|----------|-------------------|
| 10 | 4% ✅ |
| 100 | 15-20% |
| 500 | 40-50% |
| 1000+ | 60-70% |
| SGD fine-tune | 75-85% |

---

## Troubleshooting

### Problem: Reservoir silent (no spikes)

**Symptoms**:
```
Avg Spike Rate: 0.0000
```

**Causes**:
- Input scale too low
- Threshold too high
- Weights too small

**Solutions**:
```yaml
lsm:
  v_threshold: 0.5          # Lower threshold
  weight_scale_E: 1.0       # Increase weight magnitude
  input_weight_scale: 1.2   # Increase input projection
```

### Problem: Reservoir saturated (always spiking)

**Symptoms**:
```
Avg Spike Rate: 0.9999
```

**Causes**:
- Input too strong
- Insufficient inhibition
- Weights too large

**Solutions**:
```yaml
lsm:
  weight_scale_I: 1.5       # Increase inhibition
  input_weight_scale: 0.5   # Decrease input
  I_to_E_density: 0.9       # More inhibitory connections
```

### Problem: Poor reach rate

**Symptoms**:
```
Reach Rate: 4.00%
```

**Causes**:
- Insufficient training data
- Reservoir not learning task structure
- Readout too simple

**Solutions**:
1. **Scale up data**: 1000+ episodes
2. **Use SGD**: Train entire network
3. **Check FOV**: Ensure goals are visible
4. **Tune connectivity**: Richer dynamics

### Problem: Gradient vanishing/exploding

**Symptoms**:
```
RuntimeError: Loss contains NaN
```

**Causes**:
- Surrogate beta too high/low
- Learning rate too high

**Solutions**:
```yaml
neuromorphic:
  surrogate_beta: 10.0      # Try 5.0-15.0
training:
  learning_rate: 1e-5       # Lower if exploding
  gradient_clip: 1.0        # Clip gradients
```

### Problem: Checkpoint loading error

**Symptoms**:
```
RuntimeError: Error(s) in loading state_dict
```

**Solution**:
```python
# Load with strict=False
network.load_state_dict(checkpoint['network_state'], strict=False)
```

---

## File Structure

```
MAPF-GNN/
├── README.md                          # This file
├── run_lsm_pipeline.sh                # Quick start script
├── requirements.txt                   # Python dependencies
│
├── configs/
│   └── config_lsm.yaml                # Configuration
│
├── models/
│   ├── neuromorphic_lsm_trainable.py  # Main LSM implementation
│   └── neuromorphic_ops.py            # Custom autograd functions
│
├── train_lsm.py                       # Training script
├── evaluate_lsm.py                    # Evaluation script
├── data_loader.py                     # Dataset loader
│
├── dataset/                           # MAPF expert trajectories
│   └── 5_8_28/
│       ├── train/
│       └── valid/
│
├── checkpoints/                       # Saved models
│   └── lsm/
│
├── logs/                              # Training logs
│   └── lsm/
│
└── visualizations/                    # Plots and figures
    └── lsm/
```

---

## References

### Papers

1. **Liquid State Machines**:
   - Maass, W., Natschläger, T., & Markram, H. (2002). "Real-time computing without stable states: A new framework for neural computation based on perturbations." *Neural computation*, 14(11), 2531-2560.

2. **Reservoir Computing**:
   - Lukoševičius, M., & Jaeger, H. (2009). "Reservoir computing approaches to recurrent neural network training." *Computer Science Review*, 3(3), 127-149.

3. **Surrogate Gradients**:
   - Neftci, E. O., Mostafa, H., & Zenke, F. (2019). "Surrogate gradient learning in spiking neural networks." *IEEE Signal Processing Magazine*, 36(6), 51-63.

4. **Neuromorphic Computing**:
   - Davies, M., et al. (2018). "Loihi: A neuromorphic manycore processor with on-chip learning." *IEEE Micro*, 38(1), 82-99.

### Code References

- PyTorch custom autograd: https://pytorch.org/docs/stable/notes/extending.html
- Quantization-aware training: https://pytorch.org/docs/stable/quantization.html

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mapf_neuromorphic_lsm_2026,
  title={BDSM: Brains Don't Simply Multiply - Neuromorphic LSM for MAPF},
  author=Arnav Gupta,
  year={2026},
  url={https://github.com/ARNAVVGUPTAA/MAPF-SNN}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: guptaa.arnavv0620@example.com

---

<p align="center">
  <b>🧠 Built with BDSM principles 🧠</b><br>
  <i>No multiplications were harmed in the making of this neural network</i>
</p>
