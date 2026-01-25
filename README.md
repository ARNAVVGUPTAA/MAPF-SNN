# MAPF-GNN: Biologically-Inspired Spiking Neural Networks for Multi-Agent Path Finding

**A brain-inspired approach to multi-agent coordination using receptor-based SNNs and neuromodulated reinforcement learning**

[Features](#features) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [Training](#training) • [Research](#research)

---

## Overview

MAPF-GNN solves the Multi-Agent Path Finding (MAPF) problem using **biologically-inspired Spiking Neural Networks** (SNNs) with authentic neurotransmitter receptor dynamics. Unlike traditional deep learning approaches, our system mimics how real neurons communicate through:

- **AMPA receptors** (fast excitation) - τ ≈ 2ms
- **NMDA receptors** (slow excitation) - τ ≈ 50ms with Mg²⁺ voltage gating
- **GABA_A receptors** (fast inhibition) - τ ≈ 6ms  
- **GABA_B receptors** (slow inhibition) - τ ≈ 150ms

Combined with **neuromodulated reinforcement learning** (dopamine & GABA modulation), the system learns coordinated multi-agent behaviors through phase-based training that mirrors biological learning.

---

## Features

### **Biological Realism**
- **Receptor-Based Dynamics**: Authentic neurotransmitter receptor implementations with proper time constants
- **Conductance-Based Currents**: `I = g × (V_m - E_rev)` - no artificial multiplications
- **NMDA Voltage Gating**: Mg²⁺ block dependent on membrane potential for Hebbian learning
- **E/I Balance**: 80% excitatory / 20% inhibitory neurons following Dale's principle

### **Neuromodulation**
- **Dopamine System**: Enhances AMPA/NMDA transmission during reward → exploration boost
- **GABA System**: Increases inhibitory tone during punishment → network stabilization
- **Phase-Based Training**:
  - **Exploration** (0-30%): High dopamine, reduced GABA, diverse action sampling
  - **Exploitation** (30-80%): Balanced neuromodulation, policy refinement
  - **Stabilization** (80-100%): Reduced dopamine, increased GABA, converged policy

### **Advanced Architecture**
```
Input (FOV 7×7×2) 
    ↓
[Attention Feature Extractor]
    ↓
[3× SNN Processing Blocks]
    ↓
[Spiking Agent Communication (2 hops)]
    ↓
[Multi-Head Spiking Attention (4 heads)]
    ↓
[Spiking Predictive Model (3 steps ahead)]
    ↓
Output (5 actions: stay, right, up, left, down)
```

### **Training Features**
- **Full Trajectory Learning**: Learns from every timestep (7-15 steps), not just final action
- **Expert Demonstrations**: CBS (Conflict-Based Search) generates optimal trajectories
- **Adaptive Supervision**: Transitions from imitation learning to pure RL
- **Real-Time Monitoring**: Spike rates, E/I ratios, neuron health, action distributions

---

## Architecture

### **ReceptorLIFNeuron**
Core neuronal unit with biological receptor dynamics:

```python
# Membrane dynamics
dV/dt = (V_rest - V)/τ + I_synaptic + I_baseline

# Synaptic currents (conductance-based)
I_AMPA  = g_AMPA  × (V - E_AMPA)   # Fast excitation
I_NMDA  = g_NMDA  × B(V) × (V - E_NMDA)  # Slow excitation + Mg²⁺ block
I_GABAA = g_GABAA × (V - E_GABAA)  # Fast inhibition  
I_GABAB = g_GABAB × (V - E_GABAB)  # Slow inhibition

# Spike generation
spike = 1 if V ≥ V_threshold else 0
```

**Health Metrics:**
- **Spike Rate**: 10-30% (sparse but active)
- **E/I Ratio**: 3-5 (balanced excitation/inhibition)
- **Membrane Voltage**: -70mV (rest) → -55mV (threshold)

### **Neuromodulated Loss Function**

```python
# Dopamine modulation (reward-based)
dopamine = baseline + α × (reward - punishment)

# Loss with neuromodulator weighting
L = dopamine_weight × CE_loss + GABA_weight × activity_reg

# Phase-based scaling
exploration:    reward_scale=2.0,  punishment_scale=0.05
exploitation:   reward_scale=1.0,  punishment_scale=1.0  
stabilization:  reward_scale=0.9,  punishment_scale=1.1
```

---

## Installation

### Prerequisites
```bash
Python 3.14+
PyTorch 2.0+
CUDA 11.8+ (optional, for GPU)
```

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/MAPF-GNN.git
cd MAPF-GNN

# Install dependencies
pip install -r requirements.txt

# Generate dataset (8000 train, 2000 validation)
python generate_dataset.py
```

---

## Usage

### **Quick Start**
```bash
# Train with default settings (100 epochs, batch size 16)
python train_neuromod.py

# Custom training
python train_neuromod.py --epochs 50 --batch_size 32 --lr 0.0001
```

### **Training Arguments**
```bash
--epochs        Number of training epochs (default: 100)
--batch_size    Batch size (default: 16)
--lr            Learning rate (default: 0.0001)
--device        Device: 'cuda' or 'cpu' (auto-detected)
```

### **Dataset Generation**
```bash
# Generate new dataset with CBS expert trajectories
python generate_dataset.py

# Customize dataset
python generate_dataset.py --grid_size 9 --num_agents 5 --train_cases 8000 --val_cases 2000
```

---

## Training

### **Monitoring**

Training provides comprehensive real-time metrics:

```
NEUROMODULATED RL TRAINING | Epoch 1/100 | Batch 25/624
═══════════════════════════════════════════════════════════

LOSS COMPONENTS:
   Raw RL Loss:         1.756
   Scaled Loss:         0.511  (×0.3)
   Entropy Bonus:       1.609  (coeff=0.01)

REWARD/PUNISHMENT:
   Original Reward:     0.000  →  Scaled: 0.000  (÷10)
   Original Punish:    17.300  →  Scaled: 3.460  (÷5)

NEUROMODULATORS:
   Dopamine:            1.135  (baseline=0.25)
   GABA:                1.031  (baseline=0.50)
   Training Phase:      exploration

AGENT PERFORMANCE:
   Collisions:          2/80   ( 2.5%)
   Goals Reached:       0/80   ( 0.0%)

SNN HEALTH:
   Avg Spike Rate:      0.2530  Healthy (10-30%)
   E/I Ratio:           9.02    High (target: 3-5)
```

### **Checkpoints**

Models are automatically saved:
```
checkpoints/
├── best_model.pt              # Best validation performance
├── checkpoint_epoch_50.pt     # Periodic checkpoints
└── final_model.pt             # Final trained model
```

### **Logs**

Training logs saved to `logs/training_output_YYYYMMDD_HHMMSS.txt`

---

## Research

### **Key Innovations**

1. **Receptor-Based SNNs**: First implementation of AMPA/NMDA/GABA_A/GABA_B receptors in SNNs for path planning
2. **Neuromodulated RL**: Biologically-plausible dopamine/GABA modulation for adaptive learning
3. **Phase-Based Training**: Three-phase curriculum mimicking developmental learning
4. **Full Trajectory Learning**: Supervision at every timestep, not just final actions

### **Performance**

- **Dataset**: 9,998 training cases, 998 validation cases
- **Grid Size**: 9×9 with 5 agents
- **FOV**: 7×7 local observation per agent
- **Sequence Length**: 7-15 timesteps (variable)
- **Actions**: 5 discrete (stay, right, up, left, down)

### **Biological Accuracy**

| Component | Biological Value | Implementation |
|-----------|-----------------|----------------|
| AMPA τ | 2ms | 2ms |
| NMDA τ | 50ms | 50ms |
| GABA_A τ | 6ms | 6ms |
| GABA_B τ | 150ms | 150ms |
| V_rest | -70mV | -70mV |
| V_threshold | -55mV | -55mV |
| E/I Ratio | 4:1 | 80/20 split |

---

## Project Structure

```
MAPF-GNN/
├── train_neuromod.py           # Main training script
├── generate_dataset.py         # Dataset generation (CBS)
├── data_loader.py             # Data loading & batching
├── config.py                  # Configuration management
├── neuromodulated_loss_clean.py  # Neuromodulated loss function
├── optimizer_utils.py         # Optimizer setup
│
├── models/
│   ├── framework_snn.py       # SNN architecture
│   └── receptor_dynamics.py   # Receptor neuron implementation
│
├── cbs/
│   ├── cbs.py                # Conflict-Based Search
│   ├── a_star.py             # A* pathfinding
│   └── visualize.py          # Trajectory visualization
│
├── configs/
│   └── config_snn.yaml       # SNN hyperparameters
│
└── dataset/
    └── 5_8_28/
        ├── train/            # 8000 training cases
        └── valid/            # 2000 validation cases
```

---

## Configuration

Edit `configs/config_snn.yaml` to customize:

```yaml
# Training
epochs: 100
batch_size: 16
learning_rate: 0.0001

# Environment  
board_size: [9, 9]
num_agents: 5
num_actions: 5

# SNN Architecture
hidden_dim: 128
num_snn_blocks: 3
use_recurrent_memory: true
num_attention_heads: 4

# Receptor Dynamics
lif_dt: 1.0
tau_mem: 20.0
v_threshold: -55.0
v_reset: -70.0

# Neuromodulation
dopamine_baseline: 0.25
gaba_baseline: 0.50
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mapf-gnn-2025,
  title={MAPF-GNN: Biologically-Inspired Spiking Neural Networks for Multi-Agent Path Finding},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/MAPF-GNN}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## Acknowledgments

- **CBS Algorithm**: Sharon et al., "Conflict-based search for optimal multi-agent pathfinding"
- **Receptor Dynamics**: Dayan & Abbott, "Theoretical Neuroscience"
- **Neuromodulation**: Schultz, "Neuronal reward and decision signals"

---

**Bridging neuroscience and AI for intelligent multi-agent systems**
