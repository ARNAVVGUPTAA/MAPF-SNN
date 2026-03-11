"""
Neuromorphic Blueprint Visualizer
==================================

Visualizes the physical wiring of the hardware-native subsumption chip:
1. Observation Mesh: Sparse liquid dynamics (85% sparse recurrent)
2. Intent Map: Topographic spatial repulsion forcefield (hardwired diagonal)
3. CPG: Turn-taking metronome (strong mutual inhibition between peak/trough)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from models.swarm_lsm import SwarmLSM

# Set dark theme for that proper neuromorphic hacker vibe
plt.style.use('dark_background')

def plot_neuromorphic_blueprint():
    print("🔬 Extracting physical wiring from the SwarmLSM...")
    
    # Instantiate one agent to look at its brain
    model = SwarmLSM(num_agents=1, communication_range=3.0)
    ag = model.agents[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("🧠 Hardware-Native Subsumption Chip Blueprint", 
                 fontsize=18, fontweight='bold', color='cyan', y=0.98)

    # ─── PANEL 1: Observation Mesh (The Sparse Liquid) ──────────────
    # Visualizes the random, 85% sparse recurrent connections
    obs_w = ag.obs_recurrent.weight.detach().numpy()
    
    ax = axes[0]
    # Plot non-zero weights as lit pixels
    ax.imshow(obs_w != 0, cmap='inferno', interpolation='nearest')
    ax.set_title(f"Observation Mesh Recurrent Wiring\n({ag.obs_mesh.num_neurons} Neurons, 85% Sparse)", 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Source Neuron (Output)")
    ax.set_ylabel("Target Neuron (Input)")
    ax.grid(False)

    # ─── PANEL 2: Topographic Intent Map (The Spatial Forcefield) ───
    # Visualizes the beautiful hardwired diagonal you built to replace "telepathy"
    intent_w = ag.intent_map.inhibition_proj.weight.detach().numpy()
    
    ax = axes[1]
    ax.imshow(intent_w, cmap='magma', interpolation='nearest')
    ax.set_title(f"Intent Map Spatial Repulsion\n(144 Neurons, Hardwired Diagonal)", 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Source Neighbor E-Neuron")
    ax.set_ylabel("Target My Neuron")
    ax.grid(False)

    # ─── PANEL 3: Central Pattern Generator (The Turn-Taking Metronome) ───
    # Visualizes the strong -3.0 mutual inhibition between the Peak and Trough halves
    cpg_w = ag.cpg.mutual_inhibition.detach().numpy()
    
    ax = axes[2]
    im3 = ax.imshow(cpg_w, cmap='coolwarm', interpolation='nearest', vmin=-3.5, vmax=1.0)
    ax.set_title(f"CPG Mutual Inhibition\n(32 Neurons, Peak vs. Trough)", 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Source Neuron")
    ax.set_ylabel("Target Neuron")
    ax.grid(False)
    
    # Add a colorbar to the CPG to show the intense inhibitory (blue) vs weak excitatory (red) connections
    cbar = plt.colorbar(im3, ax=ax, shrink=0.7)
    cbar.set_label("Synaptic Voltage (Weight)", color='white')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    save_path = 'neuromorphic_blueprint.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"✅ Masterpiece saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    plot_neuromorphic_blueprint()
