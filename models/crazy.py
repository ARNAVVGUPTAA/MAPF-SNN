import matplotlib.pyplot as plt
import numpy as np
import os

def plot_agent_scaling():
    # Data extracted directly from your summary.txt
    agents = [2, 5, 10, 15, 20]
    
    # Rates calculated as: (reached / (episodes * agents)) * 100
    goal_rate = [
        (196/200)*100,     # N=2
        (449/500)*100,     # N=5
        (875/1000)*100,    # N=10
        (1341/1500)*100,   # N=15
        (1843/2000)*100    # N=20
    ]
    
    success_rate = [98.0, 77.0, 54.0, 44.0, 48.0] # strict success (all agents)
    collision_rate = [0.0, 0.0, 0.0, 0.0, 0.0]    # 0 collisions across the board

    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    ax.plot(agents, goal_rate, marker='o', lw=2.5, markersize=8, label='Goal Reach %', color='#1f77b4')
    ax.plot(agents, success_rate, marker='s', lw=2.5, markersize=8, label='Strict MAPF Success %', color='#ff7f0e')
    ax.plot(agents, collision_rate, marker='X', lw=2.5, markersize=8, label='Collision %', color='#d62728')

    ax.set_title('Subsumption-LSM: Density Scaling (7x7 FOV)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Agents', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xticks(agents)
    ax.set_ylim(-2, 105)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11, loc='center right')
    
    plt.tight_layout()
    plt.savefig('agent_scaling.png', dpi=200)
    print("✅ Saved agent_scaling.png")
    plt.close()


def plot_module_ablation():
    # Data extracted from summary.txt (ablation sweep at N=10)
    labels = ['Full\nArchitecture', 'w/o Shadow\n(No VETO)', 'w/o Ghost\n(No Pheromone)', 'w/o CPG\n(No Yielding)']
    
    goal_rate = [(875/1000)*100, (999/1000)*100, 0.0, (884/1000)*100]
    success_rate = [54.0, 99.0, 0.0, 52.0]
    # Collision rate based on collided_agents_total / total_agents
    collision_rate = [0.0, (478/1000)*100, 0.0, 0.0]

    x = np.arange(len(labels))
    width = 0.25  # width of the bars

    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Plotting grouped bar charts
    rects1 = ax.bar(x - width, goal_rate, width, label='Goal Reach %', color='#1f77b4')
    rects2 = ax.bar(x, success_rate, width, label='Strict MAPF Success %', color='#ff7f0e')
    rects3 = ax.bar(x + width, collision_rate, width, label='Collision %', color='#d62728')

    # Add text labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.set_title('Module Ablation Study (N=10 Agents)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 115) # Give room for labels on top
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('module_ablation.png', dpi=200)
    print("✅ Saved module_ablation.png")
    plt.close()

if __name__ == "__main__":
    plot_agent_scaling()
    plot_module_ablation()