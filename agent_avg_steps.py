import torch
import numpy as np
import yaml
from models.swarm_lsm import SwarmLSM
from ablation_eval import (
    random_grid, ensure_min_free_cells, sample_starts_goals, simulate_episode,
    load_swarm_checkpoint, EpisodeStats, ABLATION_VARIANTS
)

def run_sweep():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent_counts = [2, 5, 10, 15, 20]
    episodes = 20
    grid_size = 20
    overlap_density = 0.10
    fov_size = 7
    
    with open("configs/config_swarm.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    checkpoint = "checkpoints/lsm/phase1_cortex.pt"
    
    print(f"Running on {device}, tracking avg agent timesteps to goal...")
    for n in agent_counts:
        network = SwarmLSM(
            num_agents=n,
            communication_range=cfg["swarm"].get("communication_range", 3.0),
            enable_cpg=True,
            enable_shadow=True,
            enable_ghost=True,
            enable_veto_bridge=True,
        ).to(device)
        
        load_swarm_checkpoint(network, checkpoint, device)
        
        total_steps_to_goal = 0
        agents_reached = 0
        
        for ep in range(episodes):
            seed = 1000 + ep + n*100
            rng = np.random.default_rng(seed)
            grid = random_grid(grid_size, overlap_density, rng)
            grid = ensure_min_free_cells(grid, n * 2, rng)
            starts, goals = sample_starts_goals(grid, n, rng, min_goal_l1=10)
            
            stats = simulate_episode(
                network=network,
                grid=grid,
                starts=starts,
                goals=goals,
                fov_size=fov_size,
                num_ticks=10,
                max_steps=200,
                deadlock_patience=50,
                device=device
            )
            
            # Since simulate_episode returns paths, the step an agent reached 
            # the goal is when it stops moving or the length of its path - 1.
            # actually we can just check when it reached the goal position.
            for i in range(n):
                path = stats.agent_paths[i]
                gx, gy = int(goals[i,0].item()), int(goals[i,1].item())
                steps = len(path) - 1
                for step_idx, pos in enumerate(path):
                    if pos[0] == gx and pos[1] == gy:
                        steps = step_idx
                        break
                # Only count if actually reached
                if path[steps][0] == gx and path[steps][1] == gy:
                    total_steps_to_goal += steps
                    agents_reached += 1
                    
        avg_steps = total_steps_to_goal / max(1, agents_reached)
        print(f"N={n}: Avg Timesteps to Goal = {avg_steps:.2f} (Computed over {agents_reached}/{episodes*n} agents reached)")

if __name__ == '__main__':
    run_sweep()
