import torch
import torch.nn.functional as F
import numpy as np
import yaml
from models.swarm_lsm import SwarmLSM
from ablation_eval import (
    random_grid, ensure_min_free_cells, sample_starts_goals,
    load_swarm_checkpoint, EpisodeStats, generate_observations, Coord
)
from typing import List, Set, Tuple

def simulate_episode_with_usage(
    network: SwarmLSM,
    grid: np.ndarray,
    starts: torch.Tensor,
    goals: torch.Tensor,
    fov_size: int,
    num_ticks: int,
    max_steps: int,
    deadlock_patience: int,
    device: str,
):
    n = starts.shape[0]
    positions = starts.to(device).clone()
    goals = goals.to(device)
    reached = torch.zeros(n, dtype=torch.bool, device=device)

    network.reset()

    paths: List[List[Coord]] = [
        [(int(positions[i, 0].item()), int(positions[i, 1].item()))]
        for i in range(n)
    ]
    
    used_cpg = [False] * n
    used_ghost = [False] * n

    no_progress = 0
    delta = torch.tensor(
        [[1, 0], [0, -1], [-1, 0], [0, 1], [0, 0]],
        dtype=torch.float32,
        device=device,
    )
    phero_grid = torch.zeros((grid.shape[0], grid.shape[1]), dtype=torch.float32, device=device)
    idle_agents = torch.ones(n, dtype=torch.bool, device=device)

    for step in range(1, max_steps + 1):
        phero_grid *= 0.95
        for i in range(n):
            if reached[i]:
                continue
            if idle_agents[i]:
                px = int(positions[i, 0].item())
                py = int(positions[i, 1].item())
                phero_grid[py, px] = min(phero_grid[py, px].item() + 1.0, 5.0)

        pad = fov_size // 2
        phero_pad = F.pad(phero_grid, (pad, pad, pad, pad), value=0.0)
        phero_fovs = torch.zeros((n, fov_size, fov_size), dtype=torch.float32, device=device)
        for i in range(n):
            px = int(positions[i, 0].item())
            py = int(positions[i, 1].item())
            phero_fovs[i] = phero_pad[py:py + fov_size, px:px + fov_size]

        obs = generate_observations(grid, positions, goals, fov_size=fov_size, device=device)
        with torch.no_grad():
            action_spikes, _, _, veto_flags, _ = network(
                obs,
                positions,
                num_ticks=num_ticks,
                goals=goals,
                pheromones=phero_fovs,
            )
            actions = action_spikes.argmax(dim=-1)
            
        # Track usage
        if hasattr(network, 'last_should_stay'):
            for i in range(n):
                if network.last_should_stay[i]:
                    used_cpg[i] = True
        
        if hasattr(network, 'frustration_timers'):
            for i in range(n):
                if network.frustration_timers[i] > 0:
                    used_ghost[i] = True

        prev = positions.clone()
        attempted = prev.clone()
        movable = ~reached
        if movable.any():
            attempted[movable] = prev[movable] + delta[actions[movable]]
        new = attempted.clone()

        # Wall collisions
        for i in range(n):
            ax = int(attempted[i, 0].item())
            ay = int(attempted[i, 1].item())
            oob = ax < 0 or ax >= grid.shape[1] or ay < 0 or ay >= grid.shape[0]
            if oob or grid[ay, ax] > 1.5:
                new[i] = prev[i]

        # Edge collisions
        for i in range(n):
            for j in range(i + 1, n):
                i_to_j = (
                    int(new[i, 0].item()) == int(prev[j, 0].item())
                    and int(new[i, 1].item()) == int(prev[j, 1].item())
                )
                j_to_i = (
                    int(new[j, 0].item()) == int(prev[i, 0].item())
                    and int(new[j, 1].item()) == int(prev[i, 1].item())
                )
                if i_to_j and j_to_i:
                    new[i] = prev[i]
                    new[j] = prev[j]

        # Vertex collisions
        vertex_passes = 0
        max_vertex_passes = max(1, n)
        resolved = False
        while not resolved and vertex_passes < max_vertex_passes:
            resolved = True
            occ = {}
            for i in range(n):
                k = (int(new[i, 0].item()), int(new[i, 1].item()))
                if k in occ:
                    j = occ[k]
                    if not torch.equal(new[i], prev[i]) or not torch.equal(new[j], prev[j]):
                        new[i] = prev[i]
                        new[j] = prev[j]
                        resolved = False
                else:
                    occ[k] = i
            vertex_passes += 1

        positions = new
        idle_agents = (positions == prev).all(dim=1)
        for i in range(n):
            paths[i].append((int(positions[i, 0].item()), int(positions[i, 1].item())))

        moved = bool((positions != prev).any().item())
        no_progress = 0 if moved else no_progress + 1

        reached |= (torch.norm(positions - goals, dim=1) < 0.5)
        if reached.all() or no_progress >= deadlock_patience:
            break

    return paths, used_cpg, used_ghost


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
    
    print(f"Running on {device}, comparing avg agent dev/path length (CPG vs Ghost)...")
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
        
        cpg_steps = []
        ghost_steps = []
        both_steps = []
        neither_steps = []
        
        for ep in range(episodes):
            seed = 1000 + ep + n*100
            rng = np.random.default_rng(seed)
            grid = random_grid(grid_size, overlap_density, rng)
            grid = ensure_min_free_cells(grid, n * 2, rng)
            starts, goals = sample_starts_goals(grid, n, rng, min_goal_l1=10)
            
            paths, used_cpg, used_ghost = simulate_episode_with_usage(
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
            
            for i in range(n):
                path = paths[i]
                gx, gy = int(goals[i,0].item()), int(goals[i,1].item())
                steps = len(path) - 1
                for step_idx, pos in enumerate(path):
                    if pos[0] == gx and pos[1] == gy:
                        steps = step_idx
                        break
                
                # Check if they actually reached it
                if path[steps][0] == gx and path[steps][1] == gy:
                    if used_cpg[i] and used_ghost[i]:
                        both_steps.append(steps)
                    elif used_cpg[i]:
                        cpg_steps.append(steps)
                    elif used_ghost[i]:
                        ghost_steps.append(steps)
                    else:
                        neither_steps.append(steps)
                        
        avg_cpg = np.mean(cpg_steps) if cpg_steps else 0.0
        avg_ghost = np.mean(ghost_steps) if ghost_steps else 0.0
        avg_both = np.mean(both_steps) if both_steps else 0.0
        avg_neither = np.mean(neither_steps) if neither_steps else 0.0
        
        print(f"N={n}:")
        print(f"  CPG ONLY   : Avg Steps = {avg_cpg:.2f} (count: {len(cpg_steps)})")
        print(f"  GHOST ONLY : Avg Steps = {avg_ghost:.2f} (count: {len(ghost_steps)})")
        print(f"  BOTH       : Avg Steps = {avg_both:.2f} (count: {len(both_steps)})")
        print(f"  NEITHER    : Avg Steps = {avg_neither:.2f} (count: {len(neither_steps)})")
        print("-" * 50)

if __name__ == '__main__':
    run_sweep()
