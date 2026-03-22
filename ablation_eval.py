"""
Ablation benchmark for MAPF-SNN.

Protocol:
- Random maps only (no handcrafted pinhole/direct maps)
- Fixed obstacle density (single value)
- Supports both:
    1) Agent sweep (full model over varying agent counts)
    2) Ablation sweep (fixed N/FOV with one-module-off variants)
- Multi-agent qualitative plot for each ablation variant
- Summary plot for goal %, success %, and collision %
"""

import argparse
import csv
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Set, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.swarm_lsm import SwarmLSM, SwarmTrainer


Coord = Tuple[int, int]


ABLATION_VARIANTS = {
    "full": {
        "label": "Full Subsumption-LSM",
        "enable_cpg": True,
        "enable_shadow": True,
        "enable_ghost": True,
        "enable_veto_bridge": True,
    },
    "no_shadow": {
        "label": "w/o Shadow Caster (No VETO)",
        "enable_cpg": True,
        "enable_shadow": False,
        "enable_ghost": True,
        "enable_veto_bridge": False,
    },
    "no_ghost": {
        "label": "w/o Ghost Antenna (No Pheromones)",
        "enable_cpg": True,
        "enable_shadow": True,
        "enable_ghost": False,
        "enable_veto_bridge": True,
    },
    "no_cpg": {
        "label": "w/o CPG Mesh (No Turn-Taking)",
        "enable_cpg": False,
        "enable_shadow": True,
        "enable_ghost": True,
        "enable_veto_bridge": True,
    },
}


@dataclass
class EpisodeStats:
    success_all: bool
    reached_count: int
    not_reached_count: int
    collided_agents_count: int
    collision_events: int
    makespan: int
    agent_paths: List[List[Coord]]
    collision_points: List[Coord]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(text: str) -> List[int]:
    vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not vals:
        raise ValueError(f"Empty int list: {text}")
    return vals


def parse_ablation_list(text: str) -> List[str]:
    vals = [v.strip() for v in text.split(",") if v.strip()]
    if not vals:
        raise ValueError(f"Empty ablation list: {text}")
    unknown = [v for v in vals if v not in ABLATION_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown ablation variants: {', '.join(unknown)}")
    return vals


def random_grid(size: int, obstacle_prob: float, rng: np.random.Generator) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.float32)
    mask = rng.random((size, size)) < obstacle_prob
    grid[mask] = 2.0
    return grid


def ensure_min_free_cells(grid: np.ndarray, min_free: int, rng: np.random.Generator) -> np.ndarray:
    free = int((grid < 1.5).sum())
    if free >= min_free:
        return grid
    blocked = np.argwhere(grid > 1.5)
    need = min_free - free
    if len(blocked) == 0:
        return grid
    take = min(need, len(blocked))
    pick = rng.choice(len(blocked), size=take, replace=False)
    for i in pick:
        y, x = blocked[i]
        grid[y, x] = 0.0
    return grid


def sample_starts_goals(
    grid: np.ndarray,
    num_agents: int,
    rng: np.random.Generator,
    min_goal_l1: int,
    tries: int = 300,
) -> Tuple[torch.Tensor, torch.Tensor]:
    free = np.argwhere(grid < 1.5)
    if len(free) < num_agents * 2:
        raise ValueError("Not enough free cells")

    free_xy = free[:, [1, 0]]
    s_idx = rng.choice(len(free_xy), size=num_agents, replace=False)
    starts = free_xy[s_idx]
    rem = np.delete(free_xy, s_idx, axis=0)

    goals = None
    for _ in range(tries):
        g_idx = rng.choice(len(rem), size=num_agents, replace=False)
        cand = rem[g_idx]
        d = np.abs(starts[:, 0] - cand[:, 0]) + np.abs(starts[:, 1] - cand[:, 1])
        if bool((d >= min_goal_l1).all()):
            goals = cand
            break
    if goals is None:
        goals = rem[rng.choice(len(rem), size=num_agents, replace=False)]

    return (
        torch.tensor(starts, dtype=torch.float32),
        torch.tensor(goals, dtype=torch.float32),
    )


def generate_observations(
    grid: np.ndarray,
    positions: torch.Tensor,
    goals: torch.Tensor,
    fov_size: int,
    device: str,
) -> torch.Tensor:
    """Generate dynamic FOV observations and resample to 7x7 for the fixed network input."""
    n = positions.shape[0]
    pad = fov_size // 2
    grid_pad = np.pad(grid, ((pad, pad), (pad, pad)), constant_values=2.0)

    dyn = grid_pad.copy()
    for j in range(n):
        px = int(positions[j, 0].item()) + pad
        py = int(positions[j, 1].item()) + pad
        dyn[py, px] = 2.0

    obs = np.zeros((n, 2, fov_size, fov_size), dtype=np.float32)
    for i in range(n):
        x = int(positions[i, 0].item())
        y = int(positions[i, 1].item())
        gx = int(goals[i, 0].item())
        gy = int(goals[i, 1].item())

        fov = dyn[y:y + fov_size, x:x + fov_size].copy()
        fov[pad, pad] = 0.0
        obs[i, 0] = fov

        rel_x = gx - x + pad
        rel_y = gy - y + pad
        if 0 <= rel_x < fov_size and 0 <= rel_y < fov_size:
            obs[i, 1, rel_y, rel_x] = 3.0
        else:
            ex = max(0, min(fov_size - 1, rel_x))
            ey = max(0, min(fov_size - 1, rel_y))
            obs[i, 1, ey, ex] = 1.5

    obs_t = torch.from_numpy(obs).to(device)
    if fov_size != 7:
        obs_t = F.interpolate(obs_t, size=(7, 7), mode="nearest")
    return obs_t


def draw_multi_agent_plot(
    grid: np.ndarray,
    agent_paths: List[List[Coord]],
    goals: torch.Tensor,
    collision_points: List[Coord],
    title: str,
    out_path: str,
) -> None:
    h, w = grid.shape
    fig, ax = plt.subplots(figsize=(8, 8))

    base = np.where(grid > 1.5, 0.0, 1.0)
    ax.imshow(base, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)

    n_agents = len(agent_paths)
    cmap = plt.get_cmap("tab20", max(1, n_agents))
    for i, path in enumerate(agent_paths):
        if not path:
            continue
        color = cmap(i)
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.9)

        sx, sy = path[0]
        gx, gy = int(goals[i, 0].item()), int(goals[i, 1].item())
        ax.scatter(sx, sy, c=[color], s=42, marker="o")
        ax.scatter(gx, gy, c=[color], s=66, marker="*")
        ax.text(sx + 0.1, sy - 0.1, f"S{i}", color=color, fontsize=7)
        ax.text(gx + 0.1, gy - 0.1, f"G{i}", color=color, fontsize=7)

    if collision_points:
        cx = [p[0] for p in collision_points]
        cy = [p[1] for p in collision_points]
        ax.scatter(cx, cy, c="#7f0000", marker="x", s=64, linewidths=1.8)

    ax.set_title(title)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks(np.arange(0, w, 1))
    ax.set_yticks(np.arange(0, h, 1))
    ax.grid(which="both", color="#cccccc", linewidth=0.5, alpha=0.45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_ablation_summary_plot(summary_rows: List[Dict[str, object]], out_path: str) -> None:
    labels = [str(row["variant"]) for row in summary_rows]
    goal_pct = [100.0 * float(row["reached_agent_rate"]) for row in summary_rows]
    success_pct = [100.0 * float(row["success_all_rate"]) for row in summary_rows]
    collision_pct = [100.0 * float(row["collided_agent_rate"]) for row in summary_rows]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, goal_pct, marker="o", linewidth=2.2, label="Goal %")
    ax.plot(x, success_pct, marker="s", linewidth=2.2, label="Success %")
    ax.plot(x, collision_pct, marker="x", linewidth=2.2, color="red", label="Collision %")

    ax.set_title("Ablation Sweep: Goal %, Success %, Collision %")
    ax.set_xlabel("Architecture Variant")
    ax.set_ylabel("Percentage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def extract_checkpoint_state(raw_state: object) -> Dict[str, torch.Tensor]:
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        state_dict = raw_state["state_dict"]
        if isinstance(state_dict, dict):
            return state_dict
    if isinstance(raw_state, dict):
        return raw_state
    raise TypeError("Unsupported checkpoint format")


def load_swarm_checkpoint(network: SwarmLSM, checkpoint_path: str, device: str) -> Tuple[int, int]:
    state = extract_checkpoint_state(torch.load(checkpoint_path, map_location=device))

    agent0_state = {
        key.replace("agents.0.", "", 1): value
        for key, value in state.items()
        if key.startswith("agents.0.")
    }
    if not agent0_state:
        load_info = network.load_state_dict(state, strict=False)
        return len(load_info.missing_keys), len(load_info.unexpected_keys)

    missing_keys = 0
    unexpected_keys = 0
    for agent in network.agents:
        load_info = agent.load_state_dict(agent0_state, strict=False)
        missing_keys += len(load_info.missing_keys)
        unexpected_keys += len(load_info.unexpected_keys)

    return missing_keys, unexpected_keys


def simulate_episode(
    network: SwarmLSM,
    grid: np.ndarray,
    starts: torch.Tensor,
    goals: torch.Tensor,
    fov_size: int,
    num_ticks: int,
    max_steps: int,
    deadlock_patience: int,
    device: str,
) -> EpisodeStats:
    n = starts.shape[0]
    positions = starts.to(device).clone()
    goals = goals.to(device)
    reached = torch.zeros(n, dtype=torch.bool, device=device)

    network.reset()

    paths: List[List[Coord]] = [
        [(int(positions[i, 0].item()), int(positions[i, 1].item()))]
        for i in range(n)
    ]
    collision_points: List[Coord] = []
    collided_agents: Set[int] = set()

    no_progress = 0
    delta = torch.tensor(
        [[1, 0], [0, -1], [-1, 0], [0, 1], [0, 0]],
        dtype=torch.float32,
        device=device,
    )
    phero_grid = torch.zeros((grid.shape[0], grid.shape[1]), dtype=torch.float32, device=device)
    # Prime as idle so agents can leave an initial trace on step 1 if they do not move.
    idle_agents = torch.ones(n, dtype=torch.bool, device=device)

    for step in range(1, max_steps + 1):
        phero_grid *= 0.95
        for i in range(n):
            # Do not radiate toxicity once the agent is already at goal.
            if reached[i]:
                continue

            # Drop pheromone only when this agent was idle in the previous step.
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

        # Noradrenaline chemistry is managed internally by the network's
        # frustration_timers; no external two-pass injection needed here.

        prev = positions.clone()
        attempted = prev.clone()
        movable = ~reached
        if movable.any():
            attempted[movable] = prev[movable] + delta[actions[movable]]
        new = attempted.clone()

        # Wall / bounds collisions.
        for i in range(n):
            ax = int(attempted[i, 0].item())
            ay = int(attempted[i, 1].item())
            oob = ax < 0 or ax >= grid.shape[1] or ay < 0 or ay >= grid.shape[0]
            if oob:
                cx = min(max(ax, 0), grid.shape[1] - 1)
                cy = min(max(ay, 0), grid.shape[0] - 1)
                collision_points.append((cx, cy))
                collided_agents.add(i)
                new[i] = prev[i]
            elif grid[ay, ax] > 1.5:
                collision_points.append((ax, ay))
                collided_agents.add(i)
                new[i] = prev[i]

        # Edge swap collisions.
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
                    collision_points.append((int(prev[i, 0].item()), int(prev[i, 1].item())))
                    collision_points.append((int(prev[j, 0].item()), int(prev[j, 1].item())))
                    collided_agents.add(i)
                    collided_agents.add(j)
                    new[i] = prev[i]
                    new[j] = prev[j]

        # Vertex collisions can cascade when a revert pushes an agent back into
        # a cell claimed by someone else. Re-run until the occupancy settles.
        vertex_passes = 0
        max_vertex_passes = max(1, n)
        resolved = False
        while not resolved and vertex_passes < max_vertex_passes:
            resolved = True
            occ: Dict[Coord, int] = {}
            for i in range(n):
                k = (int(new[i, 0].item()), int(new[i, 1].item()))
                if k in occ:
                    j = occ[k]
                    if not torch.equal(new[i], prev[i]) or not torch.equal(new[j], prev[j]):
                        collision_points.append(k)
                        collided_agents.add(i)
                        collided_agents.add(j)
                        new[i] = prev[i]
                        new[j] = prev[j]
                        resolved = False
                else:
                    occ[k] = i
            vertex_passes += 1

        positions = new
        # Agents that did not move this step are marked idle for next step's deposition.
        idle_agents = (positions == prev).all(dim=1)
        for i in range(n):
            paths[i].append((int(positions[i, 0].item()), int(positions[i, 1].item())))

        moved = bool((positions != prev).any().item())
        no_progress = 0 if moved else no_progress + 1

        reached |= (torch.norm(positions - goals, dim=1) < 0.5)
        if reached.all():
            return EpisodeStats(
                success_all=True,
                reached_count=int(reached.sum().item()),
                not_reached_count=int((~reached).sum().item()),
                collided_agents_count=len(collided_agents),
                collision_events=len(collision_points),
                makespan=step,
                agent_paths=paths,
                collision_points=collision_points,
            )
        if no_progress >= deadlock_patience:
            break

    return EpisodeStats(
        success_all=False,
        reached_count=int(reached.sum().item()),
        not_reached_count=int((~reached).sum().item()),
        collided_agents_count=len(collided_agents),
        collision_events=len(collision_points),
        makespan=max_steps,
        agent_paths=paths,
        collision_points=collision_points,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reactive MAPF benchmark sweep")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["agent_sweep", "ablation_sweep", "both", "diagnostics_only"],
    )
    parser.add_argument("--config", type=str, default="configs/config_swarm.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lsm/phase1_cortex.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--obstacle-density", type=float, default=0.10)
    parser.add_argument("--num-ticks", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--deadlock-patience", type=int, default=100)
    parser.add_argument("--agent-counts", type=str, default="2,5,10,15,20")
    parser.add_argument("--agent-sweep-fov", type=int, default=7)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--fov-size", type=int, default=7)
    parser.add_argument("--ablations", type=str, default="full,no_shadow,no_ghost,no_cpg")
    parser.add_argument("--diagnostics-variant", type=str, default="full")
    parser.add_argument("--min-goal-distance", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--max-map-retries", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="logs")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ablation_names = parse_ablation_list(args.ablations)
    agent_counts = parse_int_list(args.agent_counts)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"sweep_{stamp}")
    plot_dir = os.path.join(run_dir, "qualitative")
    os.makedirs(plot_dir, exist_ok=True)

    if args.mode == "diagnostics_only":
        if args.diagnostics_variant not in ABLATION_VARIANTS:
            raise ValueError(f"Unknown diagnostics variant: {args.diagnostics_variant}")

        variant_cfg = ABLATION_VARIANTS[args.diagnostics_variant]
        network = SwarmLSM(
            num_agents=args.num_agents,
            communication_range=cfg["swarm"].get("communication_range", 3.0),
            enable_cpg=bool(variant_cfg["enable_cpg"]),
            enable_shadow=bool(variant_cfg["enable_shadow"]),
            enable_ghost=bool(variant_cfg["enable_ghost"]),
            enable_veto_bridge=bool(variant_cfg["enable_veto_bridge"]),
        ).to(device)

        if os.path.isfile(args.checkpoint):
            missing_keys, unexpected_keys = load_swarm_checkpoint(network, args.checkpoint, device)
            if missing_keys:
                print(f"[warn] Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"[warn] Unexpected keys: {unexpected_keys}")
        else:
            print(f"[warn] checkpoint not found: {args.checkpoint}")

        diag_dir = os.path.join(run_dir, "diagnostics")
        trainer = SwarmTrainer(network=network, device=device)
        trainer._save_diagnostics(dataset=None, num_ticks=args.num_ticks, save_dir=diag_dir)

        print("\nSaved diagnostics:")
        print(f"- {diag_dir}/subsumption_trace.png")
        print(f"- {diag_dir}/cpg_antiphase.png")
        print(f"- {diag_dir}/lsm_raster.png")
        return

    summary_rows: List[Dict[str, object]] = []
    per_episode_rows: List[Dict[str, object]] = []

    run_configs: List[Dict[str, object]] = []
    if args.mode in ("agent_sweep", "both"):
        for n_agents in agent_counts:
            run_configs.append(
                {
                    "mode": "agent_sweep",
                    "variant": "full",
                    "variant_cfg": ABLATION_VARIANTS["full"],
                    "num_agents": n_agents,
                    "fov_size": args.agent_sweep_fov,
                }
            )

    if args.mode in ("ablation_sweep", "both"):
        for variant_name in ablation_names:
            run_configs.append(
                {
                    "mode": "ablation_sweep",
                    "variant": variant_name,
                    "variant_cfg": ABLATION_VARIANTS[variant_name],
                    "num_agents": args.num_agents,
                    "fov_size": args.fov_size,
                }
            )

    for run_cfg in run_configs:
        run_mode = str(run_cfg["mode"])
        variant_name = str(run_cfg["variant"])
        variant_cfg = run_cfg["variant_cfg"]
        num_agents = int(run_cfg["num_agents"])
        fov_size = int(run_cfg["fov_size"])
        print("\n" + "=" * 72)
        print(
            f"Config: variant={variant_name}, agents={num_agents}, "
            f"fov={fov_size}x{fov_size}, mode={run_mode}"
        )
        print("=" * 72)

        network = SwarmLSM(
            num_agents=num_agents,
            communication_range=cfg["swarm"].get("communication_range", 3.0),
            enable_cpg=bool(variant_cfg["enable_cpg"]),
            enable_shadow=bool(variant_cfg["enable_shadow"]),
            enable_ghost=bool(variant_cfg["enable_ghost"]),
            enable_veto_bridge=bool(variant_cfg["enable_veto_bridge"]),
        ).to(device)

        if os.path.isfile(args.checkpoint):
            missing_keys, unexpected_keys = load_swarm_checkpoint(network, args.checkpoint, device)
            if missing_keys:
                print(f"[warn] Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"[warn] Unexpected keys: {unexpected_keys}")
        else:
            print(f"[warn] checkpoint not found: {args.checkpoint}")

        success_eps = 0
        total_reached = 0
        total_not_reached = 0
        total_collided_agents = 0
        total_collision_events = 0
        total_makespan = 0.0
        map_skips = 0
        runtime_errors = 0
        violation_eps = 0

        plotted = False
        for ep in range(args.episodes):
            seed = args.seed_base + ep
            set_seed(seed)
            rng = np.random.default_rng(seed)

            starts = goals = None
            grid = None
            for _ in range(args.max_map_retries):
                grid = random_grid(args.grid_size, args.obstacle_density, rng)
                grid = ensure_min_free_cells(grid, min_free=num_agents * 2, rng=rng)
                try:
                    starts, goals = sample_starts_goals(
                        grid,
                        num_agents,
                        rng,
                        min_goal_l1=args.min_goal_distance,
                    )
                    break
                except ValueError:
                    continue

            if starts is None or goals is None or grid is None:
                map_skips += 1
                continue

            try:
                stats = simulate_episode(
                    network=network,
                    grid=grid,
                    starts=starts,
                    goals=goals,
                    fov_size=fov_size,
                    num_ticks=args.num_ticks,
                    max_steps=args.max_steps,
                    deadlock_patience=args.deadlock_patience,
                    device=device,
                )
            except Exception as e:
                runtime_errors += 1
                if runtime_errors <= 3:
                    print(f"[warn] seed {seed} failed: {e}")
                continue

            success_eps += int(stats.success_all)
            total_reached += stats.reached_count
            total_not_reached += stats.not_reached_count
            total_collided_agents += stats.collided_agents_count
            total_collision_events += stats.collision_events
            total_makespan += stats.makespan

            # Core invariant requested by user: collisions should be impossible with shadow+ghost on.
            if stats.collision_events > 0:
                violation_eps += 1

            per_episode_rows.append(
                {
                    "mode": run_mode,
                    "variant": variant_name,
                    "agents": num_agents,
                    "fov": fov_size,
                    "seed": seed,
                    "success_all": int(stats.success_all),
                    "reached_agents": stats.reached_count,
                    "not_reached_agents": stats.not_reached_count,
                    "collided_agents": stats.collided_agents_count,
                    "collision_events": stats.collision_events,
                    "makespan": stats.makespan,
                }
            )

            if not plotted:
                plot_name = f"{variant_name}_a{num_agents}_f{fov_size}.png"
                draw_multi_agent_plot(
                    grid=grid,
                    agent_paths=stats.agent_paths,
                    goals=goals,
                    collision_points=stats.collision_points,
                    title=(
                        f"{variant_name} | agents={num_agents} | "
                        f"fov={fov_size} | seed={seed}"
                    ),
                    out_path=os.path.join(plot_dir, plot_name),
                )
                plotted = True

            if args.log_every > 0 and ((ep + 1) % args.log_every == 0 or (ep + 1) == args.episodes):
                print(
                    f"  progress {ep + 1:3d}/{args.episodes} | success_ep={success_eps} "
                    f"coll_events={total_collision_events} map_skips={map_skips} errors={runtime_errors}"
                )

        valid_eps = max(1, args.episodes - map_skips - runtime_errors)
        total_agent_slots = valid_eps * num_agents

        row = {
            "mode": run_mode,
            "variant": variant_name,
            "variant_label": variant_cfg["label"],
            "agents": num_agents,
            "fov": fov_size,
            "episodes_requested": args.episodes,
            "episodes_valid": valid_eps,
            "episodes_success_all": success_eps,
            "success_all_rate": success_eps / valid_eps,
            "reached_agents_total": total_reached,
            "not_reached_agents_total": total_not_reached,
            "reached_agent_rate": (total_reached / total_agent_slots) if total_agent_slots > 0 else 0.0,
            "collided_agents_total": total_collided_agents,
            "collided_agent_rate": (total_collided_agents / total_agent_slots) if total_agent_slots > 0 else 0.0,
            "collision_events_total": total_collision_events,
            "collision_event_rate": (total_collision_events / total_agent_slots) if total_agent_slots > 0 else 0.0,
            "avg_makespan": total_makespan / valid_eps,
            "invariant_collision_free_pass": int(violation_eps == 0),
            "invariant_violation_episodes": violation_eps,
            "map_skips": map_skips,
            "runtime_errors": runtime_errors,
        }
        summary_rows.append(row)

        print(
            f"  success_all={row['success_all_rate']:.3f} | reached={total_reached} "
            f"not_reached={total_not_reached} | collided_agents={total_collided_agents} "
            f"collision_events={total_collision_events}"
        )
        if violation_eps > 0:
            print(f"  [ALERT] collision-free invariant violated in {violation_eps} episodes")

    summary_csv = os.path.join(run_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    ep_csv = os.path.join(run_dir, "per_episode.csv")
    if per_episode_rows:
        with open(ep_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_episode_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_episode_rows)

    metrics_plot = os.path.join(run_dir, "ablation_metrics.png")
    draw_ablation_summary_plot(summary_rows, metrics_plot)

    summary_txt = os.path.join(run_dir, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Sweep Summary\n")
        f.write("=" * 72 + "\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Grid size: {args.grid_size}x{args.grid_size}\n")
        f.write(f"Obstacle density (fixed): {args.obstacle_density}\n")
        f.write(f"Agent sweep counts: {args.agent_counts} @ FOV={args.agent_sweep_fov}\n")
        f.write(f"Agents (fixed): {args.num_agents}\n")
        f.write(f"FOV (fixed): {args.fov_size}\n")
        f.write(f"Ablations: {args.ablations}\n")
        f.write(f"Episodes per config: {args.episodes}\n")
        f.write(f"Max steps: {args.max_steps}\n")
        f.write("\n")

        for r in summary_rows:
            f.write(
                f"[{r['mode']}] variant={r['variant']} "
                f"agents={r['agents']} fov={r['fov']}\n"
            )
            f.write(f"  episodes valid: {r['episodes_valid']}\n")
            f.write(f"  success_all_rate: {r['success_all_rate']:.3f}\n")
            f.write(f"  reached_agents_total: {r['reached_agents_total']}\n")
            f.write(f"  not_reached_agents_total: {r['not_reached_agents_total']}\n")
            f.write(f"  collided_agents_total: {r['collided_agents_total']}\n")
            f.write(f"  collision_events_total: {r['collision_events_total']}\n")
            f.write(f"  avg_makespan: {r['avg_makespan']:.2f}\n")
            f.write(f"  collision_free_invariant_pass: {bool(r['invariant_collision_free_pass'])}\n")
            f.write("\n")

    print("\nSaved:")
    print(f"- {summary_csv}")
    if per_episode_rows:
        print(f"- {ep_csv}")
    print(f"- {metrics_plot}")
    print(f"- {summary_txt}")
    print(f"- {plot_dir}")


if __name__ == "__main__":
    main()
