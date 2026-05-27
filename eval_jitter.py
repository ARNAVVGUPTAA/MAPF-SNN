"""
eval_jitter.py
==============
Asynchronous timing jitter robustness evaluation for SwarmLSM.

Injects a SpikeJitterBuffer between the Observation Mesh and Readout Mesh
of every AgentLSM without touching model internals.  Physics, pheromones,
goal-halt logic, and cascading vertex collision resolution are all
delegated to the battle-tested simulate_episode() from ablation_eval.py.

Usage:
    cd /home/charismatic/dev/summer25/MAPF-GNN
    python eval_jitter.py
"""

import os
import random
import sys

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.swarm_lsm import SwarmLSM

# Import every physics/map helper from the proven ablation harness.
# This is the single source of truth for simulation correctness.
from ablation_eval import (
    EpisodeStats,
    ensure_min_free_cells,
    load_swarm_checkpoint,
    random_grid,
    sample_starts_goals,
    simulate_episode,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT        = os.path.join(os.path.dirname(__file__), "checkpoints", "lsm", "phase1_cortex.pt")
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
NUM_AGENTS        = 10
GRID_SIZE         = 20          # square grid — matches ablation_eval default
OBSTACLE_DENSITY  = 0.10        # matches ablation_eval default (0.10, not 0.25)
FOV_SIZE          = 7
NUM_TICKS         = 10
MAX_STEPS         = 1000        # same cap as ablation_eval
DEADLOCK_PATIENCE = 100         # same patience as ablation_eval
NUM_EPISODES      = 30
JITTER_PROBS      = [0.00, 0.05, 0.10, 0.15, 0.20]
MIN_GOAL_L1       = 10          # minimum Manhattan distance between start and goal
MAX_MAP_RETRIES   = 30          # attempts to find a valid map before skipping
SEED_BASE         = 1234


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — SPIKE JITTER BUFFER
# ─────────────────────────────────────────────────────────────────────────────

class SpikeJitterBuffer(nn.Module):
    """
    Drop-in wrapper for AgentLSM.obs_to_readout (ProjectionNeurons, output=[64]).

    Dynamically inserted between the Observation Mesh and the Readout Mesh.
    No changes to SwarmLSM internals are needed; the wrapper is hot-swapped
    onto `ag.obs_to_readout` before each jitter-probability run.

    Timing model
    ------------
    At each MAPF timestep the model calls  `ag.obs_to_readout(obs_E)`.
    This wrapper intercepts that call:

    1. Runs the wrapped ProjectionNeurons to get `y ∈ R^64`.
    2. Decrements all buffered-spike delays; spikes whose countdown hits ≤1
       are drained into `released ∈ R^64` (additive superposition).
    3. With probability `jitter_prob`, delays `y` by 1 or 2 timesteps:
         - Appends `(delay, y)` to the internal FIFO buffer.
         - Returns only `released` this cycle (zeros if buffer was empty).
    4. Otherwise, returns `y + released` — on-time spikes merged with any
       previously buffered spikes that have now expired.

    Tensor dimensions: both `y` and every buffered entry are `[64]`, so
    the addition is always shape-safe.
    """

    def __init__(self, wrapped_module: nn.Module, jitter_prob: float = 0.10):
        super().__init__()
        self.wrapped     = wrapped_module
        self.jitter_prob = jitter_prob
        # List of (remaining_delay: int, spike_tensor: Tensor[64])
        self._buffer: list = []

    # ------------------------------------------------------------------
    def reset_buffer(self) -> None:
        """Flush stale delayed spikes — call once at the start of each episode."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : Tensor[num_E=204]  — E-neuron spike accumulator from Obs Mesh.

        Returns
        -------
        Tensor[64]  — compressed spike representation for Readout Mesh input.
        """
        # 1. Run the real projection.
        y = self.wrapped(x)                        # [64]

        # 2. Drain expired buffered spikes.
        released      = torch.zeros_like(y)
        still_pending = []
        for remaining, buffered in self._buffer:
            if remaining <= 1:
                released = released + buffered     # superpose; shape stays [64]
            else:
                still_pending.append((remaining - 1, buffered))
        self._buffer = still_pending

        # 3. Jitter decision.
        if random.random() < self.jitter_prob:
            delay = random.randint(1, 2)
            self._buffer.append((delay, y.detach().clone()))
            # Current cycle receives only the just-released (possibly zero) spikes.
            return released
        else:
            # Current cycle receives on-time spikes + any newly released ones.
            return y + released


# ─────────────────────────────────────────────────────────────────────────────
# INJECTION / REMOVAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def inject_jitter(model: SwarmLSM, jitter_prob: float) -> None:
    """Hot-swap every agent's obs_to_readout with a SpikeJitterBuffer."""
    for ag in model.agents:
        # Never double-wrap; always peel back to the bare ProjectionNeurons.
        inner = ag.obs_to_readout
        if isinstance(inner, SpikeJitterBuffer):
            inner = inner.wrapped
        ag.obs_to_readout = SpikeJitterBuffer(inner, jitter_prob=jitter_prob).to(DEVICE)


def remove_jitter(model: SwarmLSM) -> None:
    """Restore the original ProjectionNeurons on every agent."""
    for ag in model.agents:
        if isinstance(ag.obs_to_readout, SpikeJitterBuffer):
            ag.obs_to_readout = ag.obs_to_readout.wrapped


def reset_jitter_buffers(model: SwarmLSM) -> None:
    """Flush per-episode stale spikes from all active jitter buffers."""
    for ag in model.agents:
        if isinstance(ag.obs_to_readout, SpikeJitterBuffer):
            ag.obs_to_readout.reset_buffer()


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — LIVE JITTER SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_jitter_sweep() -> dict:
    """
    Sweep jitter_prob ∈ JITTER_PROBS.  For each value:
      • Inject SpikeJitterBuffer on every agent's obs_to_readout.
      • Run NUM_EPISODES closed-loop MAPF simulations using the proven
        simulate_episode() physics engine from ablation_eval.py.
      • Record Success Rate (fraction of episodes where all agents reach
        their goals) and Makespan (mean episode length in timesteps).

    All probabilities see the same set of pre-seeded maps, controlling for
    map-difficulty variance.

    Returns
    -------
    dict : { jitter_prob → {"success_rate": float, "makespan": float} }
    """
    print(f"\n{'='*60}")
    print(f"  JITTER ROBUSTNESS SWEEP  |  N={NUM_AGENTS}  Episodes={NUM_EPISODES}")
    print(f"  Grid: {GRID_SIZE}×{GRID_SIZE}  Obstacle density: {OBSTACLE_DENSITY}")
    print(f"{'='*60}\n")

    # ── Build model ──────────────────────────────────────────────────────
    model = SwarmLSM(
        num_agents=NUM_AGENTS,
        communication_range=3.0,
        enable_cpg=True,
        enable_shadow=True,
        enable_ghost=True,
        enable_veto_bridge=True,
    ).to(DEVICE)

    missing, unexpected = load_swarm_checkpoint(model, CHECKPOINT, DEVICE)
    model.eval()
    print(f"[MODEL] Loaded checkpoint  missing={missing}  unexpected={unexpected}\n")

    # ── Pre-generate deterministic episode seeds ─────────────────────────
    # Same seeds → same maps for every jitter probability level.
    master_rng   = np.random.default_rng(SEED_BASE)
    episode_seeds = [int(master_rng.integers(0, 2**31)) for _ in range(NUM_EPISODES)]

    results: dict = {}

    for prob in JITTER_PROBS:
        print(f"[SWEEP] jitter_prob={prob:.2f}")
        inject_jitter(model, jitter_prob=prob)

        successes: list = []
        makespans: list = []
        map_skips = 0

        for ep_idx, seed in enumerate(episode_seeds):
            rng = np.random.default_rng(seed)

            # Build a valid random map — retry if not enough free cells
            grid = None
            starts = goals = None
            for _retry in range(MAX_MAP_RETRIES):
                g = random_grid(GRID_SIZE, OBSTACLE_DENSITY, rng)
                g = ensure_min_free_cells(g, min_free=NUM_AGENTS * 2, rng=rng)
                try:
                    s, gl = sample_starts_goals(g, NUM_AGENTS, rng, min_goal_l1=MIN_GOAL_L1)
                    grid, starts, goals = g, s, gl
                    break
                except ValueError:
                    rng = np.random.default_rng(seed + _retry + 1)

            if grid is None:
                map_skips += 1
                continue

            # Flush stale jitter spikes from previous episode
            reset_jitter_buffers(model)

            # Delegate to ablation_eval's proven physics engine
            stats: EpisodeStats = simulate_episode(
                network=model,
                grid=grid,
                starts=starts,
                goals=goals,
                fov_size=FOV_SIZE,
                num_ticks=NUM_TICKS,
                max_steps=MAX_STEPS,
                deadlock_patience=DEADLOCK_PATIENCE,
                device=DEVICE,
            )

            successes.append(stats.success_all)
            makespans.append(stats.makespan)

            if (ep_idx + 1) % 10 == 0:
                n_done = len(successes)
                sr = sum(successes) / n_done * 100
                ms = sum(makespans) / n_done
                print(f"  ep {ep_idx+1:3d}/{NUM_EPISODES}  "
                      f"SR={sr:5.1f}%  Makespan={ms:.1f}  skips={map_skips}")

        n_valid = len(successes)
        sr  = sum(successes) / n_valid if n_valid else 0.0
        ms  = sum(makespans) / n_valid if n_valid else float(MAX_STEPS)
        results[prob] = {"success_rate": sr, "makespan": ms, "n_valid": n_valid}

        print(f"  ── Final: SR={sr:.1%}  Makespan={ms:.1f}  "
              f"valid_eps={n_valid}  skips={map_skips}\n")

    remove_jitter(model)

    # ── Summary table ────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"{'Jitter':>8}  {'Success Rate':>13}  {'Makespan':>10}  {'Valid Eps':>9}")
    print("-" * 48)
    for p, v in results.items():
        print(f"  {p:.2f}    {v['success_rate']:>12.1%}  "
              f"{v['makespan']:>10.1f}  {v['n_valid']:>9d}")
    print()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — IEEE DUAL-AXIS PUBLICATION FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_jitter_robustness(results: dict, out_path: str = "jitter_robustness.png") -> None:
    """
    Dual-axis line plot formatted for IEEE two-column papers.

    Left  Y-axis : Success Rate (%)
    Right Y-axis : Makespan (timesteps)
    X-axis       : Jitter Probability
    Typography   : Times New Roman, 10 pt body, 300 DPI, tight layout
    """
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":         10,
        "axes.titlesize":    10,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "lines.linewidth":   1.5,
        "lines.markersize":  5,
        "figure.dpi":        300,
    })

    probs   = sorted(results.keys())
    sr_vals = [results[p]["success_rate"] * 100 for p in probs]
    ms_vals = [results[p]["makespan"]           for p in probs]

    COLOR_SR = "#1565C0"   # IEEE blue  — success rate
    COLOR_MS = "#B71C1C"   # deep red   — makespan

    # IEEE single-column width ≈ 3.5 inches
    fig, ax1 = plt.subplots(figsize=(3.5, 2.8))

    # ── Left axis: Success Rate ──────────────────────────────────────────
    ln1 = ax1.plot(
        probs, sr_vals,
        color=COLOR_SR, marker="o", linestyle="-",
        label="Success Rate (%)",
    )
    ax1.set_xlabel("Jitter Probability", fontsize=10)
    ax1.set_ylabel("Success Rate (%)", color=COLOR_SR, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=COLOR_SR)
    ax1.set_ylim(-5, 105)
    ax1.set_xticks(probs)
    ax1.set_xticklabels([f"{p:.2f}" for p in probs])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax1.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.45)

    # ── Right axis: Makespan ─────────────────────────────────────────────
    ax2 = ax1.twinx()
    ln2 = ax2.plot(
        probs, ms_vals,
        color=COLOR_MS, marker="s", linestyle="--",
        label="Makespan (steps)",
    )
    ax2.set_ylabel("Makespan (steps)", color=COLOR_MS, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=COLOR_MS)
    ms_range = max(ms_vals) - min(ms_vals) if len(ms_vals) > 1 else 1
    ax2.set_ylim(
        max(0, min(ms_vals) - ms_range * 0.10),
        max(ms_vals) + ms_range * 0.12,
    )

    # ── Unified legend ───────────────────────────────────────────────────
    all_lines  = ln1 + ln2
    all_labels = [l.get_label() for l in all_lines]
    ax1.legend(
        all_lines, all_labels,
        loc="lower left",
        framealpha=0.88,
        handlelength=1.5,
        borderpad=0.4,
        edgecolor="#cccccc",
    )

    ax1.set_title(
        f"Jitter Robustness: SwarmLSM ($N={NUM_AGENTS}$, {NUM_EPISODES} ep.)",
        fontsize=9, fontweight="bold", pad=4,
    )

    plt.tight_layout()
    abs_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)
    plt.savefig(abs_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {abs_out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_jitter_sweep()
    plot_jitter_robustness(results, out_path="jitter_robustness.png")
    print("[DONE] jitter_robustness.png written.\n")
