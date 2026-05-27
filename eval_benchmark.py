"""
eval_benchmark.py — Native Multi-Algorithm MAPF Comparison
===========================================================

Compares LSM-MAPF (SNN) against search-based (CBS) and decentralized
greedy (PIBT) baselines on random maps.  No POGEMA dependency — uses
the same map/obs generation as ablation_eval.py.

FPGA framing for LSM-MAPF:
  • Trained params: only readout action_weights — 1020/agent
  • Reservoir/mesh weights are fixed at init, not trained
  • Compute metric: num_ticks (clock cycles on FPGA) + 0 multiplications
  • Hardware MACs = 0; only conditional additions (binary spikes × weight)

Usage:
    python eval_benchmark.py
    python eval_benchmark.py --densities 0.1 0.2 0.3 --n-agents 4 8 16
    python eval_benchmark.py --episodes 30 --grid-size 32
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'models'))

from ablation_eval import (
    random_grid,
    ensure_min_free_cells,
    sample_starts_goals,
    load_swarm_checkpoint,
    generate_observations,
)
from swarm_lsm import SwarmLSM


# ---------------------------------------------------------------------------
# MAC counter — forward hooks on Conv2d / Linear / GRU for dense neural algos
# ---------------------------------------------------------------------------

class MACCounter:
    def __init__(self, net: nn.Module):
        self._count = [0]
        self._hooks: list = []
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                self._hooks.append(m.register_forward_hook(self._conv))
            elif isinstance(m, nn.Linear):
                self._hooks.append(m.register_forward_hook(self._linear))
            elif isinstance(m, (nn.GRUCell, nn.LSTMCell, nn.RNNCell)):
                self._hooks.append(m.register_forward_hook(self._rnn_cell))
            elif isinstance(m, nn.GRU):
                self._hooks.append(m.register_forward_hook(self._gru_mod))

    def _conv(self, m, inp, out):
        b, ci = inp[0].shape[0], inp[0].shape[1]
        co, ho, wo = out.shape[1], out.shape[2], out.shape[3]
        kh = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
        kw = m.kernel_size[1] if isinstance(m.kernel_size, tuple) else m.kernel_size
        self._count[0] += b * ci * co * ho * wo * kh * kw

    def _linear(self, m, inp, out):
        inf, outf = inp[0].shape[-1], out.shape[-1]
        self._count[0] += (inp[0].numel() // inf) * inf * outf

    def _rnn_cell(self, m, inp, out):
        ins = inp[0].shape[-1]
        hs  = (out[0] if isinstance(out, tuple) else out).shape[-1]
        g   = 3 if isinstance(m, (nn.GRUCell, nn.RNNCell)) else 4
        self._count[0] += inp[0].shape[0] * g * (ins + hs) * hs

    def _gru_mod(self, m, inp, out):
        x = inp[0]
        self._count[0] += (x.shape[0] * x.shape[1] * m.num_layers
                           * 3 * (x.shape[2] + m.hidden_size) * m.hidden_size)

    def reset(self):    self._count[0] = 0
    def total(self):    return self._count[0]
    def remove(self):
        for h in self._hooks: h.remove()


def _total_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())

def _trained_params(net: nn.Module) -> int:
    """Only action_weights are trained in SwarmLSM — everything else is a fixed reservoir."""
    aw = sum(p.numel() for n, p in net.named_parameters() if 'action_weights' in n)
    return aw if aw > 0 else sum(p.numel() for p in net.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Synaptic accumulation counter for SNNs
# ---------------------------------------------------------------------------

class SpikeAccumCounter:
    """
    Hooks onto LIF neuron outputs and the immediately following Linear weight
    to estimate synaptic accumulations per forward pass.

    In a spiking network:
      spike (binary) × weight  →  conditional addition, NOT a multiply.
    This counter tallies how many such additions actually happen, i.e.
      additions = sum_layers( active_spikes × fan_out_per_spike )
    where fan_out = number of non-zero output connections from that layer.

    Usage:
        ctr = SpikeAccumCounter(network)
        with torch.no_grad(): network(...)
        adds = ctr.total_adds  # additions this forward pass
        ctr.reset()
    """

    def __init__(self, network: nn.Module):
        self._hooks: list = []
        self._spike_totals: Dict[str, float] = {}
        self._layer_fanout: Dict[str, int]   = {}  # nnz of outgoing Linear weight
        self._mul_count = [0]

        # Map each LIF layer name → name of the next Linear layer that consumes it
        # by walking named_modules in order
        modules = list(network.named_modules())
        for idx, (name, mod) in enumerate(modules):
            is_lif = False
            try:
                from spikingjelly.activation_based import neuron as _sj
                is_lif = isinstance(mod, _sj.LIFNode)
            except ImportError:
                is_lif = 'LIF' in type(mod).__name__

            if is_lif:
                # Find the next Linear sibling/descendant by scanning forward
                fanout = 0
                for _, nmod in modules[idx + 1:]:
                    if isinstance(nmod, nn.Linear):
                        fanout = int((nmod.weight.data != 0).sum().item())
                        break
                self._layer_fanout[name] = fanout

                def _make_hook(n):
                    def _hook(*args):
                        spikes = args[2].detach()
                        self._spike_totals[n] = self._spike_totals.get(n, 0.0) + float(spikes.sum().item())
                    return _hook
                self._hooks.append(mod.register_forward_hook(_make_hook(name)))

    def reset(self):
        self._spike_totals.clear()
        self._mul_count[0] = 0

    @property
    def total_adds(self) -> float:
        """Total synaptic additions in the last forward pass."""
        total = 0.0
        for name, n_spikes in self._spike_totals.items():
            fo = self._layer_fanout.get(name, 1)
            total += n_spikes * fo
        return total

    @property
    def total_muls(self) -> int:
        return 0   # always 0 for SNN — binary spikes need no multipliers

    def remove(self):
        for h in self._hooks: h.remove()

def _mem_mb(net: nn.Module) -> float:
    b = sum(p.numel() * p.element_size() for p in net.parameters())
    b += sum(x.numel() * x.element_size() for x in net.buffers())
    return b / 1e6


# ---------------------------------------------------------------------------
# DCC obs format (if DCC is ever available)
# ---------------------------------------------------------------------------

def _dcc_obs(
    grid: np.ndarray,
    positions: torch.Tensor,
    goals: torch.Tensor,
    obs_radius: int = 4,
) -> list:
    pad = obs_radius
    fov = 2 * pad + 1
    binary = (grid > 1.5).astype(float)
    padded = np.pad(binary, ((pad, pad), (pad, pad)), constant_values=1.0)
    obs_list = []
    for i in range(len(positions)):
        col = int(positions[i, 0].item())
        row = int(positions[i, 1].item())
        pr, pc = row + pad, col + pad
        fov_obs = padded[pr - pad: pr + pad + 1, pc - pad: pc + pad + 1].copy()
        fov_agents = np.zeros((fov, fov), dtype=float)
        for j in range(len(positions)):
            if j == i: continue
            dr = int(positions[j, 1].item()) - row
            dc = int(positions[j, 0].item()) - col
            fr, fc = dr + pad, dc + pad
            if 0 <= fr < fov and 0 <= fc < fov:
                fov_agents[fr, fc] = 0.5
        obs_list.append({
            'obstacles': fov_obs, 'agents': fov_agents,
            'global_xy': (pr, pc),
            'global_target_xy': (int(goals[i, 1].item()) + pad, int(goals[i, 0].item()) + pad),
            'global_obstacles': padded,
        })
    return obs_list


# ---------------------------------------------------------------------------
# Movement and collision helpers — matches ablation_eval.py exactly
# ---------------------------------------------------------------------------

_DELTA = torch.tensor([[1, 0], [0, -1], [-1, 0], [0, 1], [0, 0]], dtype=torch.float32)
# 0=RIGHT(+x)  1=UP(-y)  2=LEFT(-x)  3=DOWN(+y)  4=STAY


def _apply_movement(
    grid: np.ndarray,
    positions: torch.Tensor,
    actions: torch.Tensor,
    reached: torch.Tensor,
) -> torch.Tensor:
    """
    Resolve wall / edge-swap / vertex collisions.  Multi-pass vertex
    resolution (same as ablation_eval.py) so cascading reverts are handled.
    """
    n  = positions.shape[0]
    H, W = grid.shape

    # Compute attempted positions
    attempted = positions.clone()
    movable   = ~reached
    if movable.any():
        attempted[movable] = positions[movable] + _DELTA[actions[movable]]
    new = attempted.clone()

    # Wall / bounds
    for i in range(n):
        ax, ay = int(new[i, 0].item()), int(new[i, 1].item())
        if ax < 0 or ax >= W or ay < 0 or ay >= H or grid[ay, ax] > 1.5:
            new[i] = positions[i]

    # Edge swap
    for i in range(n):
        for j in range(i + 1, n):
            if (int(new[i, 0]) == int(positions[j, 0]) and
                    int(new[i, 1]) == int(positions[j, 1]) and
                    int(new[j, 0]) == int(positions[i, 0]) and
                    int(new[j, 1]) == int(positions[i, 1])):
                new[i] = positions[i]
                new[j] = positions[j]

    # Vertex (multi-pass until stable)
    max_passes = max(1, n)
    for _ in range(max_passes):
        resolved = True
        occ: dict = {}
        for i in range(n):
            k = (int(new[i, 0].item()), int(new[i, 1].item()))
            if k in occ:
                j = occ[k]
                if not torch.equal(new[i], positions[i]) or not torch.equal(new[j], positions[j]):
                    new[i] = positions[i]
                    new[j] = positions[j]
                    resolved = False
            else:
                occ[k] = i
        if resolved:
            break

    return new


# ---------------------------------------------------------------------------
# Algorithm wrappers
# ---------------------------------------------------------------------------

class LSMAgent:
    """SwarmLSM (SNN) with pheromone grid.

    FPGA key facts:
      - trained_params = 1020 per agent (readout action_weights only)
      - hardware_muls  = 0 (binary spikes → only conditional adds)
      - ticks_per_decision = num_ticks (maps to FPGA clock cycles)
    """

    algo_type        = 'SNN — decentralised'
    scalability      = 'O(N) — one forward per agent, pipelined on FPGA'
    muls_note        = '0 multiplications (binary spikes → conditional adds only)'
    alu_cost_profile = 'Adds-Only (0 Multipliers)'
    platform_label   = 'FPGA'

    def sequential_depth(self, n: int) -> str:
        return '1 (all agents pipelined)'

    def est_hw_latency_us(self, fpga_mhz: float) -> float:
        """FPGA decision latency: num_ticks clock cycles at fpga_mhz."""
        return (self.num_ticks / fpga_mhz) * 1000.0

    def __init__(self, n_agents: int, checkpoint: str, device: str = 'cpu', num_ticks: int = 10):
        self.n_agents  = n_agents
        self.network   = SwarmLSM(num_agents=n_agents)
        self.num_ticks = num_ticks
        self.device    = device
        load_swarm_checkpoint(self.network, checkpoint, device)
        self.network.eval()
        self.phero_grid: Optional[torch.Tensor] = None
        self._idle_agents: Optional[torch.Tensor] = None

        # Spike accumulation counter — measures synaptic additions per step
        self._acc = SpikeAccumCounter(self.network)
        self._step_adds: List[float] = []   # adds recorded each act() call

    # FPGA/ops properties
    @property
    def n_params_trained(self):
        return _trained_params(self.network)   # action_weights only: 1020 × n_agents

    @property
    def n_params_per_agent(self):
        return self.n_params_trained // max(self.n_agents, 1)

    @property
    def n_params(self):
        return self.n_params_trained

    @property
    def n_params_total(self):
        return _total_params(self.network)     # includes frozen reservoir (informational)

    @property
    def hardware_muls(self): return 0

    @property
    def macs_per_step(self): return 0

    @property
    def muls_per_step(self): return 0

    @property
    def ticks_per_decision(self): return self.num_ticks

    @property
    def memory_mb(self): return _mem_mb(self.network)

    @property
    def mean_adds_per_step(self) -> float:
        """Mean synaptic additions across all recorded steps (all agents combined)."""
        return float(np.mean(self._step_adds)) if self._step_adds else 0.0

    @property
    def mean_adds_per_step_per_agent(self) -> float:
        """Mean synaptic additions per step divided by agent count."""
        return self.mean_adds_per_step / max(self.n_agents, 1)

    def reset(self, grid_h: int, grid_w: int):
        self.network.reset()
        self.phero_grid   = torch.zeros(grid_h, grid_w, dtype=torch.float32, device=self.device)
        self._idle_agents = torch.ones(self.n_agents, dtype=torch.bool, device=self.device)
        self._acc.reset()
        self._step_adds.clear()

    def act(
        self,
        grid: np.ndarray,
        positions: torch.Tensor,
        goals: torch.Tensor,
        reached: torch.Tensor,
    ) -> torch.Tensor:
        pad = 7 // 2
        self.phero_grid *= 0.95

        for i in range(self.n_agents):
            if not reached[i] and self._idle_agents is not None and self._idle_agents[i]:
                px = int(positions[i, 0].item())
                py = int(positions[i, 1].item())
                self.phero_grid[py, px] = min(self.phero_grid[py, px].item() + 1.0, 5.0)

        phero_pad  = F.pad(self.phero_grid, (pad, pad, pad, pad), value=0.0)
        phero_fovs = torch.zeros(self.n_agents, 7, 7, device=self.device)
        for i in range(self.n_agents):
            px = int(positions[i, 0].item())
            py = int(positions[i, 1].item())
            phero_fovs[i] = phero_pad[py: py + 7, px: px + 7]

        obs = generate_observations(grid, positions.to(self.device), goals.to(self.device),
                                    fov_size=7, device=self.device)
        self._acc.reset()
        with torch.no_grad():
            spikes, _, _, _, _ = self.network(
                obs, positions.to(self.device),
                num_ticks=self.num_ticks,
                goals=goals.to(self.device),
                pheromones=phero_fovs,
            )
        self._step_adds.append(self._acc.total_adds)
        return spikes.argmax(dim=-1).cpu()

    def update_idle(self, prev: torch.Tensor, curr: torch.Tensor):
        self._idle_agents = torch.all(prev == curr, dim=1).to(self.device)


class CBSAgent:
    """Conflict-Based Search — centralized, optimal, pure Python.

    Plans the full episode once on the first act() call, then follows
    the precomputed path.  Latency shown is amortised planning / episode_len.

    Ops tracked: high-level node expansions (compute_solution calls) during
    planning.  Each expansion runs N independent A* searches — so total A*
    state evaluations ≈ node_expansions × avg_path_len × branching_factor.
    0 multiplications — pure integer heuristic (L1 distance).
    """

    algo_type        = 'CBS — centralised optimal'
    scalability      = 'O(N·bᵈ) — exponential in agents × search depth'
    muls_note        = '0 multiplications — A* with L1 heuristic (integer arithmetic)'
    alu_cost_profile = 'Standard ALU (Adds/Muls)'
    platform_label   = 'CPU'

    def sequential_depth(self, n: int) -> str:
        return f'O(b^d·{n}) (CT expansion)'

    def est_hw_latency_us(self, fpga_mhz: float) -> Optional[float]:
        return None  # not applicable — CPU solver, use cpu_wall_time_ms

    n_params           = 0
    n_params_trained   = 0
    n_params_per_agent = 0
    hardware_muls      = 0
    macs_per_step      = 0
    muls_per_step      = 0
    ticks_per_decision = None
    memory_mb          = 0.0

    _MOVE_MAP = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3, (0, 0): 4}

    def __init__(self):
        from cbs.cbs import CBS, Environment
        self._CBS = CBS
        self._Env = Environment
        self._paths: Optional[dict] = None
        self._step: int = 0
        # ops proxy: total planned path cost amortised per step per agent
        # path_cost = sum of all agents' path lengths = CBS solution cost
        # higher cost → deeper / harder search → more computational work
        self._ops_history: List[float] = []

    def reset(self, *_):
        self._paths = None
        self._step  = 0

    @property
    def mean_adds_per_step(self) -> float:
        return float(np.mean(self._ops_history)) if self._ops_history else 0.0

    @property
    def mean_adds_per_step_per_agent(self) -> float:
        return self.mean_adds_per_step

    def _plan(self, grid, positions, goals, reached) -> Optional[dict]:
        import signal
        n = len(positions)
        H, W = grid.shape

        agents = []
        for i in range(n):
            sx, sy = int(positions[i, 0].item()), int(positions[i, 1].item())
            gx = sx if reached[i] else int(goals[i, 0].item())
            gy = sy if reached[i] else int(goals[i, 1].item())
            agents.append({'name': f'a{i}', 'start': [sx, sy], 'goal': [gx, gy]})

        obstacles = {(c, r) for r in range(H) for c in range(W) if grid[r, c] > 1.5}

        try:
            env    = self._Env((W, H), agents, obstacles)
            solver = self._CBS(env, verbose=False)

            def _timeout(*_): raise TimeoutError
            old = signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(30)
            try:
                result = solver.search()
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
            return result if result else None
        except Exception:
            return None

    def act(self, grid, positions, goals, reached) -> torch.Tensor:
        n = len(positions)
        if self._paths is None:
            self._paths = self._plan(grid, positions, goals, reached) or {}
            if self._paths:
                # ops proxy: total solution cost = sum of all path lengths
                # amortised per step per agent = path_cost / makespan / n_agents
                path_cost = sum(len(p) for p in self._paths.values())
                makespan  = max((len(p) for p in self._paths.values()), default=1)
                self._ops_history.append(path_cost / max(makespan, 1) / max(n, 1))

        actions = torch.full((n,), 4, dtype=torch.long)
        t = self._step
        for i in range(n):
            path = self._paths.get(f'a{i}', [])
            if t + 1 < len(path):
                dx = path[t + 1]["x"] - path[t]["x"]
                dy = path[t + 1]["y"] - path[t]["y"]
                actions[i] = self._MOVE_MAP.get((dx, dy), 4)
        self._step += 1
        return actions


def _bfs_dist(grid: np.ndarray, gx: int, gy: int) -> np.ndarray:
    """BFS shortest-path distances from every free cell to (gx, gy)."""
    from collections import deque
    H, W = grid.shape
    dist = np.full((H, W), 1e9, dtype=np.float32)
    if grid[gy, gx] > 1.5:
        return dist
    dist[gy, gx] = 0
    q = deque([(gx, gy)])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1,0),(0,-1),(-1,0),(0,1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < W and 0 <= ny < H and grid[ny, nx] < 1.5 and dist[ny, nx] > 1e8:
                dist[ny, nx] = dist[y, x] + 1
                q.append((nx, ny))
    return dist


class PIBTAgent:
    """Priority Inheritance with Backtracking (decentralised, O(N) per step).

    Corrected Implementation:
    - Pushes low-priority agents out of the way (Priority Inheritance).
    - Checks physical occupancy before reserving cells.
    - Prevents collision overwrites on failed backtracks.
    """

    algo_type        = 'PIBT — decentralised'
    scalability      = 'O(N) per step — no exponential search'
    muls_note        = '0 multiplications — BFS distance lookups + integer comparisons'
    alu_cost_profile = 'Standard ALU (Adds/Muls)'
    platform_label   = 'CPU'

    def sequential_depth(self, n: int) -> str:
        return f'{n} (priority chain)'

    def est_hw_latency_us(self, fpga_mhz: float) -> Optional[float]:
        return None  # CPU solver

    n_params           = 0
    n_params_trained   = 0
    n_params_per_agent = 0
    hardware_muls      = 0
    macs_per_step      = 0
    muls_per_step      = 0
    ticks_per_decision = None
    memory_mb          = 0.0

    _DELTA_LIST = [(1,0,0), (0,-1,1), (-1,0,2), (0,1,3), (0,0,4)]

    def reset(self, *_):
        self._step_ops: List[float] = []
        self._dist_maps: Optional[List[np.ndarray]] = None

    def _ops(self) -> float:
        return float(np.mean(self._step_ops)) if self._step_ops else 0.0

    @property
    def mean_adds_per_step(self) -> float:
        return self._ops()

    @property
    def mean_adds_per_step_per_agent(self) -> float:
        return self._ops()

    def act(self, grid, positions, goals, reached) -> torch.Tensor:
        n    = len(positions)
        H, W = grid.shape
        actions  = torch.full((n,), 4, dtype=torch.long)
        ops = [0]

        if self._dist_maps is None:
            self._dist_maps = [
                _bfs_dist(grid, int(goals[i, 0].item()), int(goals[i, 1].item()))
                for i in range(n)
            ]

        # Priority: smallest BFS distance to goal
        dists = []
        for i in range(n):
            ops[0] += 1
            cx, cy = int(positions[i, 0].item()), int(positions[i, 1].item())
            dists.append(float(self._dist_maps[i][cy, cx]))

        priority_order = sorted(range(n), key=lambda i: (dists[i], i))

        decided  = [False] * n
        visiting = set()           # agents currently in the recursion call stack
        reserved = {}              # cell (x, y) -> agent_id (where agents are going)
        current_pos = {(int(positions[i, 0].item()), int(positions[i, 1].item())): i
                       for i in range(n)}

        def _pibt(agent: int) -> bool:
            if decided[agent]:
                return True
            if agent in visiting:
                return False       # cycle detected — can't displace an in-progress agent

            visiting.add(agent)
            pos = (int(positions[agent, 0].item()), int(positions[agent, 1].item()))
            dm  = self._dist_maps[agent]

            candidates = []
            for dx, dy, a in self._DELTA_LIST:
                nx, ny = pos[0]+dx, pos[1]+dy
                if 0 <= nx < W and 0 <= ny < H and grid[ny, nx] < 1.5:
                    ops[0] += 1
                    candidates.append((float(dm[ny, nx]), a, (nx, ny)))
            candidates.sort()
            ops[0] += len(candidates)

            moved = False
            for d, a, cell in candidates:
                if cell in reserved:
                    continue  # higher-priority agent already claimed this cell

                occupant = current_pos.get(cell)
                if occupant is not None and occupant != agent:
                    if decided[occupant]:
                        continue  # occupant already committed to staying — can't displace
                    # Priority inheritance: push undecided occupant out of the way
                    if not _pibt(occupant):
                        continue  # occupant couldn't move

                reserved[cell] = agent
                actions[agent] = a
                decided[agent] = True
                moved = True
                break

            if not moved:
                # Fallback: nowhere to go, stay
                reserved[pos] = agent
                actions[agent] = 4
                decided[agent] = True

            visiting.discard(agent)
            return moved

        for agent in priority_order:
            if not decided[agent]:
                if reached[agent]:
                    pos = (int(positions[agent, 0].item()), int(positions[agent, 1].item()))
                    if pos not in reserved:
                        reserved[pos] = agent
                        actions[agent] = 4
                        decided[agent] = True
                    else:
                        # Goal cell claimed by higher-priority agent — must move
                        _pibt(agent)
                else:
                    _pibt(agent)

        self._step_ops.append(float(ops[0]))
        return actions


class DCCAgent:
    """DCC — CNN+GRU+Attention decentralised agent (optional, needs weights)."""

    algo_type   = 'DCC — CNN+GRU+Attention (decentralised)'
    scalability = 'O(N²) — pairwise attention in comm block'
    muls_note   = 'dense MACs (measured via hooks)'

    def __init__(self, weights_path: str, device: str = 'cpu'):
        import dcc.config as dcc_cfg
        from dcc.inference import DCCInference, DCCInferenceConfig
        cfg = DCCInferenceConfig(path_to_weights=weights_path, device=device)
        self._inf = DCCInference(cfg)
        self._inf.reset_states()
        net = self._inf.agent
        self._n_params       = _total_params(net)
        self._n_trained      = sum(p.numel() for p in net.parameters() if p.requires_grad)
        self._memory_mb      = _mem_mb(net)
        self._obs_radius     = dcc_cfg.obs_radius
        obs_shape = dcc_cfg.obs_shape
        ctr = MACCounter(net)
        with torch.no_grad():
            try: net.step(torch.zeros(1, *obs_shape), torch.zeros(1, 5), torch.zeros(1, 2, dtype=torch.int))
            except Exception: pass
        self._macs = ctr.total()
        ctr.remove()

    @property
    def n_params(self):           return self._n_params
    @property
    def n_params_trained(self):   return self._n_trained
    @property
    def n_params_per_agent(self): return self._n_params
    @property
    def hardware_muls(self):      return self._macs
    @property
    def macs_per_step(self):      return self._macs
    @property
    def muls_per_step(self):      return self._macs
    @property
    def ticks_per_decision(self): return 1
    @property
    def memory_mb(self):          return self._memory_mb

    def reset(self, *_):
        self._inf.reset_states()

    def act(self, grid, positions, goals, reached):
        obs = _dcc_obs(grid, positions, goals, obs_radius=self._obs_radius)
        raw = self._inf.act(obs)
        pog = {0: 4, 1: 1, 2: 3, 3: 2, 4: 0}
        return torch.tensor([pog.get(a, 4) for a in raw], dtype=torch.long)


class LaCAMAgent:
    """LaCAM* — centralised, scalable, near-optimal MAPF solver.

    Runs the lacam0 binary via subprocess, writes temp map/scen files,
    plans the full episode in one call, then follows the precomputed paths.

    Ops proxy: amortised solution cost / step / agent (same as CBS).
    0 multiplications — A* + PIBT heuristic, pure integer arithmetic.
    """

    algo_type        = 'LaCAM* — centralised near-optimal'
    scalability      = 'O(N·bᵈ) — scalable search, better than CBS in practice'
    muls_note        = '0 multiplications — A* + PIBT config generator, integer arithmetic'
    alu_cost_profile = 'Standard ALU (Adds/Muls)'
    platform_label   = 'CPU'

    def sequential_depth(self, n: int) -> str:
        return f'O(b^d·{n}) (config search)'

    def est_hw_latency_us(self, fpga_mhz: float) -> Optional[float]:
        return None  # CPU solver

    n_params           = 0
    n_params_trained   = 0
    n_params_per_agent = 0
    hardware_muls      = 0
    macs_per_step      = 0
    muls_per_step      = 0
    ticks_per_decision = None
    memory_mb          = 0.0

    _MOVE_MAP = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3, (0, 0): 4}

    def __init__(self, binary: str = 'solvers/lacam0/build/main', time_limit: int = 30):
        self._bin  = binary
        self._tlim = time_limit
        self._paths: Optional[dict] = None
        self._step: int = 0
        self._ops_history: List[float] = []

    def reset(self, *_):
        self._paths = None
        self._step  = 0

    @property
    def mean_adds_per_step(self) -> float:
        return float(np.mean(self._ops_history)) if self._ops_history else 0.0

    @property
    def mean_adds_per_step_per_agent(self) -> float:
        return self.mean_adds_per_step

    @staticmethod
    def _write_map(grid: np.ndarray, path: str):
        H, W = grid.shape
        with open(path, 'w') as f:
            f.write(f"type octile\nheight {H}\nwidth {W}\nmap\n")
            for r in range(H):
                row = ''.join('@' if grid[r, c] > 1.5 else '.' for c in range(W))
                f.write(row + '\n')

    @staticmethod
    def _write_scen(positions, goals, map_path: str, grid_shape, scen_path: str):
        H, W = grid_shape
        with open(scen_path, 'w') as f:
            f.write("version 1\n")
            for i in range(len(positions)):
                sx, sy = int(positions[i, 0].item()), int(positions[i, 1].item())
                gx, gy = int(goals[i, 0].item()),    int(goals[i, 1].item())
                f.write(f"0\t{Path(map_path).name}\t{W}\t{H}\t"
                        f"{sx}\t{sy}\t{gx}\t{gy}\t0\n")

    @staticmethod
    def _parse_solution(result_path: str, n: int) -> Optional[dict]:
        """Parse lacam0 output → dict {agent_idx: [(col, row), ...]}."""
        try:
            with open(result_path) as f:
                content = f.read()
            if 'solved=0' in content:
                return None
            paths: dict = {i: [] for i in range(n)}
            for line in content.splitlines():
                if not line or line[0].isalpha():
                    continue
                colon = line.index(':')
                positions_str = line[colon + 1:].strip().rstrip(',')
                coords = [p.strip('()').split(',') for p in positions_str.split('),(')]
                for i, (x, y) in enumerate(coords):
                    if i < n:
                        paths[i].append((int(x), int(y)))
            return paths if all(paths[i] for i in range(n)) else None
        except Exception:
            return None

    def _plan(self, grid, positions, goals) -> Optional[dict]:
        import subprocess, tempfile
        n = len(positions)
        with tempfile.TemporaryDirectory() as tmp:
            map_p  = f"{tmp}/env.map"
            scen_p = f"{tmp}/agents.scen"
            out_p  = f"{tmp}/result.txt"
            self._write_map(grid, map_p)
            self._write_scen(positions, goals, map_p, grid.shape, scen_p)
            try:
                subprocess.run(
                    [self._bin, '-m', map_p, '-i', scen_p, '-N', str(n),
                     '-o', out_p, '-v', '0', '-t', str(self._tlim)],
                    timeout=self._tlim + 5,
                    capture_output=True,
                )
                return self._parse_solution(out_p, n)
            except Exception:
                return None

    # LaCAM plans may require goal-vacating moves (agents step aside to let
    # others pass).  Setting freeze_reached=False tells run_episode to keep
    # reached agents movable so they can follow the full planned path.
    freeze_reached = False

    def act(self, grid, positions, goals, reached) -> torch.Tensor:
        n = len(positions)
        if self._paths is None:
            raw = self._plan(grid, positions, goals)
            if raw:
                self._paths = raw
                path_cost = sum(len(p) for p in self._paths.values())
                makespan  = max((len(p) for p in self._paths.values()), default=1)
                self._ops_history.append(path_cost / max(makespan, 1) / max(n, 1))
            else:
                self._paths = {}

        actions = torch.full((n,), 4, dtype=torch.long)
        t = self._step
        for i in range(n):
            path = self._paths.get(i, [])
            if t + 1 < len(path):
                dx = path[t + 1][0] - path[t][0]
                dy = path[t + 1][1] - path[t][1]
                actions[i] = self._MOVE_MAP.get((dx, dy), 4)
        self._step += 1
        return actions


# ---------------------------------------------------------------------------
# Episode runner — matches ablation_eval.py run_episode() exactly
# ---------------------------------------------------------------------------

def run_episode(
    agent,
    grid: np.ndarray,
    starts: torch.Tensor,
    goals: torch.Tensor,
    max_steps: int,
    act_times: Optional[List[float]] = None,
    deadlock_patience: int = 100,
) -> dict:
    n  = starts.shape[0]
    H, W = grid.shape
    positions = starts.clone()
    reached   = torch.zeros(n, dtype=torch.bool)

    agent.reset(H, W)

    no_progress = 0

    for step in range(1, max_steps + 1):
        prev = positions.clone()

        t0 = time.perf_counter()
        actions = agent.act(grid, positions, goals, reached)
        t1 = time.perf_counter()
        if act_times is not None:
            act_times.append((t1 - t0) * 1000.0)

        # Plan-following agents (LaCAM, CBS) may need to vacate goal cells so
        # other agents can pass — don't freeze reached agents for those.
        reached_mask = (reached if getattr(agent, 'freeze_reached', True)
                        else torch.zeros(n, dtype=torch.bool))
        positions = _apply_movement(grid, positions, actions, reached_mask)

        if hasattr(agent, 'update_idle'):
            agent.update_idle(prev, positions)

        reached |= (torch.norm(positions.float() - goals.float(), dim=1) < 0.5)

        if reached.all():
            break

        # Deadlock detection: if no agent moved at all, increment counter
        moved = bool((positions != prev).any().item())
        no_progress = 0 if moved else no_progress + 1
        if no_progress >= deadlock_patience:
            break

    isr = reached.float().mean().item()
    csr = 1.0 if reached.all() else 0.0
    return {'CSR': csr, 'ISR': isr, 'ep_length': step}


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def build_agents(n_agents: int, args) -> Dict[str, object]:
    agents: Dict[str, object] = {}

    ckpt = args.lsm_checkpoint
    if Path(ckpt).exists():
        try:
            a = LSMAgent(n_agents, ckpt, device=args.device, num_ticks=args.num_ticks)
            agents['LSM-MAPF'] = a
            print(f"  [agent] LSM-MAPF   trained={a.n_params_trained:,}  "
                  f"({a.n_params_per_agent}/agent)  total={a.n_params_total:,}  "
                  f"mem={a.memory_mb:.2f}MB  ticks={a.ticks_per_decision}")
        except Exception as e:
            print(f"  [agent] LSM-MAPF FAILED: {e}")
    else:
        print(f"  [agent] LSM-MAPF skipped (checkpoint not found: {ckpt})")

    # PIBT — always available (pure Python)
    try:
        agents['PIBT'] = PIBTAgent()
        print(f"  [agent] PIBT       (pure Python decentralised, 0 params)")
    except Exception as e:
        print(f"  [agent] PIBT FAILED: {e}")

    # CBS — always available (pure Python)
    try:
        agents['CBS'] = CBSAgent()
        print(f"  [agent] CBS        (pure Python centralised, 0 params)")
    except Exception as e:
        print(f"  [agent] CBS skipped: {e}")

    # LaCAM* — uses pre-built binary
    lacam_bin = getattr(args, 'lacam_bin', str(_ROOT / 'solvers/lacam0/build/main'))
    if Path(lacam_bin).exists():
        try:
            agents['LaCAM'] = LaCAMAgent(binary=lacam_bin, time_limit=30)
            print(f"  [agent] LaCAM*     (C++ binary: {lacam_bin})")
        except Exception as e:
            print(f"  [agent] LaCAM* skipped: {e}")
    else:
        print(f"  [agent] LaCAM* skipped (binary not found: {lacam_bin})")

    # DCC — optional
    if args.dcc_weights:
        algos_dir = _ROOT / 'algorithms'
        sys.path.insert(0, str(algos_dir))
        try:
            a = DCCAgent(args.dcc_weights, device=args.device)
            agents['DCC'] = a
            print(f"  [agent] DCC        params={a.n_params:,}  MACs={a.macs_per_step:,}")
        except Exception as e:
            print(f"  [agent] DCC skipped: {e}")

    return agents


def sweep(
    densities: List[float],
    n_agents_list: List[int],
    n_episodes: int,
    grid_size: int,
    max_steps: int,
    deadlock_patience: int,
    args,
    fpga_mhz: float = 100.0,
) -> List[dict]:
    results: List[dict] = []

    for n_agents in n_agents_list:
        print(f"\n{'='*60}\n  n_agents = {n_agents}\n{'='*60}")
        agents = build_agents(n_agents, args)
        if not agents:
            print("  No agents available — skipping.")
            continue

        for density in densities:
            print(f"\n  density = {density:.2f}")

            for name, agent in agents.items():
                csrs: List[float] = []
                isrs: List[float] = []
                act_times: List[float] = []
                fails = 0
                t_wall = time.perf_counter()

                for ep in range(n_episodes):
                    ep_rng = np.random.default_rng(ep * 1000 + int(density * 100))
                    try:
                        grid   = random_grid(grid_size, density, ep_rng)
                        grid   = ensure_min_free_cells(grid, n_agents * 2, ep_rng)
                        starts, goals = sample_starts_goals(grid, n_agents, ep_rng,
                                                            min_goal_l1=3)
                        ep_res = run_episode(agent, grid, starts, goals,
                                             max_steps=max_steps,
                                             act_times=act_times,
                                             deadlock_patience=deadlock_patience)
                        csrs.append(ep_res['CSR'])
                        isrs.append(ep_res['ISR'])
                    except Exception as exc:
                        fails += 1
                        if args.verbose:
                            print(f"      [warn] ep={ep} failed: {exc}")

                elapsed  = time.perf_counter() - t_wall
                mean_csr = float(np.mean(csrs)) if csrs else 0.0
                mean_isr = float(np.mean(isrs)) if isrs else 0.0
                mean_ms  = float(np.mean(act_times)) if act_times else 0.0

                n_params         = getattr(agent, 'n_params',                     None)
                n_params_trained = getattr(agent, 'n_params_trained',             None)
                n_params_per_ag  = getattr(agent, 'n_params_per_agent',           None)
                hardware_muls    = getattr(agent, 'hardware_muls',                None)
                muls_per_step    = getattr(agent, 'muls_per_step',                None)
                ticks            = getattr(agent, 'ticks_per_decision',           None)
                memory_mb        = getattr(agent, 'memory_mb',                    None)
                adds_total       = getattr(agent, 'mean_adds_per_step',           None)
                adds_per_agent   = getattr(agent, 'mean_adds_per_step_per_agent', None)
                algo_type        = getattr(agent, 'algo_type',          'unknown')
                scalability      = getattr(agent, 'scalability',        'unknown')
                muls_note        = getattr(agent, 'muls_note',          '')
                alu_profile      = getattr(agent, 'alu_cost_profile',   'unknown')
                platform         = getattr(agent, 'platform_label',     'CPU')
                seq_depth        = (agent.sequential_depth(n_agents)
                                    if hasattr(agent, 'sequential_depth') else 'N/A')
                # Architecture-independent latency
                hw_latency_us: Optional[float] = None
                if hasattr(agent, 'est_hw_latency_us'):
                    hw_latency_us = agent.est_hw_latency_us(fpga_mhz)

                # Per-step print line — FPGA latency for LSM, CPU time for others
                if hw_latency_us is not None:
                    latency_str = f"est_hw_latency={hw_latency_us:.1f}us (@{fpga_mhz:.0f}MHz FPGA)"
                else:
                    latency_str = f"cpu_wall_time={mean_ms:.3f}ms/step"
                hw_m = hardware_muls if hardware_muls is not None else muls_per_step
                muls_str = f"  muls={hw_m}" if hw_m is not None else ""
                if adds_per_agent is not None:
                    ops_label = "synaptic_adds/agent" if name == 'LSM-MAPF' else "ops/agent"
                    ops_str   = f"  {ops_label}={adds_per_agent:.0f}"
                else:
                    ops_str = ""
                ticks_str = f"  ticks={ticks}" if ticks is not None else ""
                print(
                    f"    {name:10s}  CSR={mean_csr:.3f}  ISR={mean_isr:.3f}"
                    f"  ({len(csrs)}/{n_episodes})  {elapsed:.1f}s"
                    f"  {latency_str}{ticks_str}{muls_str}{ops_str}"
                    f"  seq_depth={seq_depth}"
                )

                results.append({
                    'algorithm':                name,
                    'n_agents':                 n_agents,
                    'density':                  density,
                    'mean_CSR':                 mean_csr,
                    'mean_ISR':                 mean_isr,
                    'n_episodes':               len(csrs),
                    'n_failed':                 fails,
                    'runtime_s':                round(elapsed, 3),
                    'cpu_wall_time_ms':         round(mean_ms, 4),
                    'est_hw_latency_us':        round(hw_latency_us, 3) if hw_latency_us is not None else None,
                    'platform':                 platform,
                    'n_params':                 n_params,
                    'n_params_trained':         n_params_trained,
                    'n_params_per_agent':       n_params_per_ag,
                    'hardware_muls':            hardware_muls if hardware_muls is not None else muls_per_step,
                    'ticks_per_decision':       ticks,
                    'spike_adds_per_step':      round(adds_total, 1)     if adds_total     is not None else None,
                    'spike_adds_per_agent':     round(adds_per_agent, 1) if adds_per_agent is not None else None,
                    'sequential_depth':         seq_depth,
                    'alu_cost_profile':         alu_profile,
                    'memory_mb':                round(memory_mb, 3) if memory_mb else None,
                    'algo_type':                algo_type,
                    'scalability':              scalability,
                    'muls_note':                muls_note,
                })

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_results(results: List[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {out_dir / 'results.json'}")

    if results:
        with open(out_dir / 'summary.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"[saved] {out_dir / 'summary.csv'}")

    _print_ops_table(results)

    try:
        _plot_all(results, out_dir)
    except Exception as e:
        print(f"[warn] plotting skipped: {e}")


def _fmt(v, fmt=',') -> str:
    if isinstance(v, int):   return format(v, fmt)
    if isinstance(v, float): return f"{v:.1f}"
    return 'N/A'


def _print_ops_table(results: List[dict]) -> None:
    """Hardware-independent compute profile table.

    Columns:
      Platform       — FPGA vs CPU (LSM runs on FPGA; search solvers run on CPU)
      est_hw_lat_us  — FPGA decision latency in µs, or 'CPU' for software solvers
      Seq_Depth      — sequential pipeline depth per decision
                       LSM = 1 (all N agents pipelined); PIBT = N; CBS/LaCAM = O(b^d·N)
      ALU_Profile    — arithmetic unit requirement
                       'Adds-Only' → zero DSP multiplier blocks needed on FPGA
      HW_Muls        — hardware multiplications per decision step
      Ticks          — LIF clock cycles per FPGA decision (LSM only)
      Trained_Params — learned weights per agent (reservoir is fixed, not trained)
    """
    # Deduplicate by algorithm (take first n_agents entry for the profile)
    seen: Dict[str, dict] = {}
    for r in results:
        if r['algorithm'] not in seen:
            seen[r['algorithm']] = r

    W = 165
    print(f"\n{'='*W}")
    print("  Architecture-Independent Hardware Compute Profile")
    print(f"{'='*W}")

    # Header
    hdr = (f"{'Algorithm':10}  {'Platform':6}  {'est_hw_lat':>12}  {'Seq_Depth':>26}  "
           f"{'ALU_Profile':>28}  {'HW_Muls':>8}  {'Ticks':>6}  "
           f"{'Trained':>9}  {'per-agent':>9}  {'Ops/agent':>11}")
    print(hdr)
    print('-' * W)

    for name, r in seen.items():
        plat = r.get('platform', 'CPU')
        # Latency: FPGA µs for LSM, 'CPU' for software solvers
        lat_us = r.get('est_hw_latency_us')
        lat_str = f"{lat_us:.1f}µs" if lat_us is not None else "CPU"
        sd   = str(r.get('sequential_depth', 'N/A'))[:26]
        alu  = str(r.get('alu_cost_profile', ''))[:28]
        hm   = _fmt(r.get('hardware_muls'))
        tk   = _fmt(r.get('ticks_per_decision'))
        tp   = _fmt(r.get('n_params_trained'))
        pa   = _fmt(r.get('n_params_per_agent'))
        aa   = _fmt(r.get('spike_adds_per_agent'))
        print(f"{name:10}  {plat:6}  {lat_str:>12}  {sd:>26}  "
              f"{alu:>28}  {hm:>8}  {tk:>6}  {tp:>9}  {pa:>9}  {aa:>11}")

    print(f"{'='*W}")
    print()
    print("  Key:")
    print("    est_hw_lat  — FPGA: (ticks / MHz) × 1000 µs  |  CPU: measured wall-clock")
    print("    Seq_Depth   — number of sequential compute stages per decision")
    print("                  LSM = 1 (N agents pipelined in hardware, fire concurrently)")
    print("                  PIBT = N (agents processed one-by-one in priority order)")
    print("                  CBS/LaCAM = O(b^d·N) (conflict-tree/config search)")
    print("    ALU_Profile — 'Adds-Only' = zero multiplier DSP blocks needed on FPGA")
    print("                  'Standard ALU' = requires hardware multipliers")
    print("    HW_Muls     — 0 means no DSP blocks consumed on FPGA")
    print("    Ticks       — LIF clock cycles per decision (FPGA clock domain)")
    print("    Ops/agent   — LSM: synaptic spike accumulations (parallel adds)")
    print("                  PIBT: BFS lookups + comparisons  |  CBS/LaCAM: path-cost proxy")
    print()


def _plot_all(results: List[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    all_algos    = sorted(set(r['algorithm'] for r in results))
    all_n_agents = sorted(set(r['n_agents']  for r in results))
    all_densities = sorted(set(r['density']  for r in results))
    colors = {a: c for a, c in zip(all_algos, cm.tab10(np.linspace(0, 1, len(all_algos))))}

    def lw(a): return 2.5 if a == 'LSM-MAPF' else 1.5
    def ls(a): return '-'  if a == 'LSM-MAPF' else '--'

    # ── CSR vs density ────────────────────────────────────────────────────
    n_panels = len(all_n_agents)
    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(5 * n_panels, 4), sharey=True)
    if n_panels == 1: axes = [axes]
    for ax, n in zip(axes, all_n_agents):
        for a in all_algos:
            pts = sorted((r['density'], r['mean_CSR'])
                         for r in results if r['algorithm'] == a and r['n_agents'] == n)
            if not pts: continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, label=a, color=colors[a], lw=lw(a), ls=ls(a),
                    marker='o', markersize=4)
        ax.set_title(f'{n} agents'); ax.set_xlabel('Obstacle density')
        ax.set_ylim(-0.05, 1.05); ax.set_xticks(all_densities)
        ax.tick_params(axis='x', rotation=45); ax.grid(alpha=0.3)
    axes[0].set_ylabel('CSR (↑ better)')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=9)
    fig.suptitle('CSR vs Obstacle Density', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'csr_vs_density.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_dir / 'csr_vs_density.png'}")

    # ── Decision Latency vs N — the O(1) argument plot ───────────────────
    # Panel A: CPU wall-clock for non-LSM; FPGA est latency for LSM (µs→ms scale)
    # Panel B: Theoretical sequential depth scaling (flat=1 for LSM, linear/exp for others)
    fig, (ax_cpu, ax_hw) = plt.subplots(1, 2, figsize=(13, 4))

    for a in all_algos:
        pts_list = []
        for n in all_n_agents:
            rs = [r for r in results if r['algorithm'] == a and r['n_agents'] == n]
            if not rs:
                continue
            if a == 'LSM-MAPF':
                # Use FPGA estimated latency (µs), convert to ms for same scale
                vals = [r['est_hw_latency_us'] / 1000.0 for r in rs
                        if isinstance(r.get('est_hw_latency_us'), (int, float))]
            else:
                vals = [r['cpu_wall_time_ms'] for r in rs
                        if isinstance(r.get('cpu_wall_time_ms'), (int, float))]
            if vals:
                pts_list.append((n, float(np.mean(vals))))
        if not pts_list:
            continue
        xs, ys = zip(*sorted(pts_list))
        ax_cpu.plot(xs, ys, label=a, color=colors[a], lw=lw(a), ls=ls(a), marker='o')

    ax_cpu.set_xlabel('Number of agents')
    ax_cpu.set_ylabel('Decision latency (ms/step)')
    ax_cpu.set_title('A: Decision Latency vs Agent Count\n'
                     'LSM = FPGA est. latency; others = CPU wall-clock',
                     fontsize=10, fontweight='bold')
    ax_cpu.legend(fontsize=8); ax_cpu.grid(alpha=0.3)
    ax_cpu.annotate('LSM: FPGA est.\nOthers: CPU measured', xy=(0.55, 0.7),
                    xycoords='axes fraction', fontsize=7, color='gray')

    # Panel B: FPGA/HW theoretical latency — normalised to LSM=1 tick
    # LSM: constant (num_ticks clock cycles regardless of N)
    # Search-based: O(N) sequential agent processing (Von Neumann loop)
    # This is the theoretical curve, not measured
    lsm_ticks = None
    for r in results:
        if r['algorithm'] == 'LSM-MAPF' and isinstance(r.get('ticks_per_decision'), int):
            lsm_ticks = r['ticks_per_decision']
            break

    if lsm_ticks and len(all_n_agents) > 1:
        n_range = np.array(all_n_agents)
        for a in all_algos:
            if a == 'LSM-MAPF':
                ax_hw.plot(n_range, np.ones_like(n_range) * lsm_ticks,
                           label=f'LSM-MAPF (O(1) — {lsm_ticks} ticks)',
                           color=colors[a], lw=2.5, ls='-', marker='o')
            else:
                # Sequential per-agent processing: O(N) relative to LSM base
                ax_hw.plot(n_range, lsm_ticks * n_range / n_range[0],
                           label=f'{a} (O(N) — sequential agent loop)',
                           color=colors[a], lw=1.5, ls='--', marker='s')

        ax_hw.set_xlabel('Number of agents')
        ax_hw.set_ylabel(f'Decision latency (clock cycles, relative)')
        ax_hw.set_title('B: Theoretical HW Latency Scaling\n'
                        '(LSM flat O(1); search-based O(N) CPU cascade)',
                        fontsize=10, fontweight='bold')
        ax_hw.legend(fontsize=7); ax_hw.grid(alpha=0.3)
        ax_hw.annotate('O(1): no agent negotiation,\nasynchronous spike broadcast',
                       xy=(0.35, 0.1), xycoords='axes fraction', fontsize=7,
                       color=colors.get('LSM-MAPF', 'blue'))

    fig.suptitle('Decision Latency vs Agent Count — O(1) Argument', fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'latency_vs_nagents.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_dir / 'latency_vs_nagents.png'}")

    # ── Hardware multiplications bar ──────────────────────────────────────
    mul_data = []
    for a in all_algos:
        for r in results:
            if r['algorithm'] == a and r.get('hardware_muls') is not None:
                mul_data.append((a, int(r['hardware_muls'])))
                break
    if mul_data:
        algos_bar = [x[0] for x in mul_data]
        vals      = [x[1] for x in mul_data]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(algos_bar, vals, color=[colors.get(a, 'gray') for a in algos_bar], alpha=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f'{v:,}', ha='center', fontsize=8)
        ax.set_ylabel('Hardware multiplications per decision step')
        ax.set_xlabel('Algorithm')
        ax.set_title('FPGA Compute Cost: Multiplications/Step\n(0 = no DSP blocks needed)',
                     fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / 'hw_muls_per_step.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[saved] {out_dir / 'hw_muls_per_step.png'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Native MAPF comparison benchmark')
    parser.add_argument('--densities',   nargs='+', type=float,
                        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument('--n-agents',    nargs='+', type=int,   default=[4, 8, 16])
    parser.add_argument('--episodes',    type=int,   default=20)
    parser.add_argument('--grid-size',   type=int,   default=32)
    parser.add_argument('--max-steps',   type=int,   default=256)
    parser.add_argument('--deadlock-patience', type=int, default=100,
                        help='Steps with no movement before declaring deadlock (default 100)')
    parser.add_argument('--num-ticks',   type=int,   default=10,
                        help='LIF simulation ticks for SwarmLSM (= FPGA clock cycles per decision)')
    parser.add_argument('--target-fpga-mhz', type=float, default=100.0,
                        help='Target FPGA clock frequency for LSM latency estimate (default 100 MHz)')
    parser.add_argument('--device',      type=str,   default='cpu')
    parser.add_argument('--lsm-checkpoint', type=str,
                        default='checkpoints/eval_ablation/phase1_cortex.pt')
    parser.add_argument('--lacam-bin',   type=str,
                        default=str(_ROOT / 'solvers/lacam0/build/main'),
                        help='Path to lacam0 binary')
    parser.add_argument('--dcc-weights', type=str, default=None,
                        help='Path to DCC weights dir (optional)')
    parser.add_argument('--out-dir',     type=str, default='experiments/benchmark')
    parser.add_argument('--verbose',     action='store_true')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    print(f"\n{'='*60}")
    print(f"  Native MAPF Benchmark")
    print(f"  densities  : {args.densities}")
    print(f"  n_agents   : {args.n_agents}")
    print(f"  episodes   : {args.episodes}")
    print(f"  grid       : {args.grid_size}x{args.grid_size}")
    print(f"  max_steps  : {args.max_steps}  (deadlock_patience={args.deadlock_patience})")
    print(f"  num_ticks  : {args.num_ticks}  (LSM FPGA clock cycles/decision)")
    print(f"  fpga_mhz   : {args.target_fpga_mhz} MHz  →  LSM latency = "
          f"{(args.num_ticks / args.target_fpga_mhz) * 1000:.1f} µs/decision")
    print(f"  device     : {args.device}")
    print(f"  out_dir    : {out_dir}")
    print(f"{'='*60}\n")

    results = sweep(
        densities=args.densities,
        n_agents_list=args.n_agents,
        n_episodes=args.episodes,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        deadlock_patience=args.deadlock_patience,
        args=args,
        fpga_mhz=args.target_fpga_mhz,
    )

    save_results(results, out_dir)
    print(f"\n{'='*60}\n  Benchmark complete.\n{'='*60}\n")


if __name__ == '__main__':
    main()
