"""
test_hh_smoke.py — HH-LTS Neuron Smoke Test (with retraining)
==============================================================
Trains action_weights for an HH-LTS reservoir from expert trajectories,
then compares LIF (pre-trained checkpoint) vs HH-retrained side-by-side.

Training uses the same pipeline as train_swarm_lsm.py:
  SwarmTrainer.collect_states()  →  reservoir states from HH dynamics
  SwarmTrainer.train_ridge()     →  fit action_weights to expert actions
  trainer.save()                 →  saves HH checkpoint

Evaluation uses the same random-map episode loop as ablation_eval.py.

Usage:
    python test_hh_smoke.py                        # retrain if no HH ckpt
    python test_hh_smoke.py --retrain              # always retrain HH
    python test_hh_smoke.py --skip-retrain         # eval only (needs existing HH ckpt)
    python test_hh_smoke.py --train-episodes 100   # fewer eps → faster
    python test_hh_smoke.py --episodes 5 --n-agents 5 --density 0.1
"""

from __future__ import annotations

import argparse
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'models'))

from ablation_eval import (
    random_grid,
    ensure_min_free_cells,
    sample_starts_goals,
    load_swarm_checkpoint,
    simulate_episode,
)
from models.swarm_lsm import SwarmLSM, SwarmTrainer, HHLTSParams
from data_loader import SNNDataset


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_lif_network(n_agents: int, checkpoint: str, device: str) -> SwarmLSM:
    """Load pre-trained LIF network."""
    net = SwarmLSM(num_agents=n_agents, neuron_type='lif')
    load_swarm_checkpoint(net, checkpoint, device)
    net.to(device).eval()
    return net


def build_hh_network(
    n_agents: int,
    device: str,
    hh_i_scale: float = 50.0,
) -> SwarmLSM:
    """Create a fresh (untrained) HH-LTS network."""
    net = SwarmLSM(num_agents=n_agents, neuron_type='hh')
    # Patch I_scale on every HH neuron
    from models.swarm_lsm import HHLTSNeuron
    hh_params = HHLTSParams(I_scale=hh_i_scale)
    for m in net.modules():
        if isinstance(m, HHLTSNeuron):
            m.hp = hh_params
    net.to(device).eval()
    return net


# ---------------------------------------------------------------------------
# Train HH action_weights from expert trajectories
# ---------------------------------------------------------------------------

def _find_dataset_roots(config_path: str) -> List[str]:
    """
    Return train dataset root directories that actually exist on disk.
    Falls back to scanning the dataset/ folder for any dir with a train/
    subdirectory containing case folders.
    """
    existing = []

    # Try config-specified roots first
    try:
        cfg = _load_config(config_path)
        for root in cfg.get('train', {}).get('root_dirs', []):
            p = _ROOT / root
            if (p / 'train').is_dir() and any((p / 'train').iterdir()):
                existing.append(str(p))
    except Exception:
        pass

    if existing:
        return existing

    # Auto-scan dataset/ for any populated directories
    ds_dir = _ROOT / 'dataset'
    if ds_dir.is_dir():
        for d in sorted(ds_dir.iterdir()):
            train_dir = d / 'train'
            if train_dir.is_dir() and any(train_dir.iterdir()):
                existing.append(str(d))

    return existing


def train_hh(
    net: SwarmLSM,
    config_path: str,
    *,
    max_episodes: int = 200,
    num_ticks: int = 10,
    ridge_alpha: float = 0.75,
    device: str = 'cpu',
    save_path: str = 'checkpoints/hh_lts/swarm_hh.pt',
) -> str:
    """
    Train action_weights for the HH reservoir using ridge regression on
    expert trajectory states — identical to the 'ridge' path in
    train_swarm_lsm.py.

    Args:
        net:           Fresh HH SwarmLSM (reservoir frozen, action_weights free).
        config_path:   Path to config YAML (for dataset root_dirs).
        max_episodes:  Max expert episodes to collect states from.
        num_ticks:     Ticks per decision during state collection.
        ridge_alpha:   Ridge regularisation strength.
        device:        'cpu' or 'cuda'.
        save_path:     Where to write the trained HH checkpoint.

    Returns:
        save_path  (so caller can pass it straight to build_hh_network loader).
    """
    print('\n' + '=' * 65)
    print('  HH-LTS TRAINING  (ridge regression on HH reservoir states)')
    print('=' * 65)
    print(f'  max_episodes : {max_episodes}')
    print(f'  num_ticks    : {num_ticks}')
    print(f'  ridge_alpha  : {ridge_alpha}')
    print(f'  device       : {device}')

    # --- find datasets ---
    roots = _find_dataset_roots(config_path)
    if not roots:
        print(f'  [error] No dataset directories found. Run data_generation/main_data.py first.')
        sys.exit(1)
    print(f'\n  Dataset roots: {roots}')

    # Build a minimal config dict for SNNDataset
    ds_cfg = {'train': {'root_dirs': roots}, 'valid': {'root_dirs': roots}}

    # --- dataset ---
    print('  Loading training dataset …')
    try:
        train_ds = SNNDataset(ds_cfg, 'train')
    except RuntimeError as e:
        print(f'  [error] {e}')
        sys.exit(1)
    print(f'  Loaded {len(train_ds)} episodes.')

    # --- trainer (freezes reservoir, keeps action_weights trainable) ---
    trainer = SwarmTrainer(net, device=device)

    # --- collect HH reservoir states from expert trajectories ---
    X, Y = trainer.collect_states(train_ds, max_episodes=max_episodes, num_ticks=num_ticks)

    # --- fit readout weights with ridge ---
    trainer.train_ridge(X, Y, alpha=ridge_alpha)

    # --- save ---
    save_path = str(Path(save_path))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(save_path)

    return save_path


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_one(
    net: SwarmLSM,
    grid: np.ndarray,
    starts: torch.Tensor,
    goals: torch.Tensor,
    *,
    fov_size: int = 7,
    num_ticks: int = 10,
    max_steps: int = 256,
    device: str = 'cpu',
) -> Dict:
    from ablation_eval import EpisodeStats
    stats: EpisodeStats = simulate_episode(
        network=net,
        grid=grid,
        starts=starts,
        goals=goals,
        fov_size=fov_size,
        num_ticks=num_ticks,
        max_steps=max_steps,
        deadlock_patience=30,
        device=device,
    )
    n = starts.shape[0]
    return {
        'success_all': stats.success_all,
        'reached':     stats.reached_count,
        'n_agents':    n,
        'makespan':    stats.makespan,
        'isr':         stats.reached_count / max(n, 1),
    }


# ---------------------------------------------------------------------------
# Evaluation sweep (with per-episode logging)
# ---------------------------------------------------------------------------

def sweep(
    net: SwarmLSM,
    label: str,
    n_agents: int,
    density: float,
    n_episodes: int,
    *,
    grid_size: int = 28,
    fov_size: int = 7,
    num_ticks: int = 10,
    max_steps: int = 256,
    device: str = 'cpu',
    seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)
    successes, isrs, makespans = [], [], []

    t0 = time.perf_counter()
    for ep in range(n_episodes):
        grid = random_grid(grid_size, density, rng)
        grid = ensure_min_free_cells(grid, n_agents * 4, rng)
        try:
            starts, goals = sample_starts_goals(grid, n_agents, rng, min_goal_l1=4)
        except ValueError:
            print(f'    ep {ep+1:>2}/{n_episodes}  [skip — map too crowded]')
            continue

        t_ep = time.perf_counter()
        res = run_one(
            net, grid, starts, goals,
            fov_size=fov_size, num_ticks=num_ticks,
            max_steps=max_steps, device=device,
        )
        ep_s = time.perf_counter() - t_ep

        status = '✓' if res['success_all'] else '✗'
        print(f'    ep {ep+1:>2}/{n_episodes}  {status}  '
              f'reached={res["reached"]}/{res["n_agents"]}  '
              f'makespan={res["makespan"]:>4}  ({ep_s:.1f}s)')

        successes.append(float(res['success_all']))
        isrs.append(res['isr'])
        makespans.append(res['makespan'])

    elapsed = time.perf_counter() - t0
    n_done  = len(successes)
    return {
        'label':         label,
        'csr':           float(np.mean(successes)) if n_done else 0.0,
        'isr':           float(np.mean(isrs))      if n_done else 0.0,
        'mean_makespan': float(np.mean(makespans)) if n_done else 0.0,
        'n_success':     int(sum(successes)),
        'n_episodes':    n_done,
        'elapsed_s':     round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------

def _bar(val: float, width: int = 20) -> str:
    filled = round(val * width)
    return '█' * filled + '░' * (width - filled)


def print_results(rows: List[Dict]) -> None:
    W = 92
    print()
    print('=' * W)
    print('  HH-LTS (retrained) vs LIF — SMOKE TEST RESULTS')
    print('=' * W)
    hdr = (f"  {'Variant':<26}  {'CSR':>6}  {'ISR':>6}  {'Makespan':>9}"
           f"  {'Succ/N':>8}  {'Time':>7}  {'CSR bar'}")
    print(hdr)
    print('-' * W)
    for r in rows:
        bar  = _bar(r['csr'])
        frac = f"{r['n_success']}/{r['n_episodes']}"
        print(f"  {r['label']:<26}  {r['csr']:>6.3f}  {r['isr']:>6.3f}"
              f"  {r['mean_makespan']:>9.1f}  {frac:>8}"
              f"  {r['elapsed_s']:>6.1f}s  {bar}")
    print('=' * W)
    print()

    if len(rows) >= 2:
        lif_row = next((r for r in rows if 'LIF' in r['label']), None)
        hh_row  = next((r for r in rows if 'HH'  in r['label']), None)
        if lif_row and hh_row:
            dcsr = hh_row['csr'] - lif_row['csr']
            disr = hh_row['isr'] - lif_row['isr']
            dms  = hh_row['mean_makespan'] - lif_row['mean_makespan']
            print(f"  Δ CSR  (HH − LIF) = {dcsr:+.3f}"
                  f"  ({'✓ HH better' if dcsr >= 0 else '✗ LIF better'})")
            print(f"  Δ ISR  (HH − LIF) = {disr:+.3f}")
            print(f"  Δ makespan         = {dms:+.1f} steps")
            print()
            if abs(dcsr) < 0.05:
                print('  → CSR within 5 pp: HH-LTS is a viable drop-in reservoir.')
            elif dcsr >= 0.05:
                print('  → HH-LTS outperforms LIF — T-Ca burst dynamics help!')
            else:
                print('  → LIF outperforms HH — T-Ca dynamics may need more ticks or tuning.')
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='HH-LTS vs LIF smoke test with retraining')
    parser.add_argument('--lif-checkpoint', type=str,
                        default=str(_ROOT / 'checkpoints/eval_ablation/swarm1.pt'))
    parser.add_argument('--hh-checkpoint', type=str,
                        default=str(_ROOT / 'checkpoints/hh_lts/swarm_hh.pt'),
                        help='Where to save/load the retrained HH checkpoint')
    parser.add_argument('--config', type=str,
                        default=str(_ROOT / 'configs/config_swarm.yaml'))
    # Training args
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain HH even if checkpoint exists')
    parser.add_argument('--skip-retrain', action='store_true',
                        help='Skip training; requires --hh-checkpoint to exist')
    parser.add_argument('--train-episodes', type=int, default=200,
                        help='Expert episodes to use for HH state collection (default 200)')
    parser.add_argument('--train-ticks',    type=int, default=10,
                        help='Ticks during state collection (default 10)')
    parser.add_argument('--ridge-alpha',    type=float, default=0.75)
    parser.add_argument('--hh-i-scale',     type=float, default=50.0,
                        help='HHLTSParams.I_scale (μA/cm² per unit input, default 50.0)')
    # Eval args
    parser.add_argument('--n-agents',   type=int,   default=5)
    parser.add_argument('--density',    type=float, default=0.1)
    parser.add_argument('--episodes',   type=int,   default=20)
    parser.add_argument('--grid-size',  type=int,   default=28)
    parser.add_argument('--num-ticks',  type=int,   default=10)
    parser.add_argument('--max-steps',  type=int,   default=256)
    parser.add_argument('--device',     type=str,   default='cpu')
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--lif-only',   action='store_true')
    parser.add_argument('--hh-only',    action='store_true')
    args = parser.parse_args()

    # ── resolve LIF checkpoint ───────────────────────────────────────────────
    lif_ckpt = args.lif_checkpoint
    if not Path(lif_ckpt).exists():
        alt = str(_ROOT / 'checkpoints/lsm/phase1_cortex.pt')
        if Path(alt).exists():
            print(f'[warn] LIF ckpt {lif_ckpt} not found → {alt}')
            lif_ckpt = alt
        elif args.hh_only:
            pass   # don't need it
        else:
            print(f'[error] LIF checkpoint not found: {lif_ckpt}'); sys.exit(1)

    print()
    print('HH-LTS Smoke Test  (with retraining)')
    print(f'  lif_checkpoint : {lif_ckpt}')
    print(f'  hh_checkpoint  : {args.hh_checkpoint}')
    print(f'  train_episodes : {args.train_episodes}  (HH state collection)')
    print(f'  hh_i_scale     : {args.hh_i_scale} μA/cm²')
    print(f'  eval episodes  : {args.episodes}  |  n_agents={args.n_agents}'
          f'  density={args.density}  ticks={args.num_ticks}')

    eval_kw = dict(
        n_agents=args.n_agents,
        density=args.density,
        n_episodes=args.episodes,
        grid_size=args.grid_size,
        fov_size=7,
        num_ticks=args.num_ticks,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
    )

    results: List[Dict] = []

    # ── LIF baseline ─────────────────────────────────────────────────────────
    if not args.hh_only:
        print('\n── LIF baseline ──────────────────────────────────────────────')
        lif_net = build_lif_network(args.n_agents, lif_ckpt, args.device)
        print(f'  params: {sum(p.numel() for p in lif_net.parameters()):,}')
        r = sweep(lif_net, 'LIF (pre-trained)', **eval_kw)
        results.append(r)

    # ── HH-LTS: train then eval ───────────────────────────────────────────────
    if not args.lif_only:
        hh_ckpt  = args.hh_checkpoint
        need_train = args.retrain or not Path(hh_ckpt).exists()

        if args.skip_retrain:
            if not Path(hh_ckpt).exists():
                print(f'[error] --skip-retrain set but {hh_ckpt} does not exist.'); sys.exit(1)
            need_train = False

        if need_train:
            print('\n── HH-LTS training ──────────────────────────────────────────')
            hh_net = build_hh_network(args.n_agents, args.device, args.hh_i_scale)
            hh_ckpt = train_hh(
                hh_net,
                config_path=args.config,
                max_episodes=args.train_episodes,
                num_ticks=args.train_ticks,
                ridge_alpha=args.ridge_alpha,
                device=args.device,
                save_path=hh_ckpt,
            )
        else:
            print(f'\n── HH-LTS  (loading existing checkpoint: {hh_ckpt}) ─────────')
            hh_net = build_hh_network(args.n_agents, args.device, args.hh_i_scale)
            load_swarm_checkpoint(hh_net, hh_ckpt, args.device)
            hh_net.to(args.device).eval()

        n_p  = sum(p.numel() for p in hh_net.parameters())
        n_b  = sum(b.numel() for b in hh_net.buffers())
        print(f'\n── HH-LTS eval ───────────────────────────────────────────────')
        print(f'  params: {n_p:,}  buffers (HH state): {n_b:,}')
        r = sweep(hh_net, f'HH-LTS retrained (I×{args.hh_i_scale:.0f})', **eval_kw)
        results.append(r)

    print_results(results)


if __name__ == '__main__':
    main()
