"""
Training loop for the Contractive Linearizer planning experiment.

- Pre-generates 5000 maps of size 10x10
- Trains each model for 8000 steps, batch size 32
- Loss: L_bellman + 0.5 * L_fp + 0.1 * L_multi
- Saves checkpoints to contractive_rl/planning/checkpoints/
- Logs train loss every 200 steps
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import torch
import torch.optim as optim

# Single-threaded is faster on CPU for small tensors (avoids thread-spawn overhead)
torch.set_num_threads(1)

# Ensure imports work regardless of how this script is invoked.
# Insert _this_dir FIRST to shadow contractive_rl/gridworld.py and models.py.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)

# Remove any existing entries for these dirs to avoid duplicates, then prepend
for _p in [_this_dir, _parent_dir]:
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, _parent_dir)   # shared/ lives here
sys.path.insert(0, _this_dir)     # must come first to shadow contractive_rl/*.py

from gridworld import generate_batch, GAMMA
from models import (
    ContractiveLinearizer,
    VINBaseline,
    UnconstrainedLinearizer,
    IterativeMLPBaseline,
    FullContractiveLinearizer,
    make_contractive,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = os.path.join(_this_dir, "checkpoints")
SEED = 42
BATCH_SIZE = 32
NUM_STEPS = 8000
LR = 3e-4
LAMBDA_FP = 0.5
LAMBDA_MULTI = 0.1
K_MULTI = 5
POOL_SIZE = 5000
GRID_SIZE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    pool_size: int = POOL_SIZE,
    grid_size: int = GRID_SIZE,
    seed: int = SEED,
) -> tuple:
    """
    Pre-generate a pool of maps with V* and Bellman targets.

    Returns:
        (maps, V_star, V_init, bellman_V) CPU tensors of shape (pool_size, ...)
    """
    print(f"  Pre-generating {pool_size} maps (grid={grid_size}x{grid_size})...")
    t0 = time.time()

    chunk_size = 200
    all_maps, all_V_star, all_V_inits, all_bellman = [], [], [], []

    remaining = pool_size
    current_seed = seed
    while remaining > 0:
        bs = min(chunk_size, remaining)
        maps, V_star, V_init, bellman_V = generate_batch(
            n_maps=bs,
            grid_size=grid_size,
            gamma=GAMMA,
            seed=current_seed,
        )
        all_maps.append(maps)
        all_V_star.append(V_star)
        all_V_inits.append(V_init)
        all_bellman.append(bellman_V)
        remaining -= bs
        current_seed += bs * 1000

    maps = torch.cat(all_maps, dim=0)
    V_star = torch.cat(all_V_star, dim=0)
    V_init = torch.cat(all_V_inits, dim=0)
    bellman_V = torch.cat(all_bellman, dim=0)

    elapsed = time.time() - t0
    print(f"  Done: {pool_size} maps in {elapsed:.1f}s. "
          f"maps={maps.shape}, V_star={V_star.shape}")
    return maps, V_star, V_init, bellman_V


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_model(
    model: torch.nn.Module,
    model_name: str,
    dataset: tuple,
    num_steps: int = NUM_STEPS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    lambda_fp: float = LAMBDA_FP,
    lambda_multi: float = LAMBDA_MULTI,
    k_multi: int = K_MULTI,
    seed: int = SEED,
    device: torch.device = DEVICE,
    verbose: bool = True,
) -> list:
    """
    Train a single model.

    Loss:
        L_bellman = ||T(V_init, c) - Bellman(V_init, c)||^2
        L_fp      = ||T(V*, c) - V*||^2
        L_multi   = ||T^5(V0, c) - V*||^2
        total = L_bellman + lambda_fp * L_fp + lambda_multi * L_multi

    Returns:
        losses: list of (step, total_loss, bellman_loss, fp_loss, multi_loss)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    maps_pool, V_star_pool, V_init_pool, bellman_V_pool = dataset
    pool_size = maps_pool.shape[0]

    losses = []
    t0 = time.time()
    rng = np.random.RandomState(seed)

    for step in range(num_steps):
        # Sample random batch from pool
        idx = rng.choice(pool_size, size=batch_size, replace=False)
        idx_t = torch.from_numpy(idx).long()

        maps = maps_pool[idx_t].to(device)
        V_star = V_star_pool[idx_t].to(device)
        V_init = V_init_pool[idx_t].to(device)
        bellman_V = bellman_V_pool[idx_t].to(device)

        optimizer.zero_grad()

        # L_bellman: T(V_init, c) should match Bellman backup
        V_pred = model(V_init, maps)
        loss_bellman = ((V_pred - bellman_V) ** 2).mean()

        # L_fp: T(V*, c) should equal V* (fixed point)
        V_star_pred = model(V_star, maps)
        loss_fp = ((V_star_pred - V_star) ** 2).mean()

        # L_multi: 5-step unroll from V_init should approach V*
        x_k = V_init.detach()
        for _ in range(k_multi):
            x_k = model(x_k, maps)
        loss_multi = ((x_k - V_star) ** 2).mean()

        loss = loss_bellman + lambda_fp * loss_fp + lambda_multi * loss_multi
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        losses.append((
            step,
            loss.item(),
            loss_bellman.item(),
            loss_fp.item(),
            loss_multi.item(),
        ))

        if verbose and (step + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{model_name}] step {step+1}/{num_steps} | "
                f"loss={loss.item():.4f} bel={loss_bellman.item():.4f} "
                f"fp={loss_fp.item():.4f} multi={loss_multi.item():.4f} "
                f"t={elapsed:.0f}s"
            )

    # Save checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "losses": losses,
            "config": {
                "num_steps": num_steps,
                "batch_size": batch_size,
                "lr": lr,
                "lambda_fp": lambda_fp,
                "lambda_multi": lambda_multi,
                "k_multi": k_multi,
                "seed": seed,
            },
        },
        ckpt_path,
    )

    if verbose:
        elapsed = time.time() - t0
        final_loss = losses[-1][1]
        print(
            f"  [{model_name}] DONE in {elapsed:.1f}s | "
            f"final loss={final_loss:.4f} | saved: {ckpt_path}"
        )

    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train planning models")
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lambda_fp", type=float, default=LAMBDA_FP)
    parser.add_argument("--k_multi", type=int, default=K_MULTI)
    parser.add_argument("--pool_size", type=int, default=POOL_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--grid_size", type=int, default=GRID_SIZE)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_coupling_layers", type=int, default=4)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "contractive",
            "unconstrained",
            "iterative_mlp",
            "vin",
            "contractive_05",
            "contractive_09",
            "contractive_099",
        ],
    )
    args = parser.parse_args()

    state_dim = args.grid_size * args.grid_size
    ckpt_suffix = f"_{args.grid_size}x{args.grid_size}" if args.grid_size != 10 else ""

    print(f"Device: {DEVICE}")
    print(f"Training {args.steps} steps, batch_size={args.batch_size}, "
          f"grid={args.grid_size}x{args.grid_size}, state_dim={state_dim}")

    # Pre-generate dataset
    print("\n=== Generating dataset ===")
    dataset = generate_dataset(pool_size=args.pool_size, grid_size=args.grid_size,
                               seed=args.seed)

    model_registry = {
        "contractive": ContractiveLinearizer(state_dim=state_dim, spectral_bound=0.9,
                                             n_coupling_layers=args.n_coupling_layers,
                                             hidden_dim=args.hidden_dim),
        "unconstrained": UnconstrainedLinearizer(state_dim=state_dim),
        "iterative_mlp": IterativeMLPBaseline(state_dim=state_dim),
        "vin": VINBaseline(grid_size=args.grid_size),
        "contractive_05": make_contractive(state_dim=state_dim, spectral_bound=0.5),
        "contractive_09": make_contractive(state_dim=state_dim, spectral_bound=0.9),
        "contractive_099": make_contractive(state_dim=state_dim, spectral_bound=0.99),
        "contractive_full": FullContractiveLinearizer(state_dim=state_dim,
                                                      spectral_bound=0.9,
                                                      n_coupling_layers=args.n_coupling_layers,
                                                      hidden_dim=args.hidden_dim),
    }

    for name in args.models:
        if name not in model_registry:
            print(f"Unknown model: {name}, skipping")
            continue
        ckpt_name = name + ckpt_suffix
        print(f"\n=== Training {ckpt_name} ===")
        model = model_registry[name]
        train_model(
            model=model,
            model_name=ckpt_name,
            dataset=dataset,
            num_steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_fp=args.lambda_fp,
            k_multi=args.k_multi,
            seed=args.seed,
        )

    print("\n=== All training complete ===")


if __name__ == "__main__":
    main()
