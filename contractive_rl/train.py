"""
Training loop for Contractive Linearizer RL experiments.

- Pre-generates a pool of gridworld maps for efficiency
- Loss: L = ||T(V, c) - Bellman(V, c)||^2 + lambda * ||T(V*, c) - V*||^2
- Trains each model separately
- Saves checkpoints to contractive_rl/checkpoints/
"""

import os
import time
import random
import argparse

import numpy as np
import torch
import torch.optim as optim

from gridworld import generate_batch, GRID_SIZE
from models import (
    ContractiveLinearizer,
    UnconstrainedLinearizer,
    MLPBaseline,
    VINBaseline,
    make_contractive_with_scale,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
SEED = 42
BATCH_SIZE = 32
NUM_STEPS = 5000
LR = 3e-4
LAMBDA_FP = 2.0
LAMBDA_MULTI = 0.5   # multi-step consistency loss weight
K_MULTI = 5          # unroll K steps for multi-step loss
GAMMA = 0.99
POOL_SIZE = 2000     # pre-generate this many maps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(pool_size=POOL_SIZE, batch_size=64, seed=SEED):
    """
    Pre-generate a large pool of maps and value functions.
    Returns tensors of shape (pool_size, ...).
    """
    print(f"  Pre-generating {pool_size} maps (this may take ~{pool_size*0.015:.0f}s)...")
    t0 = time.time()
    all_maps = []
    all_V_star = []
    all_V_init = []
    all_bellman_V = []

    remaining = pool_size
    current_seed = seed
    while remaining > 0:
        bs = min(batch_size, remaining)
        maps, V_star, V_init, bellman_V = generate_batch(
            batch_size=bs,
            grid_size=GRID_SIZE,
            gamma=GAMMA,
            seed=current_seed,
        )
        all_maps.append(maps)
        all_V_star.append(V_star)
        all_V_init.append(V_init)
        all_bellman_V.append(bellman_V)
        remaining -= bs
        current_seed += bs * 1000

    maps = torch.cat(all_maps, dim=0)
    V_star = torch.cat(all_V_star, dim=0)
    V_init = torch.cat(all_V_init, dim=0)
    bellman_V = torch.cat(all_bellman_V, dim=0)

    elapsed = time.time() - t0
    print(f"  Dataset ready: {pool_size} maps in {elapsed:.1f}s. "
          f"Maps shape: {maps.shape}, V_star: {V_star.shape}")
    return maps, V_star, V_init, bellman_V


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


def train_model(model, model_name, dataset, num_steps=NUM_STEPS, batch_size=BATCH_SIZE,
                lr=LR, lambda_fp=LAMBDA_FP, lambda_multi=LAMBDA_MULTI, k_multi=K_MULTI,
                seed=SEED, device=DEVICE, verbose=True):
    """
    Train a single model using pre-generated dataset.

    Args:
        model: nn.Module
        model_name: str
        dataset: (maps, V_star, V_init, bellman_V) tensors on CPU
        ...

    Returns:
        losses: list of (step, total_loss, bellman_loss, fp_loss)
    """
    rng_seed = seed
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    maps_pool, V_star_pool, V_init_pool, bellman_V_pool = dataset
    pool_size = maps_pool.shape[0]

    losses = []
    t0 = time.time()

    rng = np.random.RandomState(rng_seed)

    for step in range(num_steps):
        # Random batch from pool
        idx = rng.choice(pool_size, size=batch_size, replace=False)
        idx_t = torch.from_numpy(idx).long()

        maps = maps_pool[idx_t].to(device)
        V_star = V_star_pool[idx_t].to(device)
        V_init = V_init_pool[idx_t].to(device)
        bellman_V = bellman_V_pool[idx_t].to(device)

        optimizer.zero_grad()

        # Bellman consistency loss: T(V_init, c) ≈ Bellman(V_init, c)
        V_pred = model(V_init, maps)
        loss_bellman = ((V_pred - bellman_V) ** 2).mean()

        # Fixed point loss: T(V*, c) ≈ V*
        V_star_pred = model(V_star, maps)
        loss_fp = ((V_star_pred - V_star) ** 2).mean()

        # Multi-step consistency: T^K(V0, c) ≈ V*
        x_k = V_init.detach()
        for _ in range(k_multi):
            x_k = model(x_k, maps)
        loss_multi = ((x_k - V_star) ** 2).mean()

        loss = loss_bellman + lambda_fp * loss_fp + lambda_multi * loss_multi
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        losses.append((step, loss.item(), loss_bellman.item(), loss_fp.item()))

        if verbose and (step + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{model_name}] step {step+1}/{num_steps} | "
                  f"loss={loss.item():.4f} | "
                  f"bel={loss_bellman.item():.4f} | "
                  f"fp={loss_fp.item():.4f} | "
                  f"multi={loss_multi.item():.4f} | "
                  f"elapsed={elapsed:.1f}s")

    # Save checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "losses": losses,
        "config": {
            "num_steps": num_steps,
            "batch_size": batch_size,
            "lr": lr,
            "lambda_fp": lambda_fp,
            "seed": seed,
        }
    }, ckpt_path)

    if verbose:
        elapsed = time.time() - t0
        final_loss = losses[-1][1]
        print(f"  [{model_name}] Training complete in {elapsed:.1f}s | final loss={final_loss:.4f}")
        print(f"  Saved: {ckpt_path}")

    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    parser.add_argument("--lambda_fp", type=float, default=LAMBDA_FP)
    parser.add_argument("--lambda_multi", type=float, default=LAMBDA_MULTI)
    parser.add_argument("--k_multi", type=int, default=K_MULTI)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pool_size", type=int, default=POOL_SIZE)
    parser.add_argument("--models", nargs="+",
                        default=["contractive", "unconstrained", "mlp", "vin",
                                 "contractive_05", "contractive_09", "contractive_099"])
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Training {args.steps} steps, batch_size={args.batch_size}")
    print()

    # Pre-generate dataset
    print("=== Generating dataset ===")
    dataset = generate_dataset(pool_size=args.pool_size, seed=args.seed)
    print()

    model_registry = {
        "contractive": ContractiveLinearizer(spectral_scale=0.99),
        "unconstrained": UnconstrainedLinearizer(),
        "mlp": MLPBaseline(),
        "vin": VINBaseline(),
        "contractive_05": make_contractive_with_scale(0.5),
        "contractive_09": make_contractive_with_scale(0.9),
        "contractive_099": make_contractive_with_scale(0.99),
    }

    for name in args.models:
        if name not in model_registry:
            print(f"Unknown model: {name}, skipping")
            continue

        print(f"\n=== Training {name} ===")
        model = model_registry[name]
        train_model(
            model=model,
            model_name=name,
            dataset=dataset,
            num_steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            lambda_fp=LAMBDA_FP,
            lambda_multi=LAMBDA_MULTI,
            k_multi=K_MULTI,
        )

    print("\n=== All training complete ===")


if __name__ == "__main__":
    main()
