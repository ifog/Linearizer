"""
Evaluation script for Contractive Linearizer planning experiments.

Experiments:
  A — Convergence curves (all 4 models, RMSE vs K=0..50)
  B — Fast iteration vs loop (ContractiveLinearizer: MSE should be ~0)
  C — Planning quality (greedy policy success rate)
  D — Generalization (10x10 -> 20x20)
  E — Ablation: spectral bound in {0.5, 0.9, 0.99}

Results saved to contractive_rl/planning/results/
"""

import os
import sys
import json
import time
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Single-threaded is faster on CPU for small tensors (avoids thread-spawn overhead)
torch.set_num_threads(1)

_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)

# Insert _this_dir FIRST to shadow contractive_rl/gridworld.py and models.py
for _p in [_this_dir, _parent_dir]:
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, _parent_dir)
sys.path.insert(0, _this_dir)

from gridworld import (
    make_random_map,
    compute_V_star,
    encode_map,
    get_greedy_policy,
    simulate_episode,
    GAMMA,
    OBSTACLE_DENSITY,
    _build_transition_table,
    bellman_backup,
)
from models import (
    ContractiveLinearizer,
    VINBaseline,
    UnconstrainedLinearizer,
    IterativeMLPBaseline,
    make_contractive,
    STATE_DIM,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = os.path.join(_this_dir, "checkpoints")
RESULTS_DIR = os.path.join(_this_dir, "results")
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_GRID_SIZE = 10
TEST_GRID_SIZE_SMALL = 10
TEST_GRID_SIZE_LARGE = 20
TEST_GRID_SIZE_XL = 50

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_model(model_instance: torch.nn.Module, checkpoint_name: str,
               device: torch.device = DEVICE) -> torch.nn.Module:
    """Load a model from checkpoint. Raises FileNotFoundError or RuntimeError on failure."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_instance.load_state_dict(ckpt["model_state_dict"])  # raises RuntimeError on mismatch
    model_instance.to(device)
    model_instance.eval()
    return model_instance


def generate_test_maps(n_maps: int, grid_size: int = TEST_GRID_SIZE_SMALL,
                       seed: int = SEED + 10000) -> list:
    """Generate held-out test maps with V* pre-computed."""
    rng = np.random.RandomState(seed)
    test_data = []
    for _ in range(n_maps):
        s = int(rng.randint(0, 2**28))
        obstacles, goal = make_random_map(grid_size=grid_size, seed=s)
        grid = (obstacles, goal)
        V_star = compute_V_star(grid, gamma=GAMMA)
        map_enc = encode_map(obstacles, goal, grid_size=grid_size)
        test_data.append({
            "obstacles": obstacles,
            "goal": goal,
            "V_star": V_star,
            "map_enc": map_enc,
            "grid_size": grid_size,
        })
    return test_data


def random_V_init(obstacles: np.ndarray, goal: tuple, V_star: np.ndarray,
                  rng: np.random.RandomState) -> np.ndarray:
    """Generate random initial value estimate in [V_min, 0]."""
    grid_size = obstacles.shape[0]
    N = grid_size * grid_size
    v_min = float(V_star.min())
    V = rng.uniform(v_min, 0.0, size=(N,)).astype(np.float32)
    _, is_goal, is_obstacle = _build_transition_table(obstacles, goal, grid_size)
    V[is_obstacle] = 0.0
    V[is_goal] = 0.0
    return V


# ---------------------------------------------------------------------------
# Exp A: Convergence curves
# ---------------------------------------------------------------------------

def exp_A_convergence(models_dict: dict, n_maps: int = 200, K: int = 50,
                      seed: int = SEED) -> dict:
    """
    Track mean RMSE vs V* for K=0..50 from random V0.

    Critical: ContractiveLinearizer should converge monotonically.
    UnconstrainedLinearizer may oscillate. IterativeMLP may diverge.
    """
    print("\n=== Exp A: Convergence Curves ===")
    rng = np.random.RandomState(seed)
    test_data = generate_test_maps(n_maps, grid_size=TEST_GRID_SIZE_SMALL,
                                   seed=seed + 99999)

    errors = {name: np.zeros((K + 1,)) for name in models_dict}

    for td in test_data:
        obstacles = td["obstacles"]
        goal = td["goal"]
        V_star = td["V_star"]
        map_enc = td["map_enc"]

        V0 = random_V_init(obstacles, goal, V_star, rng)
        V_star_flat = V_star.flatten()

        map_tensor = torch.tensor(map_enc[None], dtype=torch.float32, device=DEVICE)
        V_star_t = torch.tensor(V_star_flat[None], dtype=torch.float32, device=DEVICE)
        V0_t = torch.tensor(V0[None], dtype=torch.float32, device=DEVICE)

        for name, model in models_dict.items():
            with torch.no_grad():
                x = V0_t.clone()
                e0 = ((x - V_star_t) ** 2).mean().sqrt().item()
                errors[name][0] += e0

                for k in range(1, K + 1):
                    x = model(x, map_tensor)
                    # Clamp to prevent inf from divergent models
                    x = x.clamp(-1e4, 1e4)
                    e = ((x - V_star_t) ** 2).mean().sqrt().item()
                    errors[name][k] += e

    # Average over maps
    for name in errors:
        errors[name] /= n_maps

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = {
        "contractive": "blue",
        "unconstrained": "red",
        "iterative_mlp": "green",
        "vin": "orange",
    }
    for name, err in errors.items():
        color = color_map.get(name, "gray")
        ax.plot(range(K + 1), err, label=name, color=color, linewidth=2)

    ax.set_xlabel("Iteration k", fontsize=13)
    ax.set_ylabel("Mean RMSE(V_k, V*)", fontsize=13)
    ax.set_title("Convergence Curves: Error vs Iterations", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "convergence_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    # Check monotonicity for contractive model
    contractive_err = errors.get("contractive", None)
    if contractive_err is not None:
        diffs = np.diff(contractive_err)
        increases = (diffs > 1e-8).sum()
        print(f"  [ContractiveLinearizer] initial={contractive_err[0]:.4f}, "
              f"final={contractive_err[K]:.4f}, "
              f"non-monotone steps={increases} (should be 0 by theory)")

    for name, err in errors.items():
        imp = 100.0 * (err[0] - err[K]) / (err[0] + 1e-9)
        print(f"  {name}: init={err[0]:.4f} -> k={K}: {err[K]:.4f} ({imp:+.1f}%)")

    # Save JSON
    results = {name: err.tolist() for name, err in errors.items()}
    with open(os.path.join(RESULTS_DIR, "convergence_curves.json"), "w") as f:
        json.dump(results, f, indent=2)

    return errors


# ---------------------------------------------------------------------------
# Exp B: Fast Iteration vs Loop
# ---------------------------------------------------------------------------

def exp_B_fast_iteration(contractive_model: torch.nn.Module, n_maps: int = 50,
                         K_values: list = None, seed: int = SEED + 1) -> dict:
    """
    Compare loop-K vs A^K*g(x0) for ContractiveLinearizer.

    Since g is NOW truly invertible (AffineCouplingNet), MSE should be ~0
    (machine precision). This is the key result that was broken before.
    """
    if K_values is None:
        K_values = [1, 5, 10, 20, 50]

    print("\n=== Exp B: Fast Iteration vs Loop ===")
    rng = np.random.RandomState(seed)
    test_data = generate_test_maps(n_maps, grid_size=TEST_GRID_SIZE_SMALL,
                                   seed=seed + 66666)

    results = {}

    for K in K_values:
        mse_list = []
        max_err_list = []
        loop_times = []
        fast_times = []

        for td in test_data:
            obstacles = td["obstacles"]
            goal = td["goal"]
            V_star = td["V_star"]
            map_enc = td["map_enc"]

            V0 = random_V_init(obstacles, goal, V_star, rng)
            map_tensor = torch.tensor(map_enc[None], dtype=torch.float32, device=DEVICE)
            x0 = torch.tensor(V0[None], dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                t0 = time.perf_counter()
                x_loop = contractive_model.iterate_loop(x0, map_tensor, K)
                loop_times.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                x_fast = contractive_model.iterate_fast(x0, map_tensor, K)
                fast_times.append(time.perf_counter() - t0)

                mse = ((x_loop - x_fast) ** 2).mean().item()
                max_err = (x_loop - x_fast).abs().max().item()
                mse_list.append(mse)
                max_err_list.append(max_err)

        results[str(K)] = {
            "K": K,
            "mean_mse_loop_vs_fast": float(np.mean(mse_list)),
            "max_abs_error": float(np.max(max_err_list)),
            "mean_loop_time_ms": float(np.mean(loop_times) * 1000),
            "mean_fast_time_ms": float(np.mean(fast_times) * 1000),
            "speedup": float(np.mean(loop_times) / (np.mean(fast_times) + 1e-12)),
        }
        print(
            f"  K={K:2d}: MSE={np.mean(mse_list):.2e} "
            f"MaxErr={np.max(max_err_list):.2e} "
            f"loop={np.mean(loop_times)*1000:.2f}ms "
            f"fast={np.mean(fast_times)*1000:.2f}ms "
            f"speedup={results[str(K)]['speedup']:.1f}x"
        )

    path = os.path.join(RESULTS_DIR, "fast_iteration_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")

    return results


# ---------------------------------------------------------------------------
# Exp C: Planning Quality
# ---------------------------------------------------------------------------

def exp_C_planning_quality(models_dict: dict, n_maps: int = 200, K_iter: int = 30,
                            seed: int = SEED + 2) -> dict:
    """
    Derive greedy policy from V after K_iter iterations, simulate episodes.
    Report success rate for each model.
    """
    print("\n=== Exp C: Planning Quality ===")
    rng = np.random.RandomState(seed)
    test_data = generate_test_maps(n_maps, grid_size=TEST_GRID_SIZE_SMALL,
                                   seed=seed + 88888)

    results = {}

    for name, model in models_dict.items():
        successes = 0
        total_steps = 0

        for td in test_data:
            obstacles = td["obstacles"]
            goal = td["goal"]
            V_star = td["V_star"]
            map_enc = td["map_enc"]
            grid_size = td["grid_size"]

            V0 = random_V_init(obstacles, goal, V_star, rng)
            map_tensor = torch.tensor(map_enc[None], dtype=torch.float32, device=DEVICE)
            x = torch.tensor(V0[None], dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                if hasattr(model, 'iterate_fast'):
                    x = model.iterate_fast(x, map_tensor, 1000)
                else:
                    for _ in range(K_iter):
                        x = model(x, map_tensor)
                        x = x.clamp(-1e4, 1e4)

            V_final = x.squeeze(0).cpu().numpy().flatten()
            policy = get_greedy_policy(V_final, obstacles, goal, gamma=GAMMA)
            success, steps = simulate_episode(
                policy, obstacles, goal,
                max_steps=4 * grid_size,
                seed=int(rng.randint(0, 2**28)),
            )
            if success:
                successes += 1
                total_steps += steps

        success_rate = successes / n_maps
        avg_steps = total_steps / max(successes, 1)
        results[name] = {
            "success_rate": success_rate,
            "avg_steps_on_success": avg_steps,
            "n_maps": n_maps,
            "K_iter": K_iter,
        }
        print(f"  {name}: success_rate={success_rate:.3f}, "
              f"avg_steps={avg_steps:.1f} (n={n_maps})")

    # Oracle baseline using V*
    oracle_successes = 0
    oracle_steps = 0
    for td in test_data:
        obstacles = td["obstacles"]
        goal = td["goal"]
        V_star = td["V_star"]
        grid_size = td["grid_size"]
        policy = get_greedy_policy(V_star, obstacles, goal, gamma=GAMMA)
        success, steps = simulate_episode(
            policy, obstacles, goal,
            max_steps=4 * grid_size,
            seed=int(rng.randint(0, 2**28)),
        )
        if success:
            oracle_successes += 1
            oracle_steps += steps

    results["oracle_V*"] = {
        "success_rate": oracle_successes / n_maps,
        "avg_steps_on_success": oracle_steps / max(oracle_successes, 1),
        "n_maps": n_maps,
        "K_iter": None,
    }
    print(f"  oracle_V*: success_rate={oracle_successes/n_maps:.3f}, "
          f"avg_steps={oracle_steps/max(oracle_successes,1):.1f}")

    path = os.path.join(RESULTS_DIR, "planning_quality.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")

    return results


# ---------------------------------------------------------------------------
# Exp D: Generalization (10x10 -> 20x20)
# ---------------------------------------------------------------------------

def exp_D_generalization(n_maps: int = 100, K_iter: int = 30,
                          seed: int = SEED + 3) -> dict:
    """
    Evaluate native 20x20-trained models on 20x20 maps.

    Loads contractive_20x20.pt and vin_20x20.pt from the checkpoint dir.
    Run train.py --grid_size 20 --models contractive vin first to produce them.
    """
    print("\n=== Exp D: Generalization (native 20x20 models) ===")
    rng = np.random.RandomState(seed)
    test_data_20 = generate_test_maps(n_maps, grid_size=TEST_GRID_SIZE_LARGE,
                                      seed=seed + 55555)

    model_configs_20 = [
        ("contractive_20x20", ContractiveLinearizer(state_dim=400, spectral_bound=0.9)),
        ("vin_20x20",         VINBaseline(grid_size=20)),
    ]

    results = {}
    loaded_any = False

    for ckpt_name, model_inst in model_configs_20:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {ckpt_name}: checkpoint not found ({ckpt_path})")
            results[ckpt_name] = {
                "mean_mse_20x20": float("nan"),
                "std_mse_20x20": float("nan"),
                "n_valid": 0,
                "note": "Checkpoint missing — run train.py --grid_size 20 first.",
            }
            continue

        model = load_model(model_inst, ckpt_name)
        loaded_any = True
        mse_list = []

        for td in test_data_20:
            obstacles = td["obstacles"]
            goal = td["goal"]
            V_star_20 = td["V_star"]
            map_enc_20 = td["map_enc"]

            V0_20 = random_V_init(obstacles, goal, V_star_20, rng)
            map_tensor = torch.tensor(map_enc_20[None], dtype=torch.float32, device=DEVICE)
            x = torch.tensor(V0_20[None], dtype=torch.float32, device=DEVICE)
            V_star_t = torch.tensor(V_star_20[None], dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                if hasattr(model, 'iterate_fast'):
                    x = model.iterate_fast(x, map_tensor, 1000)
                else:
                    for _ in range(K_iter):
                        x = model(x, map_tensor)
                        x = x.clamp(-1e4, 1e4)
                mse = ((x - V_star_t) ** 2).mean().item()
            mse_list.append(mse)

        avg_mse = float(np.mean(mse_list))
        std_mse = float(np.std(mse_list))
        results[ckpt_name] = {
            "mean_mse_20x20": avg_mse,
            "std_mse_20x20": std_mse,
            "n_valid": len(mse_list),
            "note": "Native 20x20 model trained and evaluated on 20x20 maps.",
        }
        print(f"  {ckpt_name}: mean_MSE={avg_mse:.4f} ± {std_mse:.4f} (n={n_maps})")

    if not loaded_any:
        print("  No 20x20 checkpoints found. Run: python train.py --grid_size 20 "
              "--models contractive vin")

    path = os.path.join(RESULTS_DIR, "generalization_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")

    return results


# ---------------------------------------------------------------------------
# Exp E: Ablation — Spectral bound
# ---------------------------------------------------------------------------

def exp_E_ablation_spectral(ablation_models: dict, n_maps: int = 200, K: int = 50,
                             seed: int = SEED + 4) -> dict:
    """
    ContractiveLinearizer with spectral_bound in {0.5, 0.9, 0.99}.
    Plot convergence curves (same format as Exp A).
    """
    print("\n=== Exp E: Ablation — Spectral Bound ===")
    rng = np.random.RandomState(seed)
    test_data = generate_test_maps(n_maps, grid_size=TEST_GRID_SIZE_SMALL,
                                   seed=seed + 55555)

    errors = {name: np.zeros((K + 1,)) for name in ablation_models}

    for td in test_data:
        obstacles = td["obstacles"]
        goal = td["goal"]
        V_star = td["V_star"]
        map_enc = td["map_enc"]

        V0 = random_V_init(obstacles, goal, V_star, rng)
        map_tensor = torch.tensor(map_enc[None], dtype=torch.float32, device=DEVICE)
        V_star_t = torch.tensor(V_star[None], dtype=torch.float32, device=DEVICE)

        for name, model in ablation_models.items():
            with torch.no_grad():
                x = torch.tensor(V0[None], dtype=torch.float32, device=DEVICE)
                errors[name][0] += ((x - V_star_t) ** 2).mean().sqrt().item()
                for k in range(1, K + 1):
                    x = model(x, map_tensor)
                    x = x.clamp(-1e4, 1e4)
                    errors[name][k] += ((x - V_star_t) ** 2).mean().sqrt().item()

    for name in errors:
        errors[name] /= n_maps

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = {
        "contractive_05": "green",
        "contractive_09": "blue",
        "contractive_099": "red",
    }
    label_map = {
        "contractive_05": "alpha=0.5",
        "contractive_09": "alpha=0.9",
        "contractive_099": "alpha=0.99",
    }
    for name, err in errors.items():
        ax.plot(range(K + 1), err,
                label=label_map.get(name, name),
                color=color_map.get(name, "gray"),
                linewidth=2)

    ax.set_xlabel("Iteration k", fontsize=13)
    ax.set_ylabel("Mean RMSE(V_k, V*)", fontsize=13)
    ax.set_title("Ablation: Effect of Spectral Bound alpha", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "ablation_spectral_bound.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    for name, err in errors.items():
        print(f"  {name}: init={err[0]:.4f} k=10={err[10]:.4f} final={err[K]:.4f}")

    results = {name: err.tolist() for name, err in errors.items()}
    with open(os.path.join(RESULTS_DIR, "ablation_spectral_bound.json"), "w") as f:
        json.dump(results, f, indent=2)

    return errors


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Exp F: Generalization to 50x50 (native models)
# ---------------------------------------------------------------------------

def exp_F_generalization_50x50(n_maps: int = 50, K_iter: int = 30,
                                seed: int = SEED + 5) -> dict:
    """
    Evaluate native 50x50-trained models on 50x50 maps.

    Loads contractive_50x50.pt and vin_50x50.pt from the checkpoint dir.
    Run train.py --grid_size 50 --models contractive vin first to produce them.
    """
    print("\n=== Exp F: Generalization (native 50x50 models) ===")
    rng = np.random.RandomState(seed)
    test_data_50 = generate_test_maps(n_maps, grid_size=TEST_GRID_SIZE_XL,
                                      seed=seed + 99999)

    model_configs_50 = [
        ("contractive_50x50", ContractiveLinearizer(state_dim=2500, spectral_bound=0.9)),
        ("vin_50x50",         VINBaseline(grid_size=50)),
    ]

    results = {}
    loaded_any = False

    for ckpt_name, model_inst in model_configs_50:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {ckpt_name}: checkpoint not found")
            results[ckpt_name] = {
                "mean_mse_50x50": float("nan"),
                "std_mse_50x50": float("nan"),
                "n_valid": 0,
                "note": "Checkpoint missing — run train.py --grid_size 50 first.",
            }
            continue

        model = load_model(model_inst, ckpt_name)
        loaded_any = True
        mse_list = []

        for td in test_data_50:
            obstacles = td["obstacles"]
            goal = td["goal"]
            V_star_50 = td["V_star"]
            map_enc_50 = td["map_enc"]

            V0_50 = random_V_init(obstacles, goal, V_star_50, rng)
            map_tensor = torch.tensor(map_enc_50[None], dtype=torch.float32, device=DEVICE)
            x = torch.tensor(V0_50[None], dtype=torch.float32, device=DEVICE)
            V_star_t = torch.tensor(V_star_50[None], dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                if hasattr(model, 'iterate_fast'):
                    x = model.iterate_fast(x, map_tensor, 1000)
                else:
                    for _ in range(K_iter):
                        x = model(x, map_tensor)
                        x = x.clamp(-1e4, 1e4)
                mse = ((x - V_star_t) ** 2).mean().item()
            mse_list.append(mse)

        avg_mse = float(np.mean(mse_list))
        std_mse = float(np.std(mse_list))
        results[ckpt_name] = {
            "mean_mse_50x50": avg_mse,
            "std_mse_50x50": std_mse,
            "n_valid": len(mse_list),
            "note": "Native 50x50 model trained and evaluated on 50x50 maps.",
        }
        print(f"  {ckpt_name}: mean_MSE={avg_mse:.4f} ± {std_mse:.4f} (n={n_maps})")

    if not loaded_any:
        print("  No 50x50 checkpoints found. Run: python train.py --grid_size 50 "
              "--models contractive vin")

    path = os.path.join(RESULTS_DIR, "generalization_50x50_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")

    return results


def write_summary_table(conv_errors: dict, planning_results: dict,
                        fast_iter_results: dict, gen_results: dict,
                        gen_50_results: dict = None, K: int = 50) -> str:
    lines = []
    lines.append("=" * 75)
    lines.append("CONTRACTIVE LINEARIZER — PLANNING EXPERIMENT SUMMARY")
    lines.append("=" * 75)

    lines.append("\n--- Exp A: Convergence (RMSE at K=0 and K=50) ---")
    lines.append(f"{'Model':<25} {'Init RMSE':>12} {'K=50 RMSE':>12} {'Reduction%':>12}")
    lines.append("-" * 65)
    for name, err in conv_errors.items():
        red = 100.0 * (err[0] - err[K]) / (err[0] + 1e-9)
        lines.append(f"{name:<25} {err[0]:>12.4f} {err[K]:>12.4f} {red:>11.1f}%")

    lines.append("\n--- Exp B: Fast Iteration vs Loop ---")
    lines.append(f"{'K':>5} {'MSE(loop vs fast)':>20} {'MaxAbsErr':>12} "
                 f"{'Loop ms':>10} {'Fast ms':>10} {'Speedup':>9}")
    lines.append("-" * 70)
    for K_str, r in fast_iter_results.items():
        lines.append(
            f"{r['K']:>5} {r['mean_mse_loop_vs_fast']:>20.2e} "
            f"{r['max_abs_error']:>12.2e} "
            f"{r['mean_loop_time_ms']:>10.3f} {r['mean_fast_time_ms']:>10.3f} "
            f"{r['speedup']:>9.2f}x"
        )

    lines.append("\n--- Exp C: Planning Quality (success rate at K=30) ---")
    lines.append(f"{'Model':<25} {'Success Rate':>14} {'Avg Steps':>12}")
    lines.append("-" * 55)
    for name, r in planning_results.items():
        lines.append(
            f"{name:<25} {r['success_rate']:>14.3f} "
            f"{r['avg_steps_on_success']:>12.1f}"
        )

    lines.append("\n--- Exp D: Generalization (native 20x20 models) ---")
    lines.append(f"{'Model':<25} {'Mean MSE':>12} {'Std MSE':>12}")
    lines.append("-" * 51)
    for name, r in gen_results.items():
        mse_val = r.get('mean_mse_20x20', float('nan'))
        std_val = r.get('std_mse_20x20', float('nan'))
        lines.append(f"{name:<25} {mse_val:>12.4f} {std_val:>12.4f}")

    if gen_50_results:
        lines.append("\n--- Exp F: Generalization (native 50x50 models) ---")
        lines.append(f"{'Model':<25} {'Mean MSE':>12} {'Std MSE':>12}")
        lines.append("-" * 51)
        for name, r in gen_50_results.items():
            mse_val = r.get('mean_mse_50x50', float('nan'))
            std_val = r.get('std_mse_50x50', float('nan'))
            lines.append(f"{name:<25} {mse_val:>12.4f} {std_val:>12.4f}")

    lines.append("\n" + "=" * 75)
    text = "\n".join(lines)

    path = os.path.join(RESULTS_DIR, "summary_table.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"\nSaved summary: {path}")
    print(text)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Loading models from: {CHECKPOINT_DIR}")

    # Load main models
    model_configs = [
        ("contractive",    ContractiveLinearizer(spectral_bound=0.9)),
        ("unconstrained",  UnconstrainedLinearizer()),
        ("iterative_mlp",  IterativeMLPBaseline()),
        ("vin",            VINBaseline()),
    ]

    models_dict = {}
    for name, inst in model_configs:
        try:
            m = load_model(inst, name)
            models_dict[name] = m
            print(f"  Loaded: {name}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  Warning: skipping {name}: {e}")

    if not models_dict:
        print("No models loaded! Run train.py first.")
        return

    # Load ablation models
    ablation_configs = [
        ("contractive_05",  make_contractive(spectral_bound=0.5)),
        ("contractive_09",  make_contractive(spectral_bound=0.9)),
        ("contractive_099", make_contractive(spectral_bound=0.99)),
    ]
    ablation_models = {}
    for name, inst in ablation_configs:
        try:
            m = load_model(inst, name)
            ablation_models[name] = m
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  Warning (ablation): skipping {name}: {e}")

    # Exp A: Convergence curves
    conv_errors = exp_A_convergence(models_dict, n_maps=200, K=50)

    # Exp B: Fast iteration
    fast_iter_results = {}
    if "contractive" in models_dict:
        fast_iter_results = exp_B_fast_iteration(
            models_dict["contractive"], n_maps=50, K_values=[1, 5, 10, 20, 50]
        )

    # Exp C: Planning quality
    planning_results = exp_C_planning_quality(models_dict, n_maps=200, K_iter=30)

    # Exp D: Generalization (20x20)
    gen_results = exp_D_generalization(n_maps=100, K_iter=30)

    # Exp E: Ablation spectral bound
    if ablation_models:
        exp_E_ablation_spectral(ablation_models, n_maps=200, K=50)

    # Exp F: Generalization (50x50) — only if checkpoints exist
    gen_50_results = exp_F_generalization_50x50(n_maps=50, K_iter=30)

    # Summary table
    if fast_iter_results and gen_results:
        write_summary_table(conv_errors, planning_results, fast_iter_results,
                            gen_results, gen_50_results, K=50)

    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    main()
