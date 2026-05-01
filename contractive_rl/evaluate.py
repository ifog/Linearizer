"""
Evaluation script for Contractive Linearizer RL experiments.

Runs 5 experiments:
  5.1 — Convergence Test
  5.2 — Planning Quality
  5.3 — Generalization (held-out maps)
  5.4 — Fast Iteration Trick
  5.5 — Ablation (different spectral bounds)

Saves results to contractive_rl/results/
"""

import os
import json
import time
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gridworld import (
    make_random_map,
    value_iteration,
    encode_map,
    generate_batch,
    get_greedy_policy,
    simulate_episode,
    GRID_SIZE,
    GAMMA,
)
from models import (
    ContractiveLinearizer,
    UnconstrainedLinearizer,
    MLPBaseline,
    VINBaseline,
    STATE_DIM,
    make_contractive_with_scale,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_model(model_class_or_instance, checkpoint_name, device=DEVICE):
    """Load a model from checkpoint."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(model_class_or_instance, torch.nn.Module):
        model = model_class_or_instance
    else:
        model = model_class_or_instance()

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def generate_test_maps(n_maps, seed=SEED + 10000):
    """Generate n_maps held-out test maps."""
    rng = np.random.RandomState(seed)
    test_data = []
    for i in range(n_maps):
        s = rng.randint(0, 2**31 - 1)
        obstacles, goal = make_random_map(grid_size=GRID_SIZE, seed=s)
        V_star = value_iteration(obstacles, goal, gamma=GAMMA)
        map_enc = encode_map(obstacles, goal, grid_size=GRID_SIZE)
        test_data.append({
            "obstacles": obstacles,
            "goal": goal,
            "V_star": V_star,
            "map_enc": map_enc,
        })
    return test_data


def random_V_init(obstacles, goal, V_star, rng):
    """Generate a random initial value estimate."""
    V = rng.uniform(V_star.min(), 0.0, size=(GRID_SIZE, GRID_SIZE)).astype(np.float32)
    V[obstacles] = 0.0
    V[goal[0], goal[1]] = 0.0
    return V


# ---------------------------------------------------------------------------
# Exp 5.1 — Convergence Test
# ---------------------------------------------------------------------------

def exp_convergence(models_dict, n_maps=20, K=50, seed=SEED):
    """
    For each model, iterate K steps from random V0 and track ||V_k - V*||.

    models_dict: dict of {name: model}
    Returns: dict of {name: errors_array shape (K+1,)}
    """
    print("\n=== Exp 5.1: Convergence Test ===")
    rng = np.random.RandomState(seed)
    test_data = generate_test_maps(n_maps, seed=seed + 99999)

    errors = {name: np.zeros((K + 1,)) for name in models_dict}
    counts = np.zeros((K + 1,))

    for td in test_data:
        obstacles = td["obstacles"]
        goal = td["goal"]
        V_star = td["V_star"]
        map_enc = td["map_enc"]

        # Random V0
        V0 = random_V_init(obstacles, goal, V_star, rng)
        V_star_flat = V_star.flatten()
        V0_flat = V0.flatten()

        map_tensor = torch.tensor(map_enc[None], dtype=torch.float32, device=DEVICE)
        V_star_tensor = torch.tensor(V_star_flat[None], dtype=torch.float32, device=DEVICE)
        V0_tensor = torch.tensor(V0_flat[None], dtype=torch.float32, device=DEVICE)

        for name, model in models_dict.items():
            with torch.no_grad():
                x = V0_tensor.clone()
                e0 = ((x - V_star_tensor) ** 2).mean().sqrt().item()
                errors[name][0] += e0

                for k in range(1, K + 1):
                    x = model(x, map_tensor)
                    e = ((x - V_star_tensor) ** 2).mean().sqrt().item()
                    errors[name][k] += e

        counts += 1.0

    # Average over maps
    for name in models_dict:
        errors[name] /= n_maps

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["blue", "red", "green", "orange"]
    for (name, err), color in zip(errors.items(), colors):
        ax.plot(range(K + 1), err, label=name, color=color, linewidth=2)

    ax.set_xlabel("Iteration k")
    ax.set_ylabel("RMSE(V_k, V*)")
    ax.set_title("Convergence: ||V_k - V*|| over iterations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plot_path = os.path.join(RESULTS_DIR, "convergence_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    for name, err in errors.items():
        print(f"  {name}: initial={err[0]:.4f}, final={err[K]:.4f}, "
              f"improvement={100*(err[0]-err[K])/(err[0]+1e-9):.1f}%")

    return errors


# ---------------------------------------------------------------------------
# Exp 5.2 — Planning Quality
# ---------------------------------------------------------------------------

def exp_planning_quality(models_dict, n_maps=100, K_iter=30, seed=SEED + 1):
    """
    Derive greedy policy from V after K iterations and measure success rate.
    """
    print("\n=== Exp 5.2: Planning Quality ===")
    rng = np.random.RandomState(seed)
    test_data = generate_test_maps(n_maps, seed=seed + 88888)

    results = {}

    for name, model in models_dict.items():
        successes = 0
        total_steps = 0

        for td in test_data:
            obstacles = td["obstacles"]
            goal = td["goal"]
            V_star = td["V_star"]
            map_enc = td["map_enc"]

            V0 = random_V_init(obstacles, goal, V_star, rng)
            V0_flat = V0.flatten()

            map_tensor = torch.tensor(map_enc[None], dtype=torch.float32, device=DEVICE)
            x = torch.tensor(V0_flat[None], dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                for _ in range(K_iter):
                    x = model(x, map_tensor)

            V_final_flat = x.squeeze(0).cpu().numpy().flatten()
            policy = get_greedy_policy(V_final_flat, obstacles, goal, gamma=GAMMA)
            success, steps = simulate_episode(
                policy, obstacles, goal, max_steps=200, seed=rng.randint(0, 2**31 - 1)
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
        }
        print(f"  {name}: success_rate={success_rate:.3f}, avg_steps={avg_steps:.1f}")

    # Also compute oracle (V* directly)
    oracle_successes = 0
    oracle_steps = 0
    for td in test_data:
        obstacles = td["obstacles"]
        goal = td["goal"]
        V_star = td["V_star"]
        policy = get_greedy_policy(V_star, obstacles, goal, gamma=GAMMA)
        success, steps = simulate_episode(
            policy, obstacles, goal, max_steps=200, seed=rng.randint(0, 2**31 - 1)
        )
        if success:
            oracle_successes += 1
            oracle_steps += steps

    results["oracle_V*"] = {
        "success_rate": oracle_successes / n_maps,
        "avg_steps_on_success": oracle_steps / max(oracle_successes, 1),
        "n_maps": n_maps,
    }
    print(f"  oracle_V*: success_rate={oracle_successes/n_maps:.3f}, "
          f"avg_steps={oracle_steps/max(oracle_successes,1):.1f}")

    path = os.path.join(RESULTS_DIR, "planning_quality.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")

    return results


# ---------------------------------------------------------------------------
# Exp 5.3 — Generalization (held-out maps)
# ---------------------------------------------------------------------------

def exp_generalization(models_dict, n_maps=100, K_iter=30, seed=SEED + 2):
    """
    Test on held-out maps (different from training distribution — different seeds).
    """
    print("\n=== Exp 5.3: Generalization (held-out maps) ===")
    rng = np.random.RandomState(seed)
    test_data = generate_test_maps(n_maps, seed=seed + 77777)

    results = {}
    for name, model in models_dict.items():
        mse_list = []
        for td in test_data:
            obstacles = td["obstacles"]
            goal = td["goal"]
            V_star = td["V_star"]
            map_enc = td["map_enc"]

            V0 = random_V_init(obstacles, goal, V_star, rng)
            map_tensor = torch.tensor(map_enc[None], dtype=torch.float32, device=DEVICE)
            x = torch.tensor(V0.flatten()[None], dtype=torch.float32, device=DEVICE)
            V_star_t = torch.tensor(V_star.flatten()[None], dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                for _ in range(K_iter):
                    x = model(x, map_tensor)

            mse = ((x - V_star_t) ** 2).mean().item()
            mse_list.append(mse)

        avg_mse = float(np.mean(mse_list))
        std_mse = float(np.std(mse_list))
        results[name] = {"mean_mse": avg_mse, "std_mse": std_mse}
        print(f"  {name}: mean_MSE={avg_mse:.4f} ± {std_mse:.4f}")

    path = os.path.join(RESULTS_DIR, "generalization.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")
    print("  Note: generalization tested on held-out 10x10 maps (same size, different seeds)")

    return results


# ---------------------------------------------------------------------------
# Exp 5.4 — Fast Iteration Trick
# ---------------------------------------------------------------------------

def exp_fast_iteration(contractive_model, n_maps=50, K_values=[1, 5, 10, 20, 50], seed=SEED + 3):
    """
    Compare loop-K vs A^K direct for ContractiveLinearizer.
    """
    print("\n=== Exp 5.4: Fast Iteration Trick ===")
    rng = np.random.RandomState(seed)
    test_data = generate_test_maps(n_maps, seed=seed + 66666)

    results = {}

    for K in K_values:
        mse_list = []
        loop_times = []
        fast_times = []

        for td in test_data:
            obstacles = td["obstacles"]
            goal = td["goal"]
            V_star = td["V_star"]
            map_enc = td["map_enc"]

            V0 = random_V_init(obstacles, goal, V_star, rng)
            map_tensor = torch.tensor(map_enc[None], dtype=torch.float32, device=DEVICE)
            x0 = torch.tensor(V0.flatten()[None], dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                # Loop method
                t0 = time.perf_counter()
                x_loop = contractive_model.iterate_loop(x0, map_tensor, K)
                t_loop = time.perf_counter() - t0

                # Fast method
                t0 = time.perf_counter()
                x_fast = contractive_model.iterate_fast(x0, map_tensor, K)
                t_fast = time.perf_counter() - t0

                mse = ((x_loop - x_fast) ** 2).mean().item()
                mse_list.append(mse)
                loop_times.append(t_loop)
                fast_times.append(t_fast)

        results[str(K)] = {
            "K": K,
            "mean_mse_loop_vs_fast": float(np.mean(mse_list)),
            "mean_loop_time_ms": float(np.mean(loop_times) * 1000),
            "mean_fast_time_ms": float(np.mean(fast_times) * 1000),
            "speedup": float(np.mean(loop_times) / (np.mean(fast_times) + 1e-9)),
        }
        print(f"  K={K}: MSE(loop vs fast)={np.mean(mse_list):.6f} | "
              f"loop={np.mean(loop_times)*1000:.2f}ms | "
              f"fast={np.mean(fast_times)*1000:.2f}ms | "
              f"speedup={results[str(K)]['speedup']:.2f}x")

    path = os.path.join(RESULTS_DIR, "fast_iteration.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")

    return results


# ---------------------------------------------------------------------------
# Exp 5.5 — Ablation: different spectral bounds
# ---------------------------------------------------------------------------

def exp_ablation(ablation_models, n_maps=20, K=50, seed=SEED + 4):
    """
    ContractiveLinearizer with spectral_scale in {0.5, 0.9, 0.99}.
    Track convergence error vs K.
    """
    print("\n=== Exp 5.5: Ablation — Spectral Bounds ===")
    rng = np.random.RandomState(seed)
    test_data = generate_test_maps(n_maps, seed=seed + 55555)

    errors = {name: np.zeros((K + 1,)) for name in ablation_models}

    for td in test_data:
        obstacles = td["obstacles"]
        goal = td["goal"]
        V_star = td["V_star"]
        map_enc = td["map_enc"]

        V0 = random_V_init(obstacles, goal, V_star, rng)
        map_tensor = torch.tensor(map_enc[None], dtype=torch.float32, device=DEVICE)
        V_star_tensor = torch.tensor(V_star.flatten()[None], dtype=torch.float32, device=DEVICE)

        for name, model in ablation_models.items():
            with torch.no_grad():
                x = torch.tensor(V0.flatten()[None], dtype=torch.float32, device=DEVICE)
                e0 = ((x - V_star_tensor) ** 2).mean().sqrt().item()
                errors[name][0] += e0

                for k in range(1, K + 1):
                    x = model(x, map_tensor)
                    e = ((x - V_star_tensor) ** 2).mean().sqrt().item()
                    errors[name][k] += e

    for name in ablation_models:
        errors[name] /= n_maps

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scale_labels = {
        "contractive_05": "scale=0.5",
        "contractive_09": "scale=0.9",
        "contractive_099": "scale=0.99",
    }
    colors = ["green", "blue", "red"]
    for (name, err), color in zip(errors.items(), colors):
        label = scale_labels.get(name, name)
        ax.plot(range(K + 1), err, label=label, color=color, linewidth=2)

    ax.set_xlabel("Iteration k")
    ax.set_ylabel("RMSE(V_k, V*)")
    ax.set_title("Ablation: Effect of Spectral Bound on Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plot_path = os.path.join(RESULTS_DIR, "ablation_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    for name, err in errors.items():
        print(f"  {name}: initial={err[0]:.4f}, k=10={err[10]:.4f}, "
              f"final={err[K]:.4f}")

    return errors


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def write_summary_table(conv_errors, planning_results, gen_results, fast_iter_results, K=50):
    lines = []
    lines.append("=" * 70)
    lines.append("CONTRACTIVE LINEARIZER RL — SUMMARY TABLE")
    lines.append("=" * 70)

    lines.append("\n--- Exp 5.1: Convergence (RMSE at K=50 iterations) ---")
    lines.append(f"{'Model':<25} {'Init RMSE':>12} {'Final RMSE':>12} {'Improvement%':>14}")
    lines.append("-" * 65)
    for name, err in conv_errors.items():
        imp = 100 * (err[0] - err[K]) / (err[0] + 1e-9)
        lines.append(f"{name:<25} {err[0]:>12.4f} {err[K]:>12.4f} {imp:>13.1f}%")

    lines.append("\n--- Exp 5.2: Planning Quality (K=30 iterations) ---")
    lines.append(f"{'Model':<25} {'Success Rate':>14} {'Avg Steps':>12}")
    lines.append("-" * 55)
    for name, r in planning_results.items():
        lines.append(f"{name:<25} {r['success_rate']:>14.3f} {r['avg_steps_on_success']:>12.1f}")

    lines.append("\n--- Exp 5.3: Generalization (held-out maps, MSE) ---")
    lines.append(f"{'Model':<25} {'Mean MSE':>12} {'Std MSE':>12}")
    lines.append("-" * 51)
    for name, r in gen_results.items():
        lines.append(f"{name:<25} {r['mean_mse']:>12.4f} {r['std_mse']:>12.4f}")

    lines.append("\n--- Exp 5.4: Fast Iteration Trick ---")
    lines.append(f"{'K':>5} {'MSE(loop vs fast)':>20} {'Loop ms':>10} {'Fast ms':>10} {'Speedup':>10}")
    lines.append("-" * 60)
    for K_str, r in fast_iter_results.items():
        lines.append(f"{r['K']:>5} {r['mean_mse_loop_vs_fast']:>20.8f} "
                     f"{r['mean_loop_time_ms']:>10.3f} {r['mean_fast_time_ms']:>10.3f} "
                     f"{r['speedup']:>10.2f}")

    lines.append("\n" + "=" * 70)

    text = "\n".join(lines)
    path = os.path.join(RESULTS_DIR, "summary_table.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"\nSaved summary: {path}")
    print(text)

    return text


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Loading models from: {CHECKPOINT_DIR}")

    # Load main models
    models_dict = {}
    model_configs = [
        ("contractive", ContractiveLinearizer(spectral_scale=0.99)),
        ("unconstrained", UnconstrainedLinearizer()),
        ("mlp", MLPBaseline()),
        ("vin", VINBaseline()),
    ]

    for name, model_inst in model_configs:
        try:
            model = load_model(model_inst, name)
            models_dict[name] = model
            print(f"  Loaded: {name}")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    if not models_dict:
        print("No models loaded! Run train.py first.")
        return

    # Load ablation models
    ablation_configs = [
        ("contractive_05", make_contractive_with_scale(0.5)),
        ("contractive_09", make_contractive_with_scale(0.9)),
        ("contractive_099", make_contractive_with_scale(0.99)),
    ]
    ablation_models = {}
    for name, model_inst in ablation_configs:
        try:
            model = load_model(model_inst, name)
            ablation_models[name] = model
        except FileNotFoundError as e:
            print(f"  Warning (ablation): {e}")

    # Run experiments
    # Exp 5.1
    conv_errors = exp_convergence(models_dict, n_maps=20, K=50)

    # Exp 5.2
    planning_results = exp_planning_quality(models_dict, n_maps=100, K_iter=30)

    # Exp 5.3
    gen_results = exp_generalization(models_dict, n_maps=100, K_iter=30)

    # Exp 5.4 — only for contractive
    fast_iter_results = {}
    if "contractive" in models_dict:
        fast_iter_results = exp_fast_iteration(
            models_dict["contractive"], n_maps=50, K_values=[1, 5, 10, 20, 50]
        )

    # Exp 5.5
    ablation_errors = {}
    if ablation_models:
        ablation_errors = exp_ablation(ablation_models, n_maps=20, K=50)

    # Summary
    write_summary_table(conv_errors, planning_results, gen_results, fast_iter_results, K=50)

    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    main()
