"""
End-to-end runner for the Contractive Linearizer planning experiment.

Steps:
  1. Verify AffineCouplingNet is truly invertible (unit test)
  2. Train all models (ContractiveLinearizer, VINBaseline,
                       UnconstrainedLinearizer, IterativeMLPBaseline,
                       ablation variants)
  3. Run all evaluation experiments (A-E)
  4. Print summary table

Usage:
    cd /home/nvidia/Linearizer
    python contractive_rl/planning/run_planning.py
"""

import os
import sys
import time

# Ensure correct import paths
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
_root_dir = os.path.dirname(_parent_dir)

for p in [_this_dir, _parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Step 0: Sanity check — AffineCouplingNet must be truly invertible
# ---------------------------------------------------------------------------

def verify_invertibility():
    """Unit test: decode(encode(x)) == x to machine precision."""
    print("=" * 60)
    print("STEP 0: Verifying AffineCouplingNet invertibility")
    print("=" * 60)

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "invertible_net",
        os.path.join(_parent_dir, "shared", "invertible_net.py")
    )
    _inv_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_inv_mod)
    AffineCouplingNet = _inv_mod.AffineCouplingNet

    torch.manual_seed(42)
    for dim in [100, 64, 128]:
        g = AffineCouplingNet(dim=dim, n_coupling_layers=6, hidden_dim=128)
        g.eval()

        x = torch.randn(8, dim)
        with torch.no_grad():
            z = g.encode(x)
            x_rec = g.decode(z)

        max_err = (x_rec - x).abs().max().item()
        mean_err = (x_rec - x).abs().mean().item()
        status = "PASS" if max_err < 1e-4 else "FAIL"
        print(f"  dim={dim:3d}: max_err={max_err:.2e}, mean_err={mean_err:.2e}  [{status}]")
        if max_err >= 1e-4:
            print(f"  ERROR: Invertibility check failed for dim={dim}!")
            sys.exit(1)

    print("  All invertibility checks passed.\n")


# ---------------------------------------------------------------------------
# Step 1: Training
# ---------------------------------------------------------------------------

def run_training():
    print("=" * 60)
    print("STEP 1: Training all models")
    print("=" * 60)

    # Use importlib to avoid shadowing by contractive_rl/models.py
    import importlib.util

    def _load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    _train_mod = _load_module("planning_train", os.path.join(_this_dir, "train.py"))
    _models_mod = _load_module("planning_models", os.path.join(_this_dir, "models.py"))

    generate_dataset = _train_mod.generate_dataset
    train_model = _train_mod.train_model
    DEVICE = _train_mod.DEVICE
    SEED = _train_mod.SEED

    ContractiveLinearizer = _models_mod.ContractiveLinearizer
    VINBaseline = _models_mod.VINBaseline
    UnconstrainedLinearizer = _models_mod.UnconstrainedLinearizer
    IterativeMLPBaseline = _models_mod.IterativeMLPBaseline
    make_contractive = _models_mod.make_contractive

    # Pre-generate dataset once (shared across models)
    print("\nGenerating training dataset (5000 maps, 10x10) ...")
    dataset = generate_dataset(pool_size=5000, grid_size=10, seed=SEED)

    # All models to train
    models_to_train = [
        ("contractive",    ContractiveLinearizer(spectral_bound=0.99)),
        ("unconstrained",  UnconstrainedLinearizer()),
        ("iterative_mlp",  IterativeMLPBaseline()),
        ("vin",            VINBaseline()),
        ("contractive_05",  make_contractive(spectral_bound=0.5)),
        ("contractive_09",  make_contractive(spectral_bound=0.9)),
        ("contractive_099", make_contractive(spectral_bound=0.99)),
    ]

    t_start = time.time()
    for name, model in models_to_train:
        print(f"\n--- Training: {name} ---")
        train_model(
            model=model,
            model_name=name,
            dataset=dataset,
            num_steps=8000,
            batch_size=32,
            lr=3e-4,
            seed=SEED,
            device=DEVICE,
            verbose=True,
        )

    elapsed = time.time() - t_start
    print(f"\nAll training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")


# ---------------------------------------------------------------------------
# Step 2: Evaluation
# ---------------------------------------------------------------------------

def run_evaluation():
    print("\n" + "=" * 60)
    print("STEP 2: Running evaluation experiments")
    print("=" * 60)

    # Import evaluate module explicitly to avoid shadowing
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "planning_evaluate", os.path.join(_this_dir, "evaluate.py")
    )
    evaluate_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate_mod)
    evaluate_mod.main()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.time()

    print("=" * 60)
    print("CONTRACTIVE LINEARIZER — PLANNING EXPERIMENT")
    print("=" * 60)
    print(f"Working dir: {_this_dir}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print()

    verify_invertibility()
    run_training()
    run_evaluation()

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"COMPLETE — Total wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    results_dir = os.path.join(_this_dir, "results")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
