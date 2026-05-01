"""
End-to-end runner for the Contractive Linearizer value-based RL experiments.

Runs:
  1. Baird's counterexample (deadly triad demonstration)
  2. CartPole-v1 + Acrobot-v1 DQN benchmark (competitive performance)
  3. Deadly triad divergence test (no target net, large lr)
  4. Instability stress test (poisoned replay buffer)

Then prints a formatted summary table.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    t_start = time.time()
    print("\n" + "=" * 60)
    print("  CONTRACTIVE LINEARIZER — VALUE-BASED RL EXPERIMENTS")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 1. Baird's Counterexample
    # ------------------------------------------------------------------
    from contractive_rl.dqn.baird import run_baird
    t0 = time.time()
    baird_results = run_baird(out_dir=RESULTS_DIR)
    print(f"  [Baird done in {time.time()-t0:.1f}s]\n")

    # ------------------------------------------------------------------
    # 2. CartPole Benchmark
    # ------------------------------------------------------------------
    from contractive_rl.dqn.cartpole import run_cartpole
    t0 = time.time()
    cartpole_results = run_cartpole(out_dir=RESULTS_DIR)
    print(f"  [CartPole done in {time.time()-t0:.1f}s]\n")

    # ------------------------------------------------------------------
    # 3. Divergence test (deadly triad: no target net, large lr)
    # ------------------------------------------------------------------
    from contractive_rl.dqn.divergence_test import run_divergence_test
    t0 = time.time()
    div_results = run_divergence_test(out_dir=RESULTS_DIR)
    print(f"  [Divergence test done in {time.time()-t0:.1f}s]\n")

    # ------------------------------------------------------------------
    # 4. DQN benchmark (CartPole + Acrobot)
    # ------------------------------------------------------------------
    from contractive_rl.dqn.train_dqn import run_train_dqn
    t0 = time.time()
    dqn_results = run_train_dqn(out_dir=RESULTS_DIR)
    print(f"  [DQN benchmark done in {time.time()-t0:.1f}s]\n")

    # ------------------------------------------------------------------
    # 5. Stress Test (poisoned replay buffer)
    # ------------------------------------------------------------------
    from contractive_rl.dqn.stress_test import run_stress_test
    t0 = time.time()
    stress_results = run_stress_test(out_dir=RESULTS_DIR)
    print(f"  [Stress test done in {time.time()-t0:.1f}s]\n")

    # ------------------------------------------------------------------
    # Summary Table
    # ------------------------------------------------------------------
    b_lin = baird_results["linear"]
    b_mlp = baird_results["mlp"]
    b_con = baird_results["contractive"]

    cp_std = cartpole_results["standard"]
    cp_dbl = cartpole_results["double"]
    cp_con = cartpole_results["contractive"]

    st_std = stress_results["standard"]
    st_con = stress_results["contractive"]

    lin_status = "DIVERGED" if b_lin["diverged"] else "stable "
    mlp_status = "DIVERGED" if b_mlp["diverged"] else "stable "
    con_status = "STABLE  " if not b_con["diverged"] else "diverged"

    print("\n" + "=" * 60)
    print("  CONTRACTIVE LINEARIZER — VALUE-BASED RL RESULTS")
    print("=" * 60)
    print()
    print("Baird's Counterexample (divergence prevention):")
    print(f"  Standard TD:      {lin_status} (final ||Q|| = {b_lin['final_norm']:.2f})")
    print(f"  MLP-TD:           {mlp_status} (final ||Q|| = {b_mlp['final_norm']:.2f})")
    print(f"  Contractive:      {con_status} (final ||Q|| = {b_con['final_norm']:.6f})")
    print()
    print("Deadly Triad Divergence Test (CartPole, no target net, lr=1e-2):")
    for v, r in div_results.items():
        status = "DIVERGED" if r["diverged"] else "stable"
        print(f"  {v:<14s}: max|Q|={r['final_max_q_mean']:.3g}  → {status}")
    print()
    print("DQN Benchmark (CartPole-v1, 50k steps; Acrobot-v1, 100k steps — 3 seeds):")
    for env_id, summ in dqn_results.items():
        print(f"  {env_id}:")
        for v in ("standard", "double", "contractive"):
            r = summ.get(v, {})
            print(f"    {'Standard/Double/Contractive'[{'standard':0,'double':7,'contractive':14}[v]:{'standard':6,'double':13,'contractive':25}[v]]:<13s}: "
                  f"{r.get('final_mean',0):.1f} ± {r.get('final_std',0):.1f}")
    print()
    print("CartPole-v1 (cartpole.py, 50k steps, 5 seeds):")
    print(f"  StandardDQN:      mean={cp_std['final_mean']:.1f} ± {cp_std['final_std']:.1f}")
    print(f"  DoubleDQN:        mean={cp_dbl['final_mean']:.1f} ± {cp_dbl['final_std']:.1f}")
    print(f"  ContractiveDQN:   mean={cp_con['final_mean']:.1f} ± {cp_con['final_std']:.1f}")
    print()
    print("Stability Stress Test (20% poisoned buffer, 5 seeds):")
    print(f"  StandardDQN:      mean={st_std['mean_return']:.1f}±{st_std['std_return']:.1f}, "
          f"collapses={st_std['n_collapses']}/{len(st_std['final_returns_per_seed'])}, "
          f"variance={st_std['return_variance']:.1f}")
    print(f"  ContractiveDQN:   mean={st_con['mean_return']:.1f}±{st_con['std_return']:.1f}, "
          f"collapses={st_con['n_collapses']}/{len(st_con['final_returns_per_seed'])}, "
          f"variance={st_con['return_variance']:.1f}")
    print()
    print(f"Total wall-clock time: {(time.time()-t_start)/60:.1f} min")
    print("=" * 60 + "\n")

    print(f"Results saved in: {os.path.abspath(RESULTS_DIR)}/")


if __name__ == "__main__":
    main()
