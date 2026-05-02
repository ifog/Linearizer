"""Print a consolidated results table for the conference paper."""

import os
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLANNING_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "planning", "results")


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def fmt(val, std=None, decimals=1):
    if val is None:
        return "N/A"
    s = f"{val:.{decimals}f}"
    if std is not None:
        s += f"±{std:.{decimals}f}"
    return s


def main():
    print("\n" + "=" * 75)
    print("  CONTRACTIVE LINEARIZER — COMPLETE RESULTS SUMMARY")
    print("=" * 75)

    # ---- DQN ----
    print("\n[DQN Benchmark — CartPole-v1 (50k steps, 3 seeds)]")
    r = load_json(os.path.join(RESULTS_DIR, "cartpole_v1_dqn_results.json"))
    if r:
        std_val = r["standard"]["final_mean"]
        dbl_val = r["double"]["final_mean"]
        con_val = r["contractive"]["final_mean"]
        std_std = r["standard"]["final_std"]
        dbl_std = r["double"]["final_std"]
        con_std = r["contractive"]["final_std"]
        print(f"  StandardDQN:     {fmt(std_val, std_std)}")
        print(f"  DoubleDQN:       {fmt(dbl_val, dbl_std)}")
        print(f"  ContractiveDQN:  {fmt(con_val, con_std)}  (+{100*(con_val-std_val)/abs(std_val):.0f}% vs std)")

    print("\n[DQN Benchmark — Acrobot-v1 (100k steps, 3 seeds)]")
    r = load_json(os.path.join(RESULTS_DIR, "acrobot_v1_dqn_results.json"))
    if r:
        std_val = r["standard"]["final_mean"]
        dbl_val = r["double"]["final_mean"]
        con_val = r["contractive"]["final_mean"]
        print(f"  StandardDQN:     {fmt(std_val, r['standard']['final_std'])}")
        print(f"  DoubleDQN:       {fmt(dbl_val, r['double']['final_std'])}")
        pct = 100 * (con_val - std_val) / abs(std_val)
        print(f"  ContractiveDQN:  {fmt(con_val, r['contractive']['final_std'])}  (+{pct:.0f}% vs std)")

    print("\n[DQN Stability Tests]")
    div = load_json(os.path.join(RESULTS_DIR, "divergence_test_results.json"))
    if div:
        print(f"  Divergence test (no target net, lr=1e-2):")
        print(f"    StandardDQN:    |Q|={div['standard']['final_max_q_mean']:.0f} DIVERGED={div['standard']['diverged']}")
        print(f"    ContractiveDQN: |Q|={div['contractive']['final_max_q_mean']:.1f} DIVERGED={div['contractive']['diverged']}")
    stress = load_json(os.path.join(RESULTS_DIR, "stress_test_results.json"))
    if stress:
        print(f"  Stress test (20% poisoned buffer, 5 seeds, 0 collapses for both):")
        print(f"    StandardDQN:    {fmt(stress['standard']['mean_return'], stress['standard']['std_return'])}")
        print(f"    ContractiveDQN: {fmt(stress['contractive']['mean_return'], stress['contractive']['std_return'])}")

    # ---- PPO ----
    print("\n[PPO Benchmark — CartPole-v1 (200k steps, 3 seeds)]")
    r = load_json(os.path.join(RESULTS_DIR, "cartpole_v1_ppo_results.json"))
    if r:
        std_val = r["standard"]["final_mean"]
        con_val = r["contractive"]["final_mean"]
        pct = 100 * (con_val - std_val) / abs(std_val)
        sign = "+" if pct >= 0 else ""
        print(f"  StandardPPO:     {fmt(std_val, r['standard']['final_std'])}  value_var={fmt(r['standard'].get('mean_value_variance'), decimals=0)}")
        print(f"  ContractivePPO:  {fmt(con_val, r['contractive']['final_std'])}  value_var={fmt(r['contractive'].get('mean_value_variance'), decimals=0)}  ({sign}{pct:.0f}%)")

    print("\n[PPO Benchmark — Acrobot-v1 (500k steps, 3 seeds)]")
    r = load_json(os.path.join(RESULTS_DIR, "acrobot_v1_ppo_results.json"))
    if r:
        std_val = r["standard"]["final_mean"]
        con_val = r["contractive"]["final_mean"]
        pct = 100 * (con_val - std_val) / abs(std_val)
        sign = "+" if pct >= 0 else ""
        print(f"  StandardPPO:     {fmt(std_val, r['standard']['final_std'])}  value_var={fmt(r['standard'].get('mean_value_variance'), decimals=0)}")
        print(f"  ContractivePPO:  {fmt(con_val, r['contractive']['final_std'])}  value_var={fmt(r['contractive'].get('mean_value_variance'), decimals=0)}  ({sign}{pct:.0f}%)")

    print("\n[PPO Benchmark — Pendulum-v0 (300k steps, 3 seeds)]")
    r = load_json(os.path.join(RESULTS_DIR, "pendulum_v0_ppo_results.json"))
    if r:
        std_val = r["standard"]["final_mean"]
        con_val = r["contractive"]["final_mean"]
        pct = 100 * (con_val - std_val) / abs(std_val)
        sign = "+" if pct >= 0 else ""
        print(f"  StandardPPO:     {fmt(std_val, r['standard']['final_std'])}  value_var={fmt(r['standard'].get('mean_value_variance'), decimals=0)}")
        print(f"  ContractivePPO:  {fmt(con_val, r['contractive']['final_std'])}  value_var={fmt(r['contractive'].get('mean_value_variance'), decimals=0)}  ({sign}{pct:.0f}%)")
    else:
        print("  [Pending — run ppo/run_pendulum.py]")

    # ---- Planning ----
    print("\n[Planning — Exp A: Convergence (K=50 RMSE reduction)]")
    conv = load_json(os.path.join(PLANNING_RESULTS_DIR, "convergence_curves.json"))
    if conv:
        for name, errors in conv.items():
            if isinstance(errors, list) and len(errors) > 50:
                init = errors[0]
                final = errors[50]
                red = 100 * (init - final) / (init + 1e-9)
                print(f"  {name:<20}: init={init:.4f}, K=50={final:.4f}, reduction={red:.1f}%")

    print("\n[Planning — Exp B: Fast Iteration (K=50)]")
    fast = load_json(os.path.join(PLANNING_RESULTS_DIR, "fast_iteration_results.json"))
    if fast:
        for k_str, v in fast.items():
            if v["K"] == 50:
                print(f"  K=50: MSE={v['mean_mse_loop_vs_fast']:.2e}, speedup={v['speedup']:.1f}x")

    print("\n[Planning — Exp C: Planning Quality]")
    pq = load_json(os.path.join(PLANNING_RESULTS_DIR, "planning_quality.json"))
    if pq:
        for name, r in pq.items():
            print(f"  {name:<20}: success={r['success_rate']:.3f}")

    print("\n[Planning — Exp D: Generalization (20×20)]")
    gen = load_json(os.path.join(PLANNING_RESULTS_DIR, "generalization_results.json"))
    if gen:
        items = list(gen.items())
        if len(items) >= 2:
            names = [n for n, _ in items]
            mses = [v.get("mean_mse_20x20", float("nan")) for _, v in items]
            if not any(m != m for m in mses):  # no NaN
                better_idx = int(mses[1] < mses[0])
                worse_idx = 1 - better_idx
                ratio = mses[worse_idx] / mses[better_idx] if mses[better_idx] > 0 else float("inf")
                better_name = names[better_idx].replace("_20x20", "")
                print(f"  {better_name} is {ratio:.1f}× better (MSE {mses[better_idx]:.1f} vs {mses[worse_idx]:.1f})")
        for name, r in gen.items():
            mse = r.get("mean_mse_20x20", float("nan"))
            std = r.get("std_mse_20x20", float("nan"))
            print(f"  {name:<25}: MSE={mse:.4f}±{std:.4f}")

    print("\n[Planning — Exp F: Generalization (50×50)]")
    gen50 = load_json(os.path.join(PLANNING_RESULTS_DIR, "generalization_50x50_results.json"))
    if gen50:
        for name, r in gen50.items():
            mse = r.get("mean_mse_50x50", float("nan"))
            std = r.get("std_mse_50x50", float("nan"))
            if mse == mse:  # not NaN
                print(f"  {name:<25}: MSE={mse:.4f}±{std:.4f}")
            else:
                print(f"  {name:<25}: [Checkpoint missing — run train.py --grid_size 50]")
    else:
        print("  [Pending — run train.py --grid_size 50 --models contractive vin]")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
