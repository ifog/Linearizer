"""Run PPO on Pendulum-v0 only (standard vs contractive), save results."""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contractive_rl.ppo.train_ppo import (
    run_env, plot_and_save, RESULTS_DIR, ENV_CONFIGS, SEEDS
)

def main():
    out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    env_id = "Pendulum-v0"
    results = run_env(env_id, seeds=SEEDS)
    summary = plot_and_save(env_id, results, out_dir)

    print("\n" + "=" * 55)
    print("  PENDULUM-v0 PPO RESULTS")
    print("=" * 55)
    for v, label in [("standard", "StandardPPO"), ("contractive", "ContractivePPO")]:
        r = summary.get(v, {})
        print(f"  {label:<20s}: {r.get('final_mean',0):.1f} ± {r.get('final_std',0):.1f}"
              f"  value_var={r.get('mean_value_variance',0):.1f}")
    print()

if __name__ == "__main__":
    main()
