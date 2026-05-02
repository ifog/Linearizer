"""Run Acrobot-only DQN benchmark and save results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
torch.set_num_threads(1)

from contractive_rl.dqn.train_dqn import run_env, plot_and_save, RESULTS_DIR, SEEDS

results = run_env("Acrobot-v1", seeds=SEEDS)
summary = plot_and_save("Acrobot-v1", results, RESULTS_DIR)
print("\nFinal:", summary)
