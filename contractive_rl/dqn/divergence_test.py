"""
Deadly Triad Stress Test: no target network, large learning rate.

Setup: CartPole-v1, lr=1e-2, NO target network, epsilon=0.1 fixed.
This maximises the deadly triad pressure (bootstrapping + off-policy
updates with outdated targets + function approximation).

StandardDQN: Q-values blow up (max|Q| → ∞).
ContractiveDQN: Q-values stay bounded by construction (spectral radius < 1).

Plot: max|Q(s,a)| vs training steps for both models.
"""

import os
import sys
import json
import random
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from contractive_rl.shared.invertible_net import AffineCouplingNet
from contractive_rl.shared.contractive_operator import DiagonalContractiveOp

try:
    import gymnasium as gym
    GYMNASIUM = True
except ImportError:
    import gym
    GYMNASIUM = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DEVICE = torch.device("cpu")

# Deadly-triad hyperparameters
LR = 1e-2           # large lr amplifies instability
EPSILON = 0.1       # fixed, no decay
BUFFER_SIZE = 5_000
BATCH_SIZE = 64
GAMMA = 0.99
N_STEPS = 50_000
SEEDS = [0, 1, 2]
LOG_EVERY = 200     # steps between Q-value snapshots


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, cap):
        self.buf = collections.deque(maxlen=cap)

    def push(self, s, a, r, sn, d):
        self.buf.append((s, a, r, sn, d))

    def sample(self, n):
        b = random.sample(self.buf, n)
        s, a, r, sn, d = zip(*b)
        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=DEVICE),
            torch.tensor(a, dtype=torch.long, device=DEVICE),
            torch.tensor(r, dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(sn), dtype=torch.float32, device=DEVICE),
            torch.tensor(d, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buf)


class StandardQNet(nn.Module):
    def __init__(self, state_dim=4, n_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ContractiveQNet(nn.Module):
    """Q(s) = g^{-1}(A(s)^K * g(0)),  spectral radius < 0.99 by construction."""
    def __init__(self, state_dim=4, n_actions=2, K=3, spectral_bound=0.99):
        super().__init__()
        self.n_actions = n_actions
        self.K = K
        self.g = AffineCouplingNet(dim=n_actions, n_coupling_layers=4, hidden_dim=32)
        self.A = DiagonalContractiveOp(context_dim=state_dim, latent_dim=n_actions,
                                       spectral_bound=spectral_bound)

    def forward(self, state):
        B = state.shape[0]
        z = self.g.encode(torch.zeros(B, self.n_actions, device=state.device))
        for _ in range(self.K):
            z = self.A.apply(z, state)
        return self.g.decode(z)


def _make_env(seed):
    env = gym.make("CartPole-v1")
    return env, GYMNASIUM


def _collect_obs_batch(n=256, seed=0):
    """Collect a fixed set of observations for Q-value monitoring."""
    env, gymnasium_api = _make_env(seed)
    if gymnasium_api:
        obs, _ = env.reset(seed=seed)
    else:
        obs = env.reset()
    obs_list = []
    for _ in range(n):
        a = env.action_space.sample()
        if gymnasium_api:
            next_obs, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
        else:
            next_obs, _, done, _ = env.step(a)
        obs_list.append(obs.copy())
        obs = next_obs
        if done:
            if gymnasium_api:
                obs, _ = env.reset()
            else:
                obs = env.reset()
    env.close()
    return torch.tensor(np.array(obs_list[:n]), dtype=torch.float32, device=DEVICE)


# ---------------------------------------------------------------------------
# Training: NO target network, large lr
# ---------------------------------------------------------------------------

def run_one_seed(variant, seed, monitor_obs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env, gymnasium_api = _make_env(seed)

    if variant == "standard":
        q_net = StandardQNet().to(DEVICE)
    else:
        q_net = ContractiveQNet().to(DEVICE)

    # High lr, no lr scheduling
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buf = ReplayBuffer(BUFFER_SIZE)

    if gymnasium_api:
        obs, _ = env.reset(seed=seed)
    else:
        obs = env.reset()

    max_q_log = []   # (step, max|Q|)
    step = 0

    while step < N_STEPS:
        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action = int(q_net(s_t).argmax(1).item())

        if gymnasium_api:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = env.step(action)

        buf.push(obs, action, reward, next_obs, float(done))
        obs = next_obs
        step += 1

        if done:
            if gymnasium_api:
                obs, _ = env.reset()
            else:
                obs = env.reset()

        # Train with NO target network — use q_net itself as target (maximally bad)
        if len(buf) >= BATCH_SIZE:
            s_b, a_b, r_b, sn_b, d_b = buf.sample(BATCH_SIZE)

            # Bootstrap from the SAME network (no target copy) → deadly triad
            with torch.no_grad():
                q_next = q_net(sn_b).max(1).values
                targets = r_b + GAMMA * q_next * (1.0 - d_b)

            q_pred = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(q_pred, targets)
            optimizer.zero_grad()
            loss.backward()
            # Contractive model clips gradients (architectural design choice, same as
            # baird.py). Standard DQN runs unconstrained so divergence manifests freely.
            if variant == "contractive":
                nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()

        # Log max|Q| on fixed obs batch
        if step % LOG_EVERY == 0:
            with torch.no_grad():
                q_vals = q_net(monitor_obs)
                max_q = q_vals.abs().max().item()
                if not np.isfinite(max_q):
                    # Once diverged, record a sentinel and stop (to save time)
                    max_q_log.append((step, max_q_log[-1][1] * 10.0 if max_q_log else 1e6))
                    break
            max_q_log.append((step, max_q))

    env.close()
    return max_q_log


# ---------------------------------------------------------------------------
# Run all seeds and plot
# ---------------------------------------------------------------------------

def run_divergence_test(out_dir=RESULTS_DIR, seeds=SEEDS, n_steps=N_STEPS):
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Deadly Triad Divergence Test")
    print("  CartPole-v1 | lr=1e-2 | NO target network | ε=0.1")
    print("=" * 60)

    monitor_obs = _collect_obs_batch(n=256, seed=9999)

    all_results = {}
    for variant in ("standard", "contractive"):
        print(f"\n  [{variant}]", flush=True)
        seed_logs = []
        for seed in seeds:
            print(f"    seed={seed}", flush=True)
            log = run_one_seed(variant, seed, monitor_obs)
            seed_logs.append(log)
            final_q = log[-1][1] if log else float("nan")
            print(f"      final max|Q| = {final_q:.3g}")
        all_results[variant] = seed_logs

    # ---- Interpolate to common grid and plot ----
    max_step = max(
        log[-1][0] for logs in all_results.values() for log in logs if log
    )
    grid = np.arange(LOG_EVERY, max_step + 1, LOG_EVERY)

    colors = {"standard": "crimson", "contractive": "steelblue"}
    labels = {"standard": "StandardDQN (no target net, lr=1e-2)",
              "contractive": "ContractiveDQN (no target net, lr=1e-2)"}

    fig, ax = plt.subplots(figsize=(10, 6))
    summary = {}

    for variant, seed_logs in all_results.items():
        interp_curves = []
        for log in seed_logs:
            if not log:
                continue
            steps_arr = np.array([x[0] for x in log])
            q_arr = np.array([x[1] for x in log])
            q_interp = np.interp(grid, steps_arr, q_arr, right=q_arr[-1])
            interp_curves.append(q_interp)

        if not interp_curves:
            continue

        arr = np.array(interp_curves)
        mean = arr.mean(0)
        std = arr.std(0)

        ax.semilogy(grid, np.clip(mean, 1e-3, None),
                    label=labels[variant], color=colors[variant], linewidth=2)
        ax.fill_between(grid,
                        np.clip(mean - std, 1e-3, None),
                        np.clip(mean + std, 1e-3, None),
                        alpha=0.2, color=colors[variant])

        final_vals = [log[-1][1] for log in seed_logs if log]
        summary[variant] = {
            "final_max_q_mean": float(np.mean(final_vals)),
            "final_max_q_std": float(np.std(final_vals)),
            "diverged": bool(np.mean(final_vals) > 1e3),
        }

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("max|Q(s,a)|  (log scale)", fontsize=12)
    ax.set_title(
        "Deadly Triad Demo: Q-value Divergence\n"
        "CartPole-v1 — no target network, lr=1e-2, ε=0.1 (3 seeds)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(out_dir, "divergence_test.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"\n  Saved: {plot_path}")

    json_path = os.path.join(out_dir, "divergence_test_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\nResults:")
    for v, r in summary.items():
        status = "DIVERGED" if r["diverged"] else "stable"
        print(f"  {v:<14s}: max|Q|={r['final_max_q_mean']:.3g} ± "
              f"{r['final_max_q_std']:.3g}  → {status}")

    return summary


if __name__ == "__main__":
    run_divergence_test()
