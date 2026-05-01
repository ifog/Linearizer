"""
Instability Stress Test: Amplifying the Deadly Triad via a Poisoned Replay Buffer.

A poisoned replay buffer (20% bad-policy transitions) creates an off-policy
distribution mismatch that amplifies divergence pressure. StandardDQN shows
higher variance / more collapses; ContractiveDQN remains stable.
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
from contractive_rl.dqn.cartpole import (
    StandardQNet, ContractiveQNet, ReplayBuffer,
    BUFFER_SIZE, BATCH_SIZE, GAMMA, TARGET_UPDATE_FREQ,
    EPS_START, EPS_END, EPS_DECAY_STEPS, LEARNING_RATE,
    MIN_REPLAY_SIZE, DEVICE, GYM_RESET_KWARGS, _make_env, epsilon
)

try:
    import gymnasium as gym
except ImportError:
    import gym

STRESS_STEPS = 50_000
POISON_RATIO = 0.20    # fraction of buffer transitions from bad policy
COLLAPSE_THRESHOLD = 50.0  # episode return below this = "collapse"
SEEDS = list(range(5))


# ---------------------------------------------------------------------------
# Pre-collect bad policy transitions (always push right = action 1)
# ---------------------------------------------------------------------------

def collect_bad_transitions(n: int = 2000) -> list:
    """Collect transitions from a bad policy (always push right)."""
    env = _make_env(seed=999)
    if GYM_RESET_KWARGS:
        obs, _ = env.reset(seed=999)
    else:
        obs = env.reset()

    transitions = []
    for _ in range(n):
        action = 1  # always push right
        if GYM_RESET_KWARGS:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = env.step(action)

        transitions.append((obs.copy(), action, reward, next_obs.copy(), float(done)))
        obs = next_obs
        if done:
            if GYM_RESET_KWARGS:
                obs, _ = env.reset()
            else:
                obs = env.reset()

    env.close()
    return transitions


def select_action_stress(q_net, state, step, n_actions=2):
    if random.random() < epsilon(step):
        return random.randrange(n_actions)
    s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return int(q_net(s).argmax(dim=1).item())


# ---------------------------------------------------------------------------
# Poisoned DQN training
# ---------------------------------------------------------------------------

def train_poisoned_dqn(q_net: nn.Module, target_net: nn.Module,
                       seed: int, variant: str,
                       bad_transitions: list,
                       n_steps: int = STRESS_STEPS) -> dict:
    """
    DQN training with a poisoned replay buffer.
    20% of the buffer is pre-filled with bad-policy transitions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(seed)
    if GYM_RESET_KWARGS:
        obs, _ = env.reset(seed=seed)
    else:
        obs = env.reset()

    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)

    # Poisoned buffer: pre-load with bad transitions
    buf = ReplayBuffer(BUFFER_SIZE)
    n_poison = int(BUFFER_SIZE * POISON_RATIO)
    for t in random.sample(bad_transitions, min(n_poison, len(bad_transitions))):
        buf.push(*t)

    episode_returns = []
    ep_return = 0.0
    step = 0

    while step < n_steps:
        action = select_action_stress(q_net, obs, step)

        if GYM_RESET_KWARGS:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = env.step(action)

        buf.push(obs, action, reward, next_obs, float(done))
        ep_return += reward
        obs = next_obs
        step += 1

        if done:
            if GYM_RESET_KWARGS:
                obs, _ = env.reset()
            else:
                obs = env.reset()
            episode_returns.append(ep_return)
            ep_return = 0.0

        if len(buf) >= MIN_REPLAY_SIZE:
            s_b, a_b, r_b, sn_b, d_b = buf.sample(BATCH_SIZE)
            with torch.no_grad():
                q_next = target_net(sn_b).max(dim=1).values
                targets = r_b + GAMMA * q_next * (1.0 - d_b)

            q_pred = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(q_pred, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())

    env.close()
    return {"episode_returns": episode_returns}


def run_stress_test(out_dir: str = "results",
                    n_steps: int = STRESS_STEPS) -> dict:
    """Run the stress test for both variants across all seeds."""
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print(f"Instability Stress Test (poisoned buffer, {n_steps} steps)")
    print(f"  Poison ratio: {POISON_RATIO*100:.0f}% bad-policy transitions")
    print("=" * 60)

    print("  Collecting bad-policy transitions...", flush=True)
    bad_transitions = collect_bad_transitions(n=3000)
    print(f"  Collected {len(bad_transitions)} bad transitions")

    all_results = {}
    for variant in ("standard", "contractive"):
        print(f"\n  Variant: {variant}", flush=True)
        seed_returns = []
        for seed in SEEDS:
            print(f"    seed={seed}", flush=True)
            if variant == "standard":
                q_net = StandardQNet().to(DEVICE)
                target_net = StandardQNet().to(DEVICE)
            else:
                q_net = ContractiveQNet().to(DEVICE)
                target_net = ContractiveQNet().to(DEVICE)
            target_net.load_state_dict(q_net.state_dict())

            res = train_poisoned_dqn(q_net, target_net, seed=seed,
                                     variant=variant,
                                     bad_transitions=bad_transitions,
                                     n_steps=n_steps)
            seed_returns.append(res["episode_returns"])

        all_results[variant] = seed_returns

    # --- Compute statistics ---
    summary = {}
    for variant, seed_returns in all_results.items():
        final_returns = [
            np.mean(r[-10:]) if len(r) >= 10 else (np.mean(r) if r else 0.0)
            for r in seed_returns
        ]
        collapses = sum(1 for r in final_returns if r < COLLAPSE_THRESHOLD)
        variance = float(np.var(final_returns))
        summary[variant] = {
            "final_returns_per_seed": [float(x) for x in final_returns],
            "mean_return": float(np.mean(final_returns)),
            "std_return": float(np.std(final_returns)),
            "return_variance": variance,
            "n_collapses": collapses,
            "collapse_threshold": COLLAPSE_THRESHOLD,
        }
        print(f"\n  {variant}: mean={np.mean(final_returns):.1f}±{np.std(final_returns):.1f}, "
              f"collapses={collapses}/{len(SEEDS)}, variance={variance:.1f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"standard": "crimson", "contractive": "steelblue"}
    labels = {"standard": "StandardDQN", "contractive": "ContractiveDQN"}

    for variant, seed_returns in all_results.items():
        max_eps = max((len(r) for r in seed_returns), default=1)
        interp = []
        for r in seed_returns:
            if len(r) == 0:
                interp.append(np.zeros(max_eps))
            else:
                xo = np.linspace(0, 1, len(r))
                xn = np.linspace(0, 1, max_eps)
                interp.append(np.interp(xn, xo, r))
        arr = np.array(interp)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        w = max(1, max_eps // 50)
        if len(mean) >= w:
            smooth_m = np.convolve(mean, np.ones(w) / w, mode="valid")
            smooth_s = np.convolve(std, np.ones(w) / w, mode="valid")
            xs = np.arange(len(smooth_m))
        else:
            smooth_m, smooth_s, xs = mean, std, np.arange(len(mean))

        axes[0].plot(xs, smooth_m, label=labels[variant],
                     color=colors[variant], linewidth=2)
        axes[0].fill_between(xs,
                             np.clip(smooth_m - smooth_s, 0, None),
                             smooth_m + smooth_s,
                             alpha=0.25, color=colors[variant])

    axes[0].set_title("Poisoned Buffer: Learning Curves")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return (smoothed)")
    axes[0].legend()
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)

    # Bar plot: variance + collapses
    variants = ["standard", "contractive"]
    variances = [summary[v]["return_variance"] for v in variants]
    collapses = [summary[v]["n_collapses"] for v in variants]
    x = np.arange(len(variants))
    bar_colors = [colors[v] for v in variants]

    ax2 = axes[1]
    bars = ax2.bar(x - 0.2, variances, width=0.35, color=bar_colors, alpha=0.8,
                   label="Return Variance")
    ax2.set_ylabel("Return Variance", color="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[v] for v in variants])
    ax2.set_title("Stability Metrics (Poisoned Buffer)")
    ax2.grid(True, alpha=0.3, axis="y")

    ax3 = ax2.twinx()
    ax3.bar(x + 0.2, collapses, width=0.35, color=bar_colors, alpha=0.4,
            hatch="//", label="# Collapses")
    ax3.set_ylabel(f"# Collapses (return < {COLLAPSE_THRESHOLD})", color="gray")
    ax3.set_ylim(0, len(SEEDS) + 0.5)

    fig.suptitle(f"Stress Test: {POISON_RATIO*100:.0f}% Poisoned Replay Buffer\n"
                 f"(off-policy distribution mismatch amplifies deadly triad)",
                 fontsize=11)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "stress_test_plot.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"\n  Saved: {plot_path}")

    json_path = os.path.join(out_dir, "stress_test_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    return summary


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    run_stress_test(out_dir=results_dir)
