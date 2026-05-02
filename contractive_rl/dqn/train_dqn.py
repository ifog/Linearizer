"""
DQN benchmark: StandardDQN vs DoubleDQN vs ContractiveDQN.

Environments: CartPole-v1 (50k steps) and Acrobot-v1 (100k steps).
3 seeds, epsilon-greedy, replay buffer 10k, target network every 500 steps.

Demonstrates that ContractiveDQN (spectral radius < 1 by construction)
matches or beats standard DQN variants while being provably stable.
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

# Single-threaded is faster on CPU for small tensors (avoids thread-spawn overhead)
torch.set_num_threads(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from contractive_rl.shared.contractive_operator import DiagonalContractiveOp

try:
    import gymnasium as gym
    def _make_env(env_id, seed):
        e = gym.make(env_id)
        return e, True
except ImportError:
    import gym
    def _make_env(env_id, seed):
        e = gym.make(env_id)
        return e, False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# ENV CONFIGS
# ---------------------------------------------------------------------------

ENV_CONFIGS = {
    "CartPole-v1": {
        "state_dim": 4,
        "n_actions": 2,
        "train_steps": 50_000,
        "target_update_freq": 500,
        "lr": 1e-3,
        "eps_decay_steps": 5_000,
        "min_replay": 200,
    },
    "Acrobot-v1": {
        "state_dim": 6,
        "n_actions": 3,
        "train_steps": 100_000,
        "target_update_freq": 500,
        "lr": 5e-4,
        "eps_decay_steps": 10_000,
        "min_replay": 500,
    },
}

BUFFER_SIZE = 10_000
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, sn, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=DEVICE),
            torch.tensor(a, dtype=torch.long, device=DEVICE),
            torch.tensor(r, dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(sn), dtype=torch.float32, device=DEVICE),
            torch.tensor(d, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# Q-networks
# ---------------------------------------------------------------------------

class StandardQNet(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ContractiveQNet(nn.Module):
    """
    Q(s,a) = head(z*)[a]  where  z* = b(s) / (1 - eigs(s))

    Closed-form fixed point of T(z) = eigs(s)*z + b(s).
    Spectral radius of A bounded by spectral_bound < 1 → unique fixed point.
    No bijection: z* fed directly to linear head (mirrors ContractiveCritic in PPO).
    """
    def __init__(self, state_dim, n_actions, latent_dim=32, spectral_bound=0.9):
        super().__init__()
        self.n_actions = n_actions
        self.A = DiagonalContractiveOp(context_dim=state_dim, latent_dim=latent_dim,
                                       spectral_bound=spectral_bound)
        self.b_net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.head = nn.Linear(latent_dim, n_actions)

    def forward(self, state):
        eigs = self.A.get_eigenvalues(state)          # (B, latent_dim)
        b = self.b_net(state)                          # (B, latent_dim)
        z_star = b / (1.0 - eigs).clamp(min=1e-2)    # closed-form fixed point
        return self.head(z_star)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _eps(step, decay_steps):
    return EPS_START + (EPS_END - EPS_START) * min(step / decay_steps, 1.0)


def train_one_seed(env_id, variant, seed, cfg):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env, gymnasium_api = _make_env(env_id, seed)
    state_dim = cfg["state_dim"]
    n_actions = cfg["n_actions"]
    n_steps = cfg["train_steps"]
    target_freq = cfg["target_update_freq"]
    lr = cfg["lr"]
    eps_decay = cfg["eps_decay_steps"]
    min_replay = cfg["min_replay"]

    if variant in ("standard", "double"):
        q_net = StandardQNet(state_dim, n_actions).to(DEVICE)
        target_net = StandardQNet(state_dim, n_actions).to(DEVICE)
    else:
        q_net = ContractiveQNet(state_dim, n_actions).to(DEVICE)
        target_net = ContractiveQNet(state_dim, n_actions).to(DEVICE)

    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buf = ReplayBuffer(BUFFER_SIZE)

    if gymnasium_api:
        obs, _ = env.reset(seed=seed)
    else:
        obs = env.reset()

    episode_returns = []
    ep_return = 0.0
    step = 0

    while step < n_steps:
        eps = _eps(step, eps_decay)
        if random.random() < eps:
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
        ep_return += reward
        obs = next_obs
        step += 1

        if done:
            episode_returns.append(ep_return)
            ep_return = 0.0
            if gymnasium_api:
                obs, _ = env.reset()
            else:
                obs = env.reset()

        if len(buf) >= min_replay:
            s_b, a_b, r_b, sn_b, d_b = buf.sample(BATCH_SIZE)
            with torch.no_grad():
                if variant == "double":
                    next_a = q_net(sn_b).argmax(1)
                    q_next = target_net(sn_b).gather(1, next_a.unsqueeze(1)).squeeze(1)
                else:
                    q_next = target_net(sn_b).max(1).values
                targets = r_b + GAMMA * q_next * (1.0 - d_b)

            q_pred = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(q_pred, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

        if step % target_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

    env.close()
    return episode_returns


def run_env(env_id, seeds=SEEDS):
    cfg = ENV_CONFIGS[env_id]
    print(f"\n{'='*55}")
    print(f"  {env_id}  ({cfg['train_steps']//1000}k steps, {len(seeds)} seeds)")
    print(f"{'='*55}")
    results = {}
    for variant in ("standard", "double", "contractive"):
        print(f"  [{variant}]", flush=True)
        all_returns = []
        for seed in seeds:
            print(f"    seed={seed}", flush=True)
            r = train_one_seed(env_id, variant, seed, cfg)
            all_returns.append(r)
        results[variant] = all_returns

        finals = [np.mean(r[-10:]) if len(r) >= 10 else (np.mean(r) if r else 0.)
                  for r in all_returns]
        print(f"    final 10-ep mean={np.mean(finals):.1f} ± {np.std(finals):.1f}")

    return results


def _smooth(arr, w=15):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def plot_and_save(env_id, results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    colors = {"standard": "crimson", "double": "darkorange", "contractive": "steelblue"}
    labels = {"standard": "StandardDQN", "double": "DoubleDQN",
              "contractive": "ContractiveDQN"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant, seed_returns in results.items():
        max_ep = max(len(r) for r in seed_returns)
        interp = []
        for r in seed_returns:
            if not r:
                interp.append(np.zeros(max_ep))
            else:
                xo = np.linspace(0, 1, len(r))
                xn = np.linspace(0, 1, max_ep)
                interp.append(np.interp(xn, xo, r))
        arr = np.array(interp)
        mean, std = arr.mean(0), arr.std(0)
        sm = _smooth(mean)
        ss = _smooth(std)
        xs = np.arange(len(sm))
        ax.plot(xs, sm, label=labels[variant], color=colors[variant], linewidth=2)
        ax.fill_between(xs, sm - ss, sm + ss, alpha=0.18, color=colors[variant])

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Return (smoothed)", fontsize=12)
    ax.set_title(f"{env_id} — DQN Variants ({len(SEEDS)} seeds)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    safe_id = env_id.replace("-", "_").lower()
    path = os.path.join(out_dir, f"{safe_id}_learning_curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")

    # JSON summary
    summary = {}
    for variant, seed_returns in results.items():
        finals = [np.mean(r[-10:]) if len(r) >= 10 else (np.mean(r) if r else 0.)
                  for r in seed_returns]
        summary[variant] = {
            "final_mean": float(np.mean(finals)),
            "final_std": float(np.std(finals)),
        }
    json_path = os.path.join(out_dir, f"{safe_id}_dqn_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")
    return summary


def run_train_dqn(out_dir=RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    all_summaries = {}
    for env_id in ("CartPole-v1", "Acrobot-v1"):
        results = run_env(env_id, seeds=SEEDS)
        summary = plot_and_save(env_id, results, out_dir)
        all_summaries[env_id] = summary

    # Print combined table
    print("\n" + "=" * 55)
    print("  DQN BENCHMARK SUMMARY")
    print("=" * 55)
    labels = {"standard": "StandardDQN", "double": "DoubleDQN",
              "contractive": "ContractiveDQN"}
    for env_id, summ in all_summaries.items():
        print(f"\n  {env_id}:")
        for v, label in labels.items():
            r = summ.get(v, {})
            print(f"    {label:<20s}: {r.get('final_mean', 0):.1f} ± {r.get('final_std', 0):.1f}")
    print()
    return all_summaries


if __name__ == "__main__":
    run_train_dqn()
