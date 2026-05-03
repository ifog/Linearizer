"""
DQN on Atari (pixel inputs, standard preprocessing): StandardDQN vs ContractiveDQN.

Architecture follows the original DQN paper (Mnih et al., 2015):
  4-frame stack → Conv(32,8×8,s4) → Conv(64,4×4,s2) → Conv(64,3×3,s1) → FC(512)
  Contractive variant replaces the output head with a closed-form fixed-point head.

Run with:
  PYTHONPATH=/home/nvidia/.local/lib/python3.8/site-packages \
    /home/nvidia/anaconda3/envs/rlgpu/bin/python \
    contractive_rl/dqn/run_atari_dqn.py --env ALE/Pong-v5
"""

import os
import sys
import json
import random
import argparse
import collections
import time

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from contractive_rl.shared.contractive_operator import DiagonalContractiveOp

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import ale_py
gym.register_envs(ale_py)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

ENV_CFGS = {
    "ALE/Pong-v5":           {"n_actions": 6},
    "ALE/Breakout-v5":       {"n_actions": 4},
    "ALE/SpaceInvaders-v5":  {"n_actions": 6},
}

SEEDS = [0, 1, 2]
TRAIN_STEPS = 2_000_000
BUFFER_SIZE = 100_000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 1_000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_STEPS = 500_000
MIN_REPLAY = 10_000
LATENT_DIM = 256
SPECTRAL_BOUND = 0.9


def make_env(env_id, seed):
    env = gym.make(env_id, frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True,
                             scale_obs=False, screen_size=84)
    env = FrameStack(env, num_stack=4)
    return env


class ReplayBuffer:
    def __init__(self, capacity):
        self.states = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.pos = 0
        self.size = 0

    def push(self, s, a, r, sn, done):
        self.states[self.pos] = s
        self.actions[self.pos] = a
        self.rewards[self.pos] = np.clip(r, -1, 1)
        self.next_states[self.pos] = sn
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.tensor(self.states[idx], device=DEVICE).float() / 255.0,
            torch.tensor(self.actions[idx], device=DEVICE),
            torch.tensor(self.rewards[idx], device=DEVICE),
            torch.tensor(self.next_states[idx], device=DEVICE).float() / 255.0,
            torch.tensor(self.dones[idx], device=DEVICE),
        )

    def __len__(self):
        return self.size


class NatureCNN(nn.Module):
    """Shared CNN backbone from Mnih et al., 2015."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
        )
        self.out_dim = 512

    def forward(self, x):
        return self.conv(x)


class StandardQNet(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.cnn = NatureCNN()
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        return self.head(self.cnn(x))


class ContractiveQNet(nn.Module):
    """Nature CNN backbone with a Contractive Linearizer head."""

    def __init__(self, n_actions, latent_dim=LATENT_DIM, spectral_bound=SPECTRAL_BOUND):
        super().__init__()
        self.cnn = NatureCNN()
        feat_dim = 512
        self.A = DiagonalContractiveOp(context_dim=feat_dim, latent_dim=latent_dim,
                                       spectral_bound=spectral_bound)
        self.b_net = nn.Linear(feat_dim, latent_dim)
        self.head = nn.Linear(latent_dim, n_actions)

    def forward(self, x):
        feat = self.cnn(x)
        eigs = self.A.get_eigenvalues(feat)
        b = self.b_net(feat)
        z_star = b / (1.0 - eigs).clamp(min=1e-2)
        return self.head(z_star)


def train_one_seed(env_id, n_actions, variant, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    env = make_env(env_id, seed)
    obs, _ = env.reset(seed=seed)
    obs = np.array(obs, dtype=np.uint8)

    if variant == "standard":
        q_net = StandardQNet(n_actions).to(DEVICE)
        target_net = StandardQNet(n_actions).to(DEVICE)
    else:
        q_net = ContractiveQNet(n_actions).to(DEVICE)
        target_net = ContractiveQNet(n_actions).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    buf = ReplayBuffer(BUFFER_SIZE)

    episode_returns, ep_return = [], 0.0
    eps_fn = lambda s: EPS_END + (EPS_START - EPS_END) * max(0, 1 - s / EPS_DECAY_STEPS)
    step = 0
    t0 = time.time()

    while step < TRAIN_STEPS:
        if random.random() < eps_fn(step):
            action = env.action_space.sample()
        else:
            s_t = torch.tensor(obs[None], device=DEVICE).float() / 255.0
            with torch.no_grad():
                action = int(q_net(s_t).argmax(dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = np.array(next_obs, dtype=np.uint8)
        done = terminated or truncated
        buf.push(obs, action, reward, next_obs, float(done))
        ep_return += reward
        obs = next_obs
        step += 1

        if done:
            episode_returns.append(ep_return)
            ep_return = 0.0
            obs, _ = env.reset()
            obs = np.array(obs, dtype=np.uint8)

        if len(buf) >= MIN_REPLAY:
            s_b, a_b, r_b, sn_b, d_b = buf.sample(BATCH_SIZE)
            with torch.no_grad():
                q_next = target_net(sn_b).max(dim=1).values
                targets = r_b + GAMMA * q_next * (1.0 - d_b)

            q_pred = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
            loss = nn.functional.huber_loss(q_pred, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())

        if step % 200_000 == 0 and step > 0:
            recent = np.mean(episode_returns[-20:]) if len(episode_returns) >= 20 else (
                np.mean(episode_returns) if episode_returns else 0.0)
            print(f"    step={step//1000}k  recent_return={recent:.2f}  eps={eps_fn(step):.3f}  t={time.time()-t0:.0f}s",
                  flush=True)

    env.close()
    return episode_returns


def _smooth(arr, w=20):
    if len(arr) < w:
        return np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def run(env_id):
    cfg = ENV_CFGS[env_id]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = env_id.lower().replace("/", "_").replace("-", "_")

    print(f"Device: {DEVICE}  |  {env_id} (pixel, 4-frame stack)  |  {TRAIN_STEPS//1000}k steps  |  {len(SEEDS)} seeds")
    print("=" * 60)

    all_results = {}
    for variant in ("standard", "contractive"):
        print(f"\n[{variant}]")
        seed_returns = []
        for seed in SEEDS:
            print(f"  seed={seed}", flush=True)
            ep_rets = train_one_seed(env_id, cfg["n_actions"], variant, seed)
            seed_returns.append(ep_rets)
        all_results[variant] = seed_returns

        finals = [np.mean(r[-20:]) if len(r) >= 20 else (np.mean(r) if r else 0.)
                  for r in seed_returns]
        print(f"  -> final 20-ep mean={np.mean(finals):.2f} ± {np.std(finals):.2f}", flush=True)

    summary = {}
    for variant, seed_returns in all_results.items():
        finals = [np.mean(r[-20:]) if len(r) >= 20 else (np.mean(r) if r else 0.)
                  for r in seed_returns]
        summary[variant] = {
            "final_mean": float(np.mean(finals)),
            "final_std": float(np.std(finals)),
        }

    json_path = os.path.join(RESULTS_DIR, f"{tag}_dqn_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")

    colors = {"standard": "crimson", "contractive": "steelblue"}
    labels = {"standard": "StandardDQN", "contractive": "ContractiveDQN"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for variant, seed_returns in all_results.items():
        max_ep = max((len(r) for r in seed_returns), default=1)
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
        sm, ss = _smooth(mean), _smooth(std)
        xs = np.arange(len(sm))
        ax.plot(xs, sm, label=labels[variant], color=colors[variant], linewidth=2)
        ax.fill_between(xs, sm - ss, sm + ss, alpha=0.18, color=colors[variant])

    ax.set_xlabel("Episode"); ax.set_ylabel("Return (smoothed)")
    ax.set_title(f"{env_id} — DQN ({len(SEEDS)} seeds, {TRAIN_STEPS//1000}k steps)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"{tag}_dqn_results.png")
    fig.savefig(plot_path, dpi=120); plt.close(fig)
    print(f"Saved: {plot_path}")

    print(f"\n{'='*60}")
    print(f"{env_id} DQN RESULTS (final 20-ep mean ± std, {len(SEEDS)} seeds)")
    print(f"{'='*60}")
    for v, lbl in [("standard", "StandardDQN"), ("contractive", "ContractiveDQN")]:
        s = summary[v]
        pct = 100 * (summary["contractive"]["final_mean"] - summary["standard"]["final_mean"]) / max(abs(summary["standard"]["final_mean"]), 0.1)
        sign = "+" if pct >= 0 else ""
        if v == "contractive":
            print(f"  {lbl:<20s}: {s['final_mean']:.2f} ± {s['final_std']:.2f}  ({sign}{pct:.1f}%)")
        else:
            print(f"  {lbl:<20s}: {s['final_mean']:.2f} ± {s['final_std']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ALE/Pong-v5",
                        choices=list(ENV_CFGS.keys()))
    args = parser.parse_args()
    run(args.env)
