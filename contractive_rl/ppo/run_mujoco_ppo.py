"""
PPO on MuJoCo continuous-control envs: StandardPPO vs ContractivePPO.

Usage:
  PYTHONPATH=/home/nvidia/.local/lib/python3.8/site-packages \
    /home/nvidia/anaconda3/envs/rlgpu/bin/python contractive_rl/ppo/run_mujoco_ppo.py \
    --env Hopper-v4
"""

import os
import sys
import json
import random
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from contractive_rl.ppo.models import ContractiveCritic, StandardCritic, GaussianActor

import gymnasium as gym

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

BASE_CFG = {
    "train_steps": 1_000_000,
    "rollout_len": 2048,
    "n_epochs": 10,
    "mini_batch": 64,
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "clip_eps": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

ENV_CFGS = {
    "Hopper-v4":    {"state_dim": 11, "action_dim": 3},
    "Walker2d-v4":  {"state_dim": 17, "action_dim": 6},
    "HalfCheetah-v4": {"state_dim": 17, "action_dim": 6},
    "Ant-v4":       {"state_dim": 27, "action_dim": 8},
}

HIDDEN_DIM = 256
SPECTRAL_BOUND = 0.9
SEEDS = [0, 1, 2]


class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones = [], [], []

    def clear(self):
        self.__init__()

    def add(self, s, a, r, lp, v, d):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(float(r))
        self.log_probs.append(float(lp))
        self.values.append(float(v))
        self.dones.append(float(d))

    def compute_returns(self, last_value, gamma, gae_lambda):
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else self.values[t + 1]
            next_done = self.dones[t]
            delta = self.rewards[t] + gamma * next_val * (1 - next_done) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def to_tensors(self, advantages, returns):
        s = torch.tensor(np.array(self.states), dtype=torch.float32, device=DEVICE)
        a = torch.tensor(np.array(self.actions), dtype=torch.float32, device=DEVICE)
        lp = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=DEVICE)
        adv = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        ret = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        return s, a, lp, adv, ret


def train_one_seed(env_id, variant, seed, cfg):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    env = gym.make(env_id)

    actor = GaussianActor(cfg["state_dim"], cfg["action_dim"], HIDDEN_DIM).to(DEVICE)
    if variant == "standard":
        critic = StandardCritic(cfg["state_dim"], HIDDEN_DIM).to(DEVICE)
    else:
        critic = ContractiveCritic(cfg["state_dim"], HIDDEN_DIM, SPECTRAL_BOUND).to(DEVICE)

    opt_actor = optim.Adam(actor.parameters(), lr=cfg["lr_actor"])
    opt_critic = optim.Adam(critic.parameters(), lr=cfg["lr_critic"])

    obs, _ = env.reset(seed=seed)
    buf = RolloutBuffer()
    episode_returns, ep_return = [], 0.0
    value_log = []
    step = 0
    t0 = time.time()

    while step < cfg["train_steps"]:
        buf.clear()
        for _ in range(cfg["rollout_len"]):
            s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                dist = actor(s_t)
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t).sum(-1).item()
                action = action_t.squeeze(0).cpu().numpy()
                value = critic(s_t).item()

            value_log.append(abs(value))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buf.add(obs, action, reward, log_prob, value, float(done))
            ep_return += reward
            obs = next_obs
            step += 1
            if done:
                episode_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()

        with torch.no_grad():
            last_s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            last_val = critic(last_s).item()

        advantages, returns = buf.compute_returns(last_val, cfg["gamma"], cfg["gae_lambda"])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        s_b, a_b, lp_old, adv_b, ret_b = buf.to_tensors(advantages, returns)

        n = len(s_b)
        for _ in range(cfg["n_epochs"]):
            idx = torch.randperm(n, device=DEVICE)
            for start in range(0, n, cfg["mini_batch"]):
                mb = idx[start:start + cfg["mini_batch"]]
                dist = actor(s_b[mb])
                lp_new = dist.log_prob(a_b[mb]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                ratio = (lp_new - lp_old[mb]).exp()
                surr = torch.min(ratio * adv_b[mb],
                                 ratio.clamp(1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv_b[mb])
                actor_loss = -surr.mean() - cfg["ent_coef"] * entropy
                opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), cfg["max_grad_norm"])
                opt_actor.step()

                v_pred = critic(s_b[mb])
                critic_loss = nn.functional.mse_loss(v_pred, ret_b[mb])
                opt_critic.zero_grad()
                (cfg["vf_coef"] * critic_loss).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), cfg["max_grad_norm"])
                opt_critic.step()

        if step % 100_000 == 0:
            recent = np.mean(episode_returns[-20:]) if len(episode_returns) >= 20 else (
                np.mean(episode_returns) if episode_returns else 0.0)
            print(f"    step={step//1000}k  recent_return={recent:.1f}  t={time.time()-t0:.0f}s",
                  flush=True)

    env.close()
    return episode_returns, value_log


def _smooth(arr, w=30):
    if len(arr) < w:
        return np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def run(env_id):
    cfg = {**BASE_CFG, **ENV_CFGS[env_id]}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = env_id.lower().replace("-", "_")

    print(f"Device: {DEVICE}  |  {env_id}  |  {cfg['train_steps']//1000}k steps  |  hidden={HIDDEN_DIM}")
    print(f"{'='*60}")

    all_results = {}
    for variant in ("standard", "contractive"):
        print(f"\n[{variant}]")
        seed_returns, seed_val_logs = [], []
        for seed in SEEDS:
            print(f"  seed={seed}", flush=True)
            ep_rets, val_log = train_one_seed(env_id, variant, seed, cfg)
            seed_returns.append(ep_rets)
            seed_val_logs.append(val_log)
        all_results[variant] = {"returns": seed_returns, "value_logs": seed_val_logs}

        finals = [np.mean(r[-20:]) if len(r) >= 20 else (np.mean(r) if r else 0.)
                  for r in seed_returns]
        value_vars = [float(np.var(vl)) for vl in seed_val_logs]
        print(f"  -> final 20-ep mean={np.mean(finals):.1f} ± {np.std(finals):.1f}  "
              f"value_var={np.mean(value_vars):.1f}", flush=True)

    summary = {}
    for variant, data in all_results.items():
        finals = [np.mean(r[-20:]) if len(r) >= 20 else (np.mean(r) if r else 0.)
                  for r in data["returns"]]
        value_vars = [float(np.var(vl)) for vl in data["value_logs"]]
        summary[variant] = {
            "final_mean": float(np.mean(finals)),
            "final_std": float(np.std(finals)),
            "mean_value_variance": float(np.mean(value_vars)),
        }

    json_path = os.path.join(RESULTS_DIR, f"{tag}_ppo_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")

    colors = {"standard": "crimson", "contractive": "steelblue"}
    labels = {"standard": "StandardPPO", "contractive": "ContractivePPO"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for variant, data in all_results.items():
        seed_returns = data["returns"]
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
        sm, ss = _smooth(mean), _smooth(std)
        xs = np.arange(len(sm))
        axes[0].plot(xs, sm, label=labels[variant], color=colors[variant], linewidth=2)
        axes[0].fill_between(xs, sm - ss, sm + ss, alpha=0.18, color=colors[variant])

    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Return (smoothed)")
    axes[0].set_title(f"{env_id} — PPO ({len(SEEDS)} seeds)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    var_means = {v: summary[v]["mean_value_variance"] for v in summary}
    x = np.arange(len(var_means))
    axes[1].bar(x, list(var_means.values()),
                color=[colors[v] for v in var_means], alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[v] for v in var_means])
    axes[1].set_ylabel("Value Estimate Variance"); axes[1].set_title("Critic Stability")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"ContractivePPO: V*(s)=b(s)/(1-eigs(s)), |eigs|<{SPECTRAL_BOUND}  |  hidden={HIDDEN_DIM}",
                 fontsize=10)
    fig.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"{tag}_ppo_results.png")
    fig.savefig(plot_path, dpi=120); plt.close(fig)
    print(f"Saved: {plot_path}")

    print(f"\n{'='*60}")
    print(f"{env_id.upper()} PPO RESULTS")
    print(f"{'='*60}")
    for v, lbl in [("standard", "StandardPPO"), ("contractive", "ContractivePPO")]:
        s = summary[v]
        pct = 100 * (summary["contractive"]["final_mean"] - summary["standard"]["final_mean"]) / max(abs(summary["standard"]["final_mean"]), 1)
        sign = "+" if pct >= 0 else ""
        if v == "contractive":
            print(f"  {lbl:<20s}: {s['final_mean']:.1f} ± {s['final_std']:.1f}  value_var={s['mean_value_variance']:.0f}  ({sign}{pct:.1f}%)")
        else:
            print(f"  {lbl:<20s}: {s['final_mean']:.1f} ± {s['final_std']:.1f}  value_var={s['mean_value_variance']:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v4", choices=list(ENV_CFGS.keys()))
    args = parser.parse_args()
    run(args.env)
