"""
PPO benchmark: StandardPPO vs ContractivePPO.

ContractivePPO uses an identical actor but replaces the critic with
ContractiveCritic — a learned approximation of the Bellman operator T̂.

The critic architecture encodes the contractiveness guarantee directly:
  V*(s) = b(s) / (1 - eigs(s)),  |eigs(s)| < spectral_bound < 1

Environments: CartPole-v1 (discrete), Acrobot-v1 (discrete).
3 seeds each. Reports return + value stability metrics.
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

# Single-threaded is faster on CPU for small tensors (avoids thread-spawn overhead)
torch.set_num_threads(1)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from contractive_rl.ppo.models import (
    ContractiveCritic, StandardCritic,
    CategoricalActor, GaussianActor,
)

try:
    import gymnasium as gym
    def _make_env(env_id, seed):
        env = gym.make(env_id)
        return env, True
except ImportError:
    import gym
    def _make_env(env_id, seed):
        env = gym.make(env_id)
        return env, False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Environment configs
# ---------------------------------------------------------------------------

ENV_CONFIGS = {
    "CartPole-v1": {
        "state_dim": 4, "n_actions": 2, "continuous": False,
        "train_steps": 200_000,
        "rollout_len": 512,
        "n_epochs": 4,
        "lr_actor": 3e-4, "lr_critic": 3e-4,
        "clip_eps": 0.2, "gamma": 0.99, "gae_lambda": 0.95,
        "ent_coef": 0.01, "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "Acrobot-v1": {
        "state_dim": 6, "n_actions": 3, "continuous": False,
        "train_steps": 500_000,
        "rollout_len": 512,
        "n_epochs": 4,
        "lr_actor": 3e-4, "lr_critic": 1e-3,
        "clip_eps": 0.2, "gamma": 0.99, "gae_lambda": 0.95,
        "ent_coef": 0.0, "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "Pendulum-v0": {
        "state_dim": 3, "action_dim": 1, "continuous": True, "action_clip": 2.0,
        "train_steps": 300_000,
        "rollout_len": 2048,
        "n_epochs": 10,
        "lr_actor": 3e-4, "lr_critic": 1e-3,
        "clip_eps": 0.2, "gamma": 0.99, "gae_lambda": 0.95,
        "ent_coef": 0.0, "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
}

SEEDS = [0, 1, 2]
HIDDEN_DIM = 64
SPECTRAL_BOUND = 0.9
MINI_BATCH = 64


# ---------------------------------------------------------------------------
# GAE rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones = [], [], []

    def clear(self):
        self.__init__()

    def add(self, s, a, r, lp, v, d):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.log_probs.append(lp)
        self.values.append(v)
        self.dones.append(d)

    def compute_returns(self, last_value, gamma, gae_lambda):
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else self.values[t + 1]
            next_done = self.dones[t]  # done at step t means s_{t+1} is terminal
            delta = self.rewards[t] + gamma * next_val * (1 - next_done) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def get_tensors(self, advantages, returns):
        s = torch.tensor(np.array(self.states), dtype=torch.float32, device=DEVICE)
        actions_arr = np.array(self.actions)
        if actions_arr.dtype in (np.int32, np.int64):
            a = torch.tensor(actions_arr, dtype=torch.long, device=DEVICE)
        else:
            a = torch.tensor(actions_arr, dtype=torch.float32, device=DEVICE)
        lp = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=DEVICE)
        adv = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        ret = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        return s, a, lp, adv, ret


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_seed(env_id, variant, seed, cfg):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    env, gymnasium_api = _make_env(env_id, seed)
    state_dim = cfg["state_dim"]
    continuous = cfg.get("continuous", False)

    if continuous:
        action_dim = cfg["action_dim"]
        action_clip = cfg.get("action_clip", None)
        actor = GaussianActor(state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
    else:
        n_actions = cfg["n_actions"]
        actor = CategoricalActor(state_dim, n_actions, HIDDEN_DIM).to(DEVICE)

    if variant == "standard":
        critic = StandardCritic(state_dim, HIDDEN_DIM).to(DEVICE)
    else:
        critic = ContractiveCritic(state_dim, HIDDEN_DIM, SPECTRAL_BOUND).to(DEVICE)

    opt_actor  = optim.Adam(actor.parameters(),  lr=cfg["lr_actor"])
    opt_critic = optim.Adam(critic.parameters(), lr=cfg["lr_critic"])

    if gymnasium_api:
        obs, _ = env.reset(seed=seed)
    else:
        obs = env.reset()

    buf = RolloutBuffer()
    episode_returns, ep_return = [], 0.0
    value_log = []
    step = 0

    while step < cfg["train_steps"]:
        # --- Collect rollout ---
        buf.clear()
        for _ in range(cfg["rollout_len"]):
            s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                dist = actor(s_t)
                if continuous:
                    action_t = dist.sample()                         # (1, action_dim)
                    log_prob = dist.log_prob(action_t).sum(-1).item()
                    action = action_t.squeeze(0).numpy()
                    if action_clip is not None:
                        action = np.clip(action, -action_clip, action_clip)
                else:
                    action_t = dist.sample()                         # (1,)
                    log_prob = dist.log_prob(action_t).item()
                    action = action_t.item()
                value = critic(s_t).item()

            value_log.append(abs(value))

            if gymnasium_api:
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = env.step(action)

            buf.add(obs, action, reward, log_prob, value, float(done))
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

        # --- Bootstrap last value ---
        with torch.no_grad():
            last_s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            last_val = critic(last_s).item()

        advantages, returns = buf.compute_returns(last_val, cfg["gamma"], cfg["gae_lambda"])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        s_b, a_b, lp_old, adv_b, ret_b = buf.get_tensors(advantages, returns)

        # --- PPO update epochs ---
        n = len(s_b)
        for _ in range(cfg["n_epochs"]):
            idx = torch.randperm(n)
            for start in range(0, n, MINI_BATCH):
                mb = idx[start:start + MINI_BATCH]
                s_mb, a_mb = s_b[mb], a_b[mb]
                lp_old_mb, adv_mb, ret_mb = lp_old[mb], adv_b[mb], ret_b[mb]

                dist = actor(s_mb)
                if continuous:
                    lp_new = dist.log_prob(a_mb).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()
                else:
                    lp_new = dist.log_prob(a_mb)
                    entropy = dist.entropy().mean()

                ratio = (lp_new - lp_old_mb).exp()
                surr1 = ratio * adv_mb
                surr2 = ratio.clamp(1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean() - cfg["ent_coef"] * entropy

                v_pred = critic(s_mb)
                critic_loss = nn.functional.mse_loss(v_pred, ret_mb)

                opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), cfg["max_grad_norm"])
                opt_actor.step()

                opt_critic.zero_grad()
                (cfg["vf_coef"] * critic_loss).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), cfg["max_grad_norm"])
                opt_critic.step()

    env.close()
    return episode_returns, value_log


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_env(env_id, seeds=SEEDS):
    cfg = ENV_CONFIGS[env_id]
    print(f"\n{'='*55}")
    print(f"  {env_id}  ({cfg['train_steps']//1000}k steps, {len(seeds)} seeds)")
    print(f"{'='*55}")

    results = {}
    for variant in ("standard", "contractive"):
        print(f"  [{variant}]", flush=True)
        all_returns, all_value_logs = [], []
        for seed in seeds:
            print(f"    seed={seed}", flush=True)
            ep_rets, val_log = train_one_seed(env_id, variant, seed, cfg)
            all_returns.append(ep_rets)
            all_value_logs.append(val_log)

        finals = [np.mean(r[-20:]) if len(r) >= 20 else (np.mean(r) if r else 0.)
                  for r in all_returns]
        value_vars = [float(np.var(vl)) for vl in all_value_logs]
        print(f"    final 20-ep mean={np.mean(finals):.1f} ± {np.std(finals):.1f}  "
              f"value_var={np.mean(value_vars):.1f}", flush=True)

        results[variant] = {
            "all_returns": all_returns,
            "value_logs": all_value_logs,
        }

    return results


def _smooth(arr, w=20):
    if len(arr) < w:
        return np.array(arr)
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def plot_and_save(env_id, results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    colors = {"standard": "crimson", "contractive": "steelblue"}
    labels = {"standard": "StandardPPO", "contractive": "ContractivePPO"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for variant, data in results.items():
        seed_returns = data["all_returns"]
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
    axes[0].set_title(f"{env_id} — PPO Variants ({len(SEEDS)} seeds)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Value variance comparison
    var_means = {v: np.mean([np.var(vl) for vl in data["value_logs"]])
                 for v, data in results.items()}
    x = np.arange(len(var_means))
    axes[1].bar(x, list(var_means.values()),
                color=[colors[v] for v in var_means], alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[v] for v in var_means])
    axes[1].set_ylabel("Value Estimate Variance")
    axes[1].set_title("Critic Stability")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"ContractivePPO: Bellman Operator Approximation\n"
                 f"V*(s) = b(s)/(1-eigs(s)),  |eigs| < {SPECTRAL_BOUND}",
                 fontsize=10)
    fig.tight_layout()

    safe_id = env_id.replace("-", "_").lower()
    plot_path = os.path.join(out_dir, f"{safe_id}_ppo_results.png")
    fig.savefig(plot_path, dpi=120); plt.close(fig)
    print(f"  Saved: {plot_path}")

    summary = {}
    for variant, data in results.items():
        finals = [np.mean(r[-20:]) if len(r) >= 20 else (np.mean(r) if r else 0.)
                  for r in data["all_returns"]]
        summary[variant] = {
            "final_mean": float(np.mean(finals)),
            "final_std": float(np.std(finals)),
            "mean_value_variance": float(var_means[variant]),
        }
    json_path = os.path.join(out_dir, f"{safe_id}_ppo_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")
    return summary


def run_ppo(out_dir=RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    all_summaries = {}
    for env_id in ("CartPole-v1", "Acrobot-v1", "Pendulum-v0"):
        results = run_env(env_id, seeds=SEEDS)
        summary = plot_and_save(env_id, results, out_dir)
        all_summaries[env_id] = summary

    print("\n" + "=" * 55)
    print("  PPO BENCHMARK SUMMARY")
    print("=" * 55)
    for env_id, summ in all_summaries.items():
        print(f"\n  {env_id}:")
        for v, label in [("standard", "StandardPPO"), ("contractive", "ContractivePPO")]:
            r = summ.get(v, {})
            print(f"    {label:<20s}: {r.get('final_mean',0):.1f} ± {r.get('final_std',0):.1f}"
                  f"  value_var={r.get('mean_value_variance',0):.1f}")
    print()
    return all_summaries


if __name__ == "__main__":
    run_ppo()
