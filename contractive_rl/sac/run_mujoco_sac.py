"""
SAC on MuJoCo continuous-control envs: StandardSAC vs ContractiveSAC.

The only difference between variants is the Q-critic architecture:
  - StandardSAC:   StandardCritic (MLP)
  - ContractiveSAC: ContractiveCritic (closed-form fixed-point parameterization)

Everything else (actor, replay buffer, automatic temperature, soft target
update) is identical.

Usage:
  PYTHONPATH=/home/nvidia/.local/lib/python3.8/site-packages \
    /home/nvidia/anaconda3/envs/rlgpu/bin/python contractive_rl/sac/run_mujoco_sac.py \
    --env Hopper-v4
"""

import os
import sys
import json
import random
import time
import argparse
import collections

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from contractive_rl.sac.models import (
    StandardCritic, ContractiveCritic, SquashedGaussianActor,
)

import gymnasium as gym

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

ENV_CFGS = {
    "Hopper-v4":      {"state_dim": 11, "action_dim": 3},
    "Walker2d-v4":    {"state_dim": 17, "action_dim": 6},
    "HalfCheetah-v4": {"state_dim": 17, "action_dim": 6},
    "Ant-v4":         {"state_dim": 27, "action_dim": 8},
}

CFG = {
    "train_steps":       1_000_000,
    "warmup_steps":      10_000,    # random actions until this many env steps
    "buffer_size":       1_000_000,
    "batch_size":        256,
    "lr":                3e-4,
    "gamma":             0.99,
    "tau":               0.005,
    "init_temperature":  0.2,
    "update_every":      1,
    "log_every":         100_000,
    "eval_value_every":  5_000,     # log critic mean(Q) for variance metric
}
HIDDEN_DIM = 256
SPECTRAL_BOUND = 0.9
SEEDS = [0, 1, 2]


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.s  = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r  = np.zeros(capacity, dtype=np.float32)
        self.sn = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d  = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.pos, self.size = capacity, 0, 0

    def push(self, s, a, r, sn, d):
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.sn[i], self.d[i] = s, a, r, sn, d
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.from_numpy(self.s[idx]).to(DEVICE),
            torch.from_numpy(self.a[idx]).to(DEVICE),
            torch.from_numpy(self.r[idx]).to(DEVICE),
            torch.from_numpy(self.sn[idx]).to(DEVICE),
            torch.from_numpy(self.d[idx]).to(DEVICE),
        )

    def __len__(self):
        return self.size


def make_critics(state_dim, action_dim, variant):
    if variant == "standard":
        Q1 = StandardCritic(state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
        Q2 = StandardCritic(state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
    elif variant == "contractive":
        Q1 = ContractiveCritic(state_dim, action_dim, HIDDEN_DIM, SPECTRAL_BOUND).to(DEVICE)
        Q2 = ContractiveCritic(state_dim, action_dim, HIDDEN_DIM, SPECTRAL_BOUND).to(DEVICE)
    else:
        raise ValueError(variant)
    return Q1, Q2


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def train_one_seed(env_id, variant, seed, cfg):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    env = gym.make(env_id)
    state_dim = ENV_CFGS[env_id]["state_dim"]
    action_dim = ENV_CFGS[env_id]["action_dim"]
    action_scale = float(env.action_space.high[0])

    actor = SquashedGaussianActor(state_dim, action_dim, HIDDEN_DIM, action_scale).to(DEVICE)
    Q1, Q2 = make_critics(state_dim, action_dim, variant)
    Q1_target, Q2_target = make_critics(state_dim, action_dim, variant)
    Q1_target.load_state_dict(Q1.state_dict())
    Q2_target.load_state_dict(Q2.state_dict())
    for p in Q1_target.parameters(): p.requires_grad = False
    for p in Q2_target.parameters(): p.requires_grad = False

    actor_opt = optim.Adam(actor.parameters(), lr=cfg["lr"])
    Q1_opt    = optim.Adam(Q1.parameters(),    lr=cfg["lr"])
    Q2_opt    = optim.Adam(Q2.parameters(),    lr=cfg["lr"])

    log_alpha = torch.tensor(np.log(cfg["init_temperature"]), device=DEVICE, requires_grad=True)
    alpha_opt = optim.Adam([log_alpha], lr=cfg["lr"])
    target_entropy = -float(action_dim)

    buf = ReplayBuffer(cfg["buffer_size"], state_dim, action_dim)
    obs, _ = env.reset(seed=seed)

    episode_returns, ep_return = [], 0.0
    value_log = []
    t0 = time.time()
    step = 0
    while step < cfg["train_steps"]:
        if step < cfg["warmup_steps"]:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s_t = torch.from_numpy(obs).to(DEVICE).float().unsqueeze(0)
                a_t, _, _ = actor.sample(s_t)
            action = a_t.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # SAC convention: only "true" termination contributes to bootstrap; we
        # treat truncation as non-terminal for the value target.
        bootstrap_done = float(terminated)
        buf.push(obs, action, reward, next_obs, bootstrap_done)
        ep_return += reward
        obs = next_obs
        step += 1

        if done:
            episode_returns.append(ep_return)
            ep_return = 0.0
            obs, _ = env.reset()

        if len(buf) >= max(cfg["batch_size"], cfg["warmup_steps"]) and \
                step % cfg["update_every"] == 0:
            s_b, a_b, r_b, sn_b, d_b = buf.sample(cfg["batch_size"])

            with torch.no_grad():
                an_b, log_pi_n, _ = actor.sample(sn_b)
                q_next = torch.min(Q1_target(sn_b, an_b), Q2_target(sn_b, an_b))
                alpha = log_alpha.exp()
                target = r_b + cfg["gamma"] * (1.0 - d_b) * (q_next - alpha * log_pi_n)

            q1 = Q1(s_b, a_b); q2 = Q2(s_b, a_b)
            Q1_loss = F.mse_loss(q1, target)
            Q2_loss = F.mse_loss(q2, target)

            Q1_opt.zero_grad(); Q1_loss.backward(); Q1_opt.step()
            Q2_opt.zero_grad(); Q2_loss.backward(); Q2_opt.step()

            new_a, log_pi, _ = actor.sample(s_b)
            q_min_new = torch.min(Q1(s_b, new_a), Q2(s_b, new_a))
            actor_loss = (log_alpha.exp().detach() * log_pi - q_min_new).mean()
            actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

            alpha_loss = -(log_alpha * (log_pi.detach() + target_entropy)).mean()
            alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

            soft_update(Q1_target, Q1, cfg["tau"])
            soft_update(Q2_target, Q2, cfg["tau"])

            if step % cfg["eval_value_every"] == 0:
                value_log.append(float(q1.mean().item()))

        if step % cfg["log_every"] == 0:
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
    cfg = CFG.copy()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = env_id.lower().replace("-", "_")
    print(f"Device: {DEVICE}  |  {env_id}  |  {cfg['train_steps']//1000}k steps  |  hidden={HIDDEN_DIM}")
    print("=" * 60)

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
            "final_std":  float(np.std(finals)),
            "mean_value_variance": float(np.mean(value_vars)),
        }

    json_path = os.path.join(RESULTS_DIR, f"{tag}_sac_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")

    colors = {"standard": "crimson", "contractive": "steelblue"}
    labels = {"standard": "StandardSAC", "contractive": "ContractiveSAC"}
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for variant, data in all_results.items():
        seed_returns = data["returns"]
        max_ep = max((len(r) for r in seed_returns), default=0)
        if max_ep == 0:
            continue
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
    ax.set_title(f"{env_id} — SAC ({len(SEEDS)} seeds)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"{tag}_sac_results.png")
    fig.savefig(plot_path, dpi=120)
    print(f"Saved: {plot_path}")

    print(f"\n{'='*60}\n{env_id.upper()} SAC RESULTS\n{'='*60}")
    std_mean = summary["standard"]["final_mean"]
    for variant in ("standard", "contractive"):
        s = summary[variant]
        lbl = labels[variant]
        if std_mean != 0 and variant == "contractive":
            pct = 100.0 * (s["final_mean"] - std_mean) / abs(std_mean)
            sign = "+" if pct >= 0 else ""
            print(f"  {lbl:<20s}: {s['final_mean']:.1f} ± {s['final_std']:.1f}  "
                  f"value_var={s['mean_value_variance']:.0f}  ({sign}{pct:.1f}%)")
        else:
            print(f"  {lbl:<20s}: {s['final_mean']:.1f} ± {s['final_std']:.1f}  "
                  f"value_var={s['mean_value_variance']:.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Hopper-v4",
                        choices=list(ENV_CFGS.keys()))
    args = parser.parse_args()
    run(args.env)


if __name__ == "__main__":
    main()
