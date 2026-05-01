"""
CartPole-v1 Benchmark: StandardDQN vs DoubleDQN vs ContractiveDQN.

Demonstrates competitive performance of the Contractive Linearizer
on a standard RL benchmark.
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

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from contractive_rl.shared.invertible_net import AffineCouplingNet
from contractive_rl.shared.contractive_operator import DiagonalContractiveOp

try:
    import gymnasium as gym
    GYM_RESET_KWARGS = True  # gymnasium requires seed kwarg in reset
except ImportError:
    import gym
    GYM_RESET_KWARGS = False

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BUFFER_SIZE = 10_000
BATCH_SIZE = 64
GAMMA = 0.99
TARGET_UPDATE_FREQ = 100       # steps between target network syncs
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_STEPS = 5_000
TRAIN_STEPS = 50_000
LEARNING_RATE = 1e-3
SEEDS = list(range(5))          # seeds 0–4
MIN_REPLAY_SIZE = 200           # wait before training

# Force CPU: CUDA is detected but may not be functional on this machine
# (torch.cuda.is_available() can return True even when CUDA ops cause bus errors)
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
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
# Q-Networks
# ---------------------------------------------------------------------------

class StandardQNet(nn.Module):
    """Linear(4) → 128 → 64 → 2"""

    def __init__(self, state_dim: int = 4, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContractiveQNet(nn.Module):
    """
    Contractive Linearizer Q-network for CartPole.

    Q-vector is 2-dim (one per action).
    context = state (4-dim).
    Iterates K=3 steps from q_prev=0.
    """

    def __init__(self, state_dim: int = 4, n_actions: int = 2,
                 n_coupling: int = 4, hidden_dim: int = 32, K: int = 3):
        super().__init__()
        self.n_actions = n_actions
        self.K = K

        # g: AffineCouplingNet on 2-dim Q-vector
        self.g = AffineCouplingNet(dim=n_actions,
                                   n_coupling_layers=n_coupling,
                                   hidden_dim=hidden_dim)
        # A: state-conditioned diagonal contractive operator
        self.A = DiagonalContractiveOp(context_dim=state_dim,
                                       latent_dim=n_actions,
                                       spectral_bound=0.99)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all actions given state.

        Args:
            state: (B, state_dim)

        Returns:
            q: (B, n_actions)
        """
        B = state.shape[0]
        q_prev = torch.zeros(B, self.n_actions, device=state.device)
        z = self.g.encode(q_prev)
        for _ in range(self.K):
            z = self.A.apply(z, state)
        return self.g.decode(z)


# ---------------------------------------------------------------------------
# Epsilon-greedy policy
# ---------------------------------------------------------------------------

def epsilon(step: int) -> float:
    frac = min(step / EPS_DECAY_STEPS, 1.0)
    return EPS_START + (EPS_END - EPS_START) * frac


def select_action(q_net: nn.Module, state: np.ndarray, step: int,
                  n_actions: int = 2) -> int:
    if random.random() < epsilon(step):
        return random.randrange(n_actions)
    s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return int(q_net(s).argmax(dim=1).item())


# ---------------------------------------------------------------------------
# DQN Training Loops
# ---------------------------------------------------------------------------

def _make_env(seed: int):
    if GYM_RESET_KWARGS:
        env = gym.make("CartPole-v1")
    else:
        env = gym.make("CartPole-v1")
    return env


def train_dqn(q_net: nn.Module, target_net: nn.Module,
              seed: int, variant: str = "standard",
              n_steps: int = TRAIN_STEPS) -> dict:
    """
    Generic DQN training loop.

    Args:
        q_net:       online Q-network
        target_net:  target Q-network (copy of q_net, updated periodically)
        seed:        random seed
        variant:     "standard", "double", or "contractive"
        n_steps:     total environment steps

    Returns:
        dict with episode_returns, q_value_means, q_value_stds
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
    buf = ReplayBuffer(BUFFER_SIZE)

    episode_returns = []
    q_value_means = []
    q_value_stds = []
    ep_return = 0.0
    step = 0

    while step < n_steps:
        # Select action
        action = select_action(q_net, obs, step)

        # Step environment
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

        # Train
        if len(buf) >= MIN_REPLAY_SIZE:
            s_b, a_b, r_b, sn_b, d_b = buf.sample(BATCH_SIZE)
            with torch.no_grad():
                if variant == "double":
                    # Double DQN: select action with online net, evaluate with target
                    next_actions = q_net(sn_b).argmax(dim=1)
                    q_next = target_net(sn_b).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    q_next = target_net(sn_b).max(dim=1).values
                targets = r_b + GAMMA * q_next * (1.0 - d_b)

            q_pred = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(q_pred, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

            # Track Q-value statistics
            if step % 500 == 0:
                with torch.no_grad():
                    q_vals = q_net(s_b)
                    q_value_means.append(q_vals.mean().item())
                    q_value_stds.append(q_vals.std().item())

        # Update target network
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())

    env.close()
    return {
        "episode_returns": episode_returns,
        "q_value_means": q_value_means,
        "q_value_stds": q_value_stds,
    }


def run_variant(variant: str, seeds: list = SEEDS,
                n_steps: int = TRAIN_STEPS) -> dict:
    """Run a DQN variant across multiple seeds."""
    print(f"  Training {variant} ({len(seeds)} seeds × {n_steps} steps)...",
          flush=True)
    all_returns = []
    all_q_means = []
    all_q_stds = []

    for seed in seeds:
        print(f"    seed={seed}", flush=True)
        if variant in ("standard", "double"):
            q_net = StandardQNet().to(DEVICE)
            target_net = StandardQNet().to(DEVICE)
        else:  # contractive
            q_net = ContractiveQNet().to(DEVICE)
            target_net = ContractiveQNet().to(DEVICE)

        target_net.load_state_dict(q_net.state_dict())

        res = train_dqn(q_net, target_net, seed=seed,
                        variant=variant, n_steps=n_steps)
        all_returns.append(res["episode_returns"])
        all_q_means.append(res["q_value_means"])
        all_q_stds.append(res["q_value_stds"])

    # Align episode lengths by interpolating to fixed grid
    max_eps = max(len(r) for r in all_returns)
    interp_returns = []
    for r in all_returns:
        if len(r) == 0:
            interp_returns.append(np.zeros(max_eps))
        else:
            x_old = np.linspace(0, 1, len(r))
            x_new = np.linspace(0, 1, max_eps)
            interp_returns.append(np.interp(x_new, x_old, r))

    arr = np.array(interp_returns)
    final_10_returns = [np.mean(r[-10:]) if len(r) >= 10 else np.mean(r)
                        for r in all_returns]

    return {
        "returns_arr": arr,           # (n_seeds, max_eps)
        "q_value_means": all_q_means,
        "q_value_stds": all_q_stds,
        "final_mean": float(np.mean(final_10_returns)),
        "final_std": float(np.std(final_10_returns)),
        "q_var": float(np.mean([np.var(m) for m in all_q_means if len(m) > 0])),
    }


def _smooth(arr: np.ndarray, w: int = 10) -> np.ndarray:
    """Moving average smoothing."""
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def _verify_invertibility_cartpole():
    """Verify AffineCouplingNet is invertible for CartPole's 2-dim Q-vector."""
    with torch.no_grad():
        net = AffineCouplingNet(dim=2, n_coupling_layers=4, hidden_dim=32)
        x = torch.randn(16, 2)
        recon = net.decode(net.encode(x))
        err = (recon - x).abs().max().item()
        assert err < 1e-5, (
            f"AffineCouplingNet(dim=2) is not invertible: err={err:.2e}"
        )
        print(f"  Invertibility check (dim=2) passed: max error = {err:.2e}")


def run_cartpole(out_dir: str = "results", n_steps: int = TRAIN_STEPS) -> dict:
    """Run CartPole benchmark for all three variants."""
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print(f"CartPole-v1 Benchmark ({n_steps} steps, {len(SEEDS)} seeds)")
    print("=" * 60)
    _verify_invertibility_cartpole()

    results = {}
    for variant in ("standard", "double", "contractive"):
        results[variant] = run_variant(variant, seeds=SEEDS, n_steps=n_steps)

    # ---- Learning curves plot ----
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"standard": "crimson", "double": "darkorange", "contractive": "steelblue"}
    labels = {"standard": "StandardDQN", "double": "DoubleDQN",
              "contractive": "ContractiveDQN"}

    for v in ("standard", "double", "contractive"):
        arr = results[v]["returns_arr"]  # (seeds, episodes)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        n_ep = len(mean)
        xs = np.arange(n_ep)
        sm_mean = _smooth(mean)
        sm_std = _smooth(std)
        xs_sm = np.arange(len(sm_mean))
        ax.plot(xs_sm, sm_mean, label=labels[v], color=colors[v], linewidth=2)
        ax.fill_between(xs_sm,
                        np.clip(sm_mean - sm_std, 0, None),
                        sm_mean + sm_std,
                        alpha=0.2, color=colors[v])

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Return (smoothed)", fontsize=12)
    ax.set_title(f"CartPole-v1 Learning Curves ({n_steps} steps, {len(SEEDS)} seeds)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path1 = os.path.join(out_dir, "cartpole_learning_curves.png")
    fig.savefig(path1, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ---- Q-value stability plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for v in ("standard", "contractive"):
        q_means_all = results[v]["q_value_means"]
        q_stds_all = results[v]["q_value_stds"]
        all_means_flat = [m for seed_means in q_means_all for m in seed_means]
        all_stds_flat = [s for seed_stds in q_stds_all for s in seed_stds]
        if all_means_flat:
            xs = np.arange(len(all_means_flat))
            smooth_m = _smooth(np.array(all_means_flat), w=10)
            smooth_s = _smooth(np.array(all_stds_flat), w=10)
            xs_sm = np.arange(len(smooth_m))
            axes[0].plot(xs_sm, smooth_m, label=labels[v], color=colors[v])
            axes[1].plot(xs_sm, smooth_s, label=labels[v], color=colors[v])

    axes[0].set_title("Q-value Mean over Training")
    axes[0].set_xlabel("Checkpoint")
    axes[0].set_ylabel("Mean Q-value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Q-value Std over Training")
    axes[1].set_xlabel("Checkpoint")
    axes[1].set_ylabel("Std Q-value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Q-Value Stability: Standard vs Contractive DQN", fontsize=12)
    fig.tight_layout()
    path2 = os.path.join(out_dir, "cartpole_qvalue_stability.png")
    fig.savefig(path2, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path2}")

    # ---- Save JSON results ----
    summary = {}
    for v in results:
        r = results[v]
        summary[v] = {
            "final_mean_return": r["final_mean"],
            "final_std_return": r["final_std"],
            "q_value_variance": r["q_var"],
        }

    json_path = os.path.join(out_dir, "cartpole_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    # Print summary
    print("\nCartPole Results (final 10-ep mean ± std):")
    for v, label in labels.items():
        r = results[v]
        print(f"  {label:20s}: {r['final_mean']:.1f} ± {r['final_std']:.1f}")

    return results


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    run_cartpole(out_dir=results_dir)
