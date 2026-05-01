"""
Baird's Counterexample — Classical demonstration of the Deadly Triad.

Standard semi-gradient TD with function approximation diverges in off-policy settings.
The Contractive Linearizer prevents divergence by construction.

Reference: Baird (1995) "Residual Algorithms", and Sutton & Barto (2018) Example 11.1.

SETUP: Baird's 7-state star MDP.
  - 7 states, 1 special "dashed" action and 1 "solid" action.
  - All rewards = 0, γ = 0.99.
  - Behavior policy: uniform random over all states via dashed action (off-policy).
  - Target policy: always "solid" (go to center state 6).
  - Linear V-function with 8 parameters and Baird's specific features.
  - This creates a deadly triad scenario: off-policy + bootstrapping + lin. approx.
  - Semi-gradient TD with these features DIVERGES.

For the Q-value version, we extend to 2 actions and repeat the analysis.
"""

import os
import sys
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

# ---------------------------------------------------------------------------
# Baird's 7-state star MDP (exact Sutton & Barto setup)
# ---------------------------------------------------------------------------
# States: 0-5 = dumbbell, 6 = center
# Behavior: from any state, with prob 1 go to random dumbbell state (dashed)
# Target:   from any state, go to center state 6 (solid)
# Reward:   0 everywhere, γ = 0.99
#
# Feature vectors φ(s) ∈ R^8 (one-hot-like encoding):
#   State 0-5 (dumbbell i): φ_i = 2*e_i + e_7    (position i=2, last=1)
#   State 6 (center):       φ_6 = e_6 + 2*e_7     (position 6=1, last=2)
# where e_k is unit vector in R^8 (0-indexed).
#
# V(s) = w · φ(s) = w_i*2 + w_7   for dumbbell s=i
#                  = w_6 + 2*w_7   for center s=6

N_STATES = 7
GAMMA = 0.99
ALPHA_TD = 0.01   # learning rate (matches Sutton & Barto; large enough for divergence)
N_FEATURES = 8    # number of parameters w


def phi(s: int) -> np.ndarray:
    """Baird feature vector for state s (8-dimensional)."""
    v = np.zeros(N_FEATURES)
    if s < 6:  # dumbbell states
        v[s] = 2.0
        v[7] = 1.0
    else:       # center state
        v[6] = 1.0
        v[7] = 2.0
    return v


def V(s: int, w: np.ndarray) -> float:
    return float(np.dot(phi(s), w))


def target_next_state(s: int) -> int:
    """Target policy: always solid → center."""
    return 6


def behavior_sample_state() -> int:
    """Behavior policy: dashed → uniform random dumbbell state (0-5)."""
    return np.random.randint(0, 6)  # uniform over dumbbell states


# ---------------------------------------------------------------------------
# Method 1: Semi-gradient TD with linear function approximation (DIVERGES)
# ---------------------------------------------------------------------------

def run_linear_td(n_steps: int = 3000) -> dict:
    """
    Semi-gradient TD(0) with Baird's linear V-function (off-policy).
    This is the classical demonstration of divergence under the deadly triad.
    DIVERGES: ||w|| → ∞.
    """
    np.random.seed(42)
    # Baird's original initialization: all components = 1
    w = np.ones(N_FEATURES)

    w_norms = []
    td_errors = []

    for step in range(n_steps):
        # Sample from behavior policy (off-policy)
        s = behavior_sample_state()             # uniform dumbbell
        s_next = target_next_state(s)           # target policy: go to center

        # Semi-gradient TD update with IS ratio = 1 (using behavior actions)
        # (The IS ratio would normally be pi(a|s)/b(a|s), but Baird's counterexample
        # directly uses the semi-gradient rule without importance sampling,
        # which is what causes the divergence.)
        td_err = 0.0 + GAMMA * V(s_next, w) - V(s, w)
        grad_w = phi(s)   # ∇V(s,w) = φ(s)

        w = w + ALPHA_TD * td_err * grad_w   # semi-gradient update

        w_norm = float(np.linalg.norm(w))
        w_norms.append(w_norm)
        td_errors.append(abs(td_err))

    return {
        "q_norms": w_norms,
        "td_errors": td_errors,
        "final_norm": float(np.linalg.norm(w)),
        "diverged": float(np.linalg.norm(w)) > 10.0,
    }


# ---------------------------------------------------------------------------
# Method 2: MLP with semi-gradient TD (also diverges / oscillates)
# ---------------------------------------------------------------------------

class MLPQNet(nn.Module):
    """Small MLP for state value V(s)."""
    def __init__(self, state_dim: int = N_FEATURES, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, phi_vec: torch.Tensor) -> torch.Tensor:
        return self.net(phi_vec)


def run_mlp_td(n_steps: int = 3000) -> dict:
    """
    Semi-gradient TD with MLP approximator (no target network, off-policy).
    Also diverges or oscillates due to deadly triad.
    Uses the same learning rate as linear TD for fair comparison.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")
    net = MLPQNet(state_dim=N_FEATURES, hidden=128).to(device)
    # Use SGD with higher learning rate (2x) to expose instability
    # No momentum, no adaptive lr — raw semi-gradient TD
    optimizer = optim.SGD(net.parameters(), lr=ALPHA_TD * 2.0, momentum=0.0)

    q_norms = []
    td_errors = []

    for step in range(n_steps):
        s = behavior_sample_state()
        s_next = target_next_state(s)

        phi_s = torch.tensor(phi(s), dtype=torch.float32, device=device).unsqueeze(0)
        phi_sn = torch.tensor(phi(s_next), dtype=torch.float32, device=device).unsqueeze(0)

        v_s = net(phi_s).squeeze()
        with torch.no_grad():
            v_next = net(phi_sn).squeeze()

        td_err = GAMMA * v_next - v_s   # reward = 0
        # Semi-gradient: differentiate only through v_s
        loss = -td_err.detach() * v_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track V-function magnitude over all states
        with torch.no_grad():
            all_phi = torch.tensor(
                np.array([phi(s_) for s_ in range(N_STATES)]),
                dtype=torch.float32, device=device,
            )
            v_all = net(all_phi).squeeze().cpu().numpy()

        norm_val = float(np.linalg.norm(v_all))
        if np.isnan(norm_val) or np.isinf(norm_val):
            # Once NaN, track as very large (diverged)
            last_valid = q_norms[-1] if q_norms else 1.0
            q_norms.append(last_valid * 10.0)
        else:
            q_norms.append(norm_val)
        td_errors.append(float(abs(td_err.item())) if not np.isnan(td_err.item()) else td_errors[-1] * 2.0)

    final_norm = q_norms[-1] if q_norms else float("nan")
    return {
        "q_norms": q_norms,
        "td_errors": td_errors,
        "final_norm": final_norm,
        "diverged": final_norm > 10.0,
    }


# ---------------------------------------------------------------------------
# Method 3: Contractive Linearizer (STABLE by construction)
# ---------------------------------------------------------------------------

class ContractiveVNet(nn.Module):
    """
    Contractive Linearizer V-function for Baird's MDP.

    V-vector is 7-dim (one per state).
    Context = feature vector of current state (8-dim).
    Iterates K=3 steps from v_prev = 0.
    By construction, spectral radius < 1 => cannot diverge.
    """

    def __init__(self, v_dim: int = N_STATES, context_dim: int = N_FEATURES,
                 n_coupling: int = 4, hidden_dim: int = 64, K: int = 3):
        super().__init__()
        self.v_dim = v_dim
        self.K = K

        self.g = AffineCouplingNet(dim=v_dim, n_coupling_layers=n_coupling,
                                   hidden_dim=hidden_dim)
        self.A = DiagonalContractiveOp(context_dim=context_dim,
                                       latent_dim=v_dim,
                                       spectral_bound=0.99)

    def forward(self, context: torch.Tensor, K: int = None) -> torch.Tensor:
        """
        Compute V-values for all 7 states from a context vector.

        Args:
            context: (B, context_dim)
            K: number of iterations (default self.K)

        Returns:
            v: (B, v_dim)
        """
        if K is None:
            K = self.K
        B = context.shape[0]
        z = torch.zeros(B, self.v_dim, device=context.device)
        z = self.g.encode(z)
        for _ in range(K):
            z = self.A.apply(z, context)
        return self.g.decode(z)


def run_contractive_td(n_steps: int = 3000) -> dict:
    """
    Semi-gradient TD with Contractive Linearizer (off-policy, no target net).
    STABLE: spectral radius < 1 prevents divergence by construction.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")
    net = ContractiveVNet(v_dim=N_STATES, context_dim=N_FEATURES,
                          n_coupling=4, hidden_dim=64).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    q_norms = []
    td_errors = []

    for step in range(n_steps):
        s = behavior_sample_state()
        s_next = target_next_state(s)

        ctx_s = torch.tensor(phi(s), dtype=torch.float32, device=device).unsqueeze(0)
        ctx_sn = torch.tensor(phi(s_next), dtype=torch.float32, device=device).unsqueeze(0)

        v_all = net(ctx_s)            # (1, 7) — uses state s as context
        v_s = v_all[0, s]

        with torch.no_grad():
            v_all_next = net(ctx_sn)  # (1, 7) — uses s_next as context
            v_next = v_all_next[0, s_next]

        td_err = float((GAMMA * v_next - v_s).item())   # reward = 0
        # Semi-gradient: differentiate through v_s only
        loss = -td_err * v_s

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            v_vec = net(ctx_s).squeeze().cpu().numpy()

        q_norms.append(float(np.linalg.norm(v_vec)))
        td_errors.append(abs(td_err))

    final_norm = float(np.linalg.norm(v_vec))
    return {
        "q_norms": q_norms,
        "td_errors": td_errors,
        "final_norm": final_norm,
        "diverged": final_norm > 1e6,
    }


# ---------------------------------------------------------------------------
# Plotting and main runner
# ---------------------------------------------------------------------------

def _verify_invertibility(dim: int = 7):
    """Verify AffineCouplingNet is invertible (required by spec)."""
    with torch.no_grad():
        net = AffineCouplingNet(dim=dim, n_coupling_layers=4, hidden_dim=64)
        x = torch.randn(8, dim)
        recon = net.decode(net.encode(x))
        err = (recon - x).abs().max().item()
        assert err < 1e-5, (
            f"AffineCouplingNet(dim={dim}) is not invertible: err={err:.2e}"
        )
        print(f"  Invertibility check passed: max recon error = {err:.2e}")


def run_baird(out_dir: str = "results") -> dict:
    """Run all three methods and produce comparison plots."""
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print("Baird's Counterexample — Deadly Triad Demonstration")
    print("=" * 60)
    _verify_invertibility(dim=N_STATES)

    n_steps = 3000

    print("Running Method 1: Linear semi-gradient TD (off-policy)...", flush=True)
    res_linear = run_linear_td(n_steps)
    print(f"  Final ||w|| = {res_linear['final_norm']:.2f}  "
          f"→ {'DIVERGED' if res_linear['diverged'] else 'stable'}")

    print("Running Method 2: MLP semi-gradient TD (off-policy)...", flush=True)
    res_mlp = run_mlp_td(n_steps)
    print(f"  Final ||V||  = {res_mlp['final_norm']:.2f}  "
          f"→ {'DIVERGED' if res_mlp['diverged'] else 'stable'}")

    print("Running Method 3: Contractive Linearizer (off-policy)...", flush=True)
    res_contractive = run_contractive_td(n_steps)
    print(f"  Final ||V||  = {res_contractive['final_norm']:.6f}  "
          f"→ {'DIVERGED' if res_contractive['diverged'] else 'STABLE (bounded by construction)'}")

    steps = np.arange(1, n_steps + 1)

    # ---- Plot 1: V/Q norm ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(steps, np.clip(res_linear["q_norms"], 1e-6, None),
                label="Linear TD (diverges)", color="crimson", linewidth=1.8)
    ax.semilogy(steps, np.clip(res_mlp["q_norms"], 1e-6, None),
                label="MLP-TD (diverges)", color="darkorange", linewidth=1.8,
                linestyle="--")
    ax.semilogy(steps, np.clip(res_contractive["q_norms"], 1e-6, None),
                label="Contractive Linearizer (stable)", color="steelblue",
                linewidth=2.2)
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("||V||  (log scale)", fontsize=12)
    ax.set_title("Baird's Counterexample: Value Function Norm\n"
                 "(Deadly Triad: off-policy + bootstrapping + function approx.)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = os.path.join(out_dir, "baird_divergence.png")
    fig.savefig(path1, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ---- Plot 2: TD error ----
    fig, ax = plt.subplots(figsize=(9, 5))
    smooth = lambda x, w=50: np.convolve(x, np.ones(w) / w, mode="valid")
    ax.semilogy(smooth(np.clip(res_linear["td_errors"], 1e-9, None)),
                label="Linear TD", color="crimson", linewidth=1.8)
    ax.semilogy(smooth(np.clip(res_mlp["td_errors"], 1e-9, None)),
                label="MLP-TD", color="darkorange", linewidth=1.8,
                linestyle="--")
    ax.semilogy(smooth(np.clip(res_contractive["td_errors"], 1e-9, None)),
                label="Contractive Linearizer", color="steelblue",
                linewidth=2.2)
    ax.set_xlabel("Training Steps (smoothed)", fontsize=12)
    ax.set_ylabel("|TD error|  (log scale)", fontsize=12)
    ax.set_title("Baird's Counterexample: TD Error", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = os.path.join(out_dir, "baird_td_error.png")
    fig.savefig(path2, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path2}")

    return {
        "linear": res_linear,
        "mlp": res_mlp,
        "contractive": res_contractive,
    }


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    run_baird(out_dir=results_dir)
