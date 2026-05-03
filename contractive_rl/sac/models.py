"""
SAC actor-critic models.

Two critic variants — direct analogue of contractive_rl/ppo/models.py:

  StandardCritic:   Q(s, a) = MLP(concat(s, a))                         [vanilla]
  ContractiveCritic: Q(s, a) = h(b(s, a) / (1 - eigs(s, a)))             [ours]

The contractive Q-function is the closed-form fixed point of an affine
contraction on a per-(s,a) latent. The architecture mirrors the PPO
ContractiveCritic, but the input is concat(s, a) (Q-function) rather than
just s (V-function).

The actor is a squashed Gaussian (tanh-Gaussian) shared between variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Critic variants
# ---------------------------------------------------------------------------

class StandardCritic(nn.Module):
    """Vanilla MLP Q(s, a)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class ContractiveCritic(nn.Module):
    """
    Q(s, a) = b(s, a) / (1 - eigs(s, a)),  |eigs| < spectral_bound < 1.

    Closed-form fixed point of T(z; s, a) = eigs(s, a) z + b(s, a). No bijection
    or iteration; the architecture directly encodes contractiveness.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 spectral_bound: float = 0.9):
        super().__init__()
        self.spectral_bound = spectral_bound

        in_dim = state_dim + action_dim
        self.b_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.eigs_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        eigs = self.spectral_bound * torch.tanh(self.eigs_net(sa))
        b = self.b_net(sa)
        denom = (1.0 - eigs).clamp(min=1e-2)
        return (b / denom).squeeze(-1)


# ---------------------------------------------------------------------------
# Actor: squashed Gaussian (tanh-Gaussian)
# ---------------------------------------------------------------------------

class SquashedGaussianActor(nn.Module):
    """
    π(a|s) = tanh(N(μ(s), σ(s))).

    Standard SAC policy; same architecture for both Standard and Contractive
    variants. log-prob accounts for the tanh squash.
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 action_scale: float = 1.0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.action_scale = action_scale

    def forward(self, state: torch.Tensor):
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        """Return (action, log_prob, deterministic_action)."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()                         # reparameterized
        y = torch.tanh(x)
        action = y * self.action_scale
        # Tanh-corrected log-prob: log_pi(a|s) = log N(x|μ,σ) - sum log(1 - tanh(x)^2)
        log_prob = normal.log_prob(x) - torch.log(1.0 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)
        return action, log_prob, torch.tanh(mean) * self.action_scale
