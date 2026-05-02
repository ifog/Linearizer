"""
PPO actor-critic models.

ContractiveCritic: approximates the Bellman operator T̂(v, s) = eigs(s)*v + b(s).
  - T̂ is contractive with rate spectral_bound (< γ by construction)
  - Fixed point v* = b(s)/(1-eigs(s)) gives V*(s) directly in O(1)
  - eigs(s) ≈ effective discount at state s; b(s) ≈ r(s) + γ·E[V*(s')]

StandardCritic: plain MLP baseline.
Actor: shared architecture for both variants (Gaussian for continuous, categorical for discrete).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Critic variants
# ---------------------------------------------------------------------------

class StandardCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class ContractiveCritic(nn.Module):
    """
    Approximates T̂(v, s) = eigs(s) · v + b(s), a contractive affine Bellman operator.

    Fixed point: V*(s) = b(s) / (1 - eigs(s))
    Contraction rate: |eigs(s)| < spectral_bound < 1.

    No bijection or iteration needed — V*(s) is the closed-form output.
    The architecture directly encodes the contractiveness guarantee.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64,
                 spectral_bound: float = 0.99):
        super().__init__()
        self.spectral_bound = spectral_bound

        # b(s): reward + discounted future value component
        self.b_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # eigs(s): effective contraction rate at state s, bounded in (-sb, sb)
        self.eigs_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_operator(self, state: torch.Tensor):
        """Return (eigs, b) for state — the parameters of T̂(·, s)."""
        eigs = self.spectral_bound * torch.tanh(self.eigs_net(state))  # (B, 1)
        b = self.b_net(state)                                           # (B, 1)
        return eigs, b

    def bellman_step(self, v: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Apply one step of the learned Bellman operator: T̂(v, s) = eigs*v + b."""
        eigs, b = self.get_operator(state)
        return (eigs * v.unsqueeze(-1) + b).squeeze(-1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        V*(s) = b(s) / (1 - eigs(s))  — fixed point of T̂(·, s).

        This is V* directly, not an approximation that requires iteration.
        """
        eigs, b = self.get_operator(state)
        # Clamp denominator away from zero to prevent gradient spikes
        denom = (1.0 - eigs).clamp(min=1e-2)
        return (b / denom).squeeze(-1)


# ---------------------------------------------------------------------------
# Actor (shared for standard and contractive PPO)
# ---------------------------------------------------------------------------

class GaussianActor(nn.Module):
    """Continuous action space actor (MuJoCo-style)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor):
        mean = self.net(state)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward(state).log_prob(action).sum(-1)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state).entropy().sum(-1)


class CategoricalActor(nn.Module):
    """Discrete action space actor (CartPole/Acrobot-style)."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor):
        return torch.distributions.Categorical(logits=self.net(state))

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward(state).log_prob(action)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state).entropy()
