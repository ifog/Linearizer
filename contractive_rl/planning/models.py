"""
Models for the Contractive Linearizer planning experiment.

Four models:
1. ContractiveLinearizer  — proposed method with truly invertible g (AffineCouplingNet)
2. VINBaseline            — proper Value Iteration Network (Tamar et al. 2016)
3. UnconstrainedLinearizer — ablation: same arch but A(c) unconstrained
4. IterativeMLPBaseline   — residual MLP, no contraction guarantee

Grid: 10x10 = 100 states, 20x20 = 400 states
Context: 128-dim CNN encoding of obstacle/goal map
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow importing from parent package
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from shared.invertible_net import AffineCouplingNet
from shared.contractive_operator import DiagonalContractiveOp

GRID_SIZE = 10
STATE_DIM = GRID_SIZE * GRID_SIZE   # 100
MAP_CHANNELS = 3
CONTEXT_DIM = 128


# ---------------------------------------------------------------------------
# Shared Map Encoder (CNN)
# ---------------------------------------------------------------------------

class MapEncoder(nn.Module):
    """
    Encode a (3, H, W) obstacle/goal map into a 128-dim context vector.
    Works for any grid size (adaptive average pooling).
    """

    def __init__(self, map_channels: int = MAP_CHANNELS, out_dim: int = CONTEXT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(map_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Adaptive pooling to a fixed 4x4 spatial size → 32*4*4 = 512
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, out_dim),
            nn.ReLU(),
        )

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            maps: (B, 3, H, W)
        Returns:
            ctx: (B, out_dim)
        """
        x = self.conv(maps)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 1. ContractiveLinearizer — THE PROPOSED METHOD
# ---------------------------------------------------------------------------

class ContractiveLinearizer(nn.Module):
    """
    V_{k+1} = g^{-1}(A(c) * g(V_k))

    where:
      g = AffineCouplingNet(dim=state_dim, n_coupling_layers=6)  [truly invertible]
      A(c) = DiagonalContractiveOp(context_dim=128, spectral_bound<1)
      c = MapEncoder(map)  [CNN context]

    Fast iteration: iterate_fast(V0, c, K) uses A^K in O(1) instead of O(K) passes.
    Since g is truly invertible, the fast and loop methods give identical results
    (up to floating-point precision).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        map_channels: int = MAP_CHANNELS,
        spectral_bound: float = 0.99,
        n_coupling_layers: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.spectral_bound = spectral_bound

        self.map_encoder = MapEncoder(map_channels, out_dim=CONTEXT_DIM)
        self.g = AffineCouplingNet(dim=state_dim, n_coupling_layers=n_coupling_layers,
                                   hidden_dim=hidden_dim)
        self.A = DiagonalContractiveOp(context_dim=CONTEXT_DIM, latent_dim=state_dim,
                                       spectral_bound=spectral_bound)

    def forward(self, V: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        """
        One application of T: V_{k+1} = g^{-1}(A(c) * g(V_k)).

        Args:
            V:    (B, state_dim)
            maps: (B, 3, H, W)

        Returns:
            V_next: (B, state_dim)
        """
        c = self.map_encoder(maps)
        z = self.g.encode(V)
        z_next = self.A.apply(z, c)
        return self.g.decode(z_next)

    def iterate_loop(self, V0: torch.Tensor, maps: torch.Tensor, K: int) -> torch.Tensor:
        """Apply T K times using a loop (O(K) forward passes)."""
        V = V0
        for _ in range(K):
            V = self.forward(V, maps)
        return V

    def iterate_fast(self, V0: torch.Tensor, maps: torch.Tensor, K: int) -> torch.Tensor:
        """
        Apply T^K using the fast diagonal power trick: z_K = A^K * g(V0).

        O(1) matrix operations instead of O(K) forward passes.
        Since g is truly invertible (AffineCouplingNet), MSE vs loop should be ~0.
        """
        c = self.map_encoder(maps)
        z0 = self.g.encode(V0)
        z_K = self.A.apply_power(z0, c, K)
        return self.g.decode(z_K)


# ---------------------------------------------------------------------------
# 2. VINBaseline — Standard Value Iteration Network (Tamar et al. 2016)
# ---------------------------------------------------------------------------

class VINBaseline(nn.Module):
    """
    Standard Value Iteration Network.

    Architecture:
    - Reward map R(s) predicted from map features via CNN
    - Q_{k+1}(s,a) = R(s) + gamma * sum_{s'} P(s'|s,a) * V_k(s')
      implemented as conv operations
    - V_k(s) = max_a Q_k(s,a)

    forward() = one VIN step (can iterate K times for convergence).
    """

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        map_channels: int = MAP_CHANNELS,
        hidden_dim: int = 64,
        gamma: float = 0.95,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.state_dim = grid_size * grid_size
        self.gamma = gamma

        # Reward prediction: map features -> per-cell reward
        self.reward_net = nn.Sequential(
            nn.Conv2d(map_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        # Q-value update: [V (1ch), R (1ch)] -> Q for 4 actions
        # Convolutional kernel encodes transition dynamics (4-connected grid)
        self.q_conv = nn.Conv2d(2, 4, kernel_size=3, padding=1)

        # Optional learned gamma scaling per action
        self.gamma_scale = nn.Parameter(torch.ones(4) * gamma)

    def _one_step(self, V_grid: torch.Tensor, R_grid: torch.Tensor) -> torch.Tensor:
        """
        One VIN update step.

        Args:
            V_grid: (B, 1, H, W)
            R_grid: (B, 1, H, W)

        Returns:
            V_next: (B, 1, H, W)
        """
        inp = torch.cat([V_grid, R_grid], dim=1)   # (B, 2, H, W)
        Q = self.q_conv(inp)                        # (B, 4, H, W)
        V_next = Q.max(dim=1, keepdim=True)[0]      # (B, 1, H, W)
        return V_next

    def forward(self, V: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        """
        One VIN step.

        Args:
            V:    (B, state_dim) flattened value map
            maps: (B, 3, H, W)

        Returns:
            V_next: (B, state_dim)
        """
        B = V.size(0)
        V_grid = V.view(B, 1, self.grid_size, self.grid_size)
        R_grid = self.reward_net(maps)
        V_next = self._one_step(V_grid, R_grid)
        return V_next.view(B, self.state_dim)

    def iterate_loop(self, V0: torch.Tensor, maps: torch.Tensor, K: int) -> torch.Tensor:
        """Apply K VIN steps."""
        V = V0
        for _ in range(K):
            V = self.forward(V, maps)
        return V


# ---------------------------------------------------------------------------
# 3. UnconstrainedLinearizer — Ablation: no contraction guarantee
# ---------------------------------------------------------------------------

class UnconstrainedLinearizer(nn.Module):
    """
    Same as ContractiveLinearizer but A(c) is an unconstrained linear layer.
    No tanh scaling, so spectral radius can be > 1 → may diverge.

    Key ablation: shows what contraction buys in terms of convergence stability.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        map_channels: int = MAP_CHANNELS,
        n_coupling_layers: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim

        self.map_encoder = MapEncoder(map_channels, out_dim=CONTEXT_DIM)
        self.g = AffineCouplingNet(dim=state_dim, n_coupling_layers=n_coupling_layers,
                                   hidden_dim=hidden_dim)
        # Unconstrained: just an MLP producing diagonal, no tanh bounding
        self.A_net = nn.Sequential(
            nn.Linear(CONTEXT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim),
        )

    def _get_diagonal(self, maps: torch.Tensor) -> torch.Tensor:
        c = self.map_encoder(maps)
        return self.A_net(c)  # unconstrained, may have |eig| > 1

    def forward(self, V: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        z = self.g.encode(V)
        diag = self._get_diagonal(maps)
        z_next = diag * z
        return self.g.decode(z_next)

    def iterate_loop(self, V0: torch.Tensor, maps: torch.Tensor, K: int) -> torch.Tensor:
        V = V0
        for _ in range(K):
            V = self.forward(V, maps)
        return V


# ---------------------------------------------------------------------------
# 4. IterativeMLPBaseline — Residual MLP, no contraction guarantee
# ---------------------------------------------------------------------------

class IterativeMLPBaseline(nn.Module):
    """
    Standard residual MLP: V_{k+1} = V_k + MLP(V_k, c).

    No contraction guarantee — can diverge with many iterations.
    Demonstrates what happens without architectural inductive bias.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        map_channels: int = MAP_CHANNELS,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim

        self.map_encoder = MapEncoder(map_channels, out_dim=CONTEXT_DIM)
        # Residual MLP taking [V, context] -> delta_V
        self.delta_net = nn.Sequential(
            nn.Linear(state_dim + CONTEXT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        # Initialize last layer to near-zero for stable training start
        nn.init.uniform_(self.delta_net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.delta_net[-1].bias)

    def forward(self, V: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        """
        V_{k+1} = V_k + MLP(V_k, c)
        """
        c = self.map_encoder(maps)
        inp = torch.cat([V, c], dim=-1)
        delta = self.delta_net(inp)
        return V + delta

    def iterate_loop(self, V0: torch.Tensor, maps: torch.Tensor, K: int) -> torch.Tensor:
        V = V0
        for _ in range(K):
            V = self.forward(V, maps)
        return V


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_contractive(spectral_bound: float = 0.99, **kwargs) -> ContractiveLinearizer:
    """Create a ContractiveLinearizer with the given spectral bound."""
    return ContractiveLinearizer(spectral_bound=spectral_bound, **kwargs)
