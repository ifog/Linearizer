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
from shared.contractive_operator import DiagonalContractiveOp, FullMatrixContractiveOp

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
    V_{k+1} = g^{-1}(A(c) * g(V_k) + b(c))

    Affine contractive map with context-dependent bias b(c).
    Fixed point: g(V*) = (I - A(c))^{-1} * b(c), which varies per map context c.
    Spectral radius of A bounded by spectral_bound < 1 → unique fixed point per c.

    Fast iteration: iterate_fast(V0, c, K) computes T^K in O(1) via affine formula.
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
        # Bias: context -> state_dim (latent). Provides context-dependent fixed point.
        self.b_net = nn.Sequential(
            nn.Linear(CONTEXT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim),
        )

    def forward(self, V: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        """One step: V_{k+1} = g^{-1}(A(c)*g(V_k) + b(c))."""
        c = self.map_encoder(maps)
        z = self.g.encode(V)
        b = self.b_net(c)
        z_next = self.A.apply(z, c) + b
        return self.g.decode(z_next)

    def iterate_loop(self, V0: torch.Tensor, maps: torch.Tensor, K: int) -> torch.Tensor:
        """Apply T K times using a loop (O(K) forward passes)."""
        V = V0
        for _ in range(K):
            V = self.forward(V, maps)
        return V

    def iterate_fast(self, V0: torch.Tensor, maps: torch.Tensor, K: int) -> torch.Tensor:
        """
        Apply T^K using the affine fast formula: O(1) in K.

        For T(z) = eigs*z + b, after K steps from z0:
          z_K = eigs^K * (z0 - z*) + z*,  where z* = b / (1 - eigs).
        """
        c = self.map_encoder(maps)
        z0 = self.g.encode(V0)
        b = self.b_net(c)
        eigs = self.A.get_eigenvalues(c)
        z_star = b / (1.0 - eigs)             # fixed point in latent space
        eigs_K = eigs ** K
        z_K = eigs_K * (z0 - z_star) + z_star
        return self.g.decode(z_K)


# ---------------------------------------------------------------------------
# 2. VINBaseline — Standard Value Iteration Network (Tamar et al. 2016)
# ---------------------------------------------------------------------------

class VINBaseline(nn.Module):
    """
    Standard Value Iteration Network (corrected architecture).

    Architecture:
    - R(s) predicted from map features via CNN
    - Q_{k+1}(s,a) = R(s) + transition_conv(V_k)[s,a]  — V and R are separate
    - V_k(s) = max_a Q_k(s,a)
    - Obstacle cells zeroed after each step to prevent contamination

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

        # Reward prediction: map features -> per-cell reward (1 channel)
        self.reward_net = nn.Sequential(
            nn.Conv2d(map_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        # Transition conv: V only -> Q for 4 actions (correct VIN decomposition)
        # R is added separately, ensuring Q = transition(V) + R
        self.q_conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)

    def _one_step(self, V_grid: torch.Tensor, R_grid: torch.Tensor,
                  obstacle_mask: torch.Tensor) -> torch.Tensor:
        """
        One VIN update step: Q = transition(V) + R, V = max_a Q, mask obstacles.

        Args:
            V_grid:       (B, 1, H, W)
            R_grid:       (B, 1, H, W)
            obstacle_mask:(B, 1, H, W) — 1 at obstacle cells

        Returns:
            V_next: (B, 1, H, W)
        """
        Q = self.q_conv(V_grid) + R_grid    # (B, 4, H, W): transition + reward
        V_next = Q.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        return V_next * (1.0 - obstacle_mask)   # zero obstacle cells

    def forward(self, V: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        """
        One VIN step.

        Args:
            V:    (B, state_dim) flattened value map
            maps: (B, 3, H, W)  — channel 0 = obstacles

        Returns:
            V_next: (B, state_dim)
        """
        B = V.size(0)
        V_grid = V.view(B, 1, self.grid_size, self.grid_size)
        R_grid = self.reward_net(maps)
        obstacle_mask = maps[:, 0:1, :, :]          # (B, 1, H, W)
        V_next = self._one_step(V_grid, R_grid, obstacle_mask)
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
    Ablation: same architecture as ContractiveLinearizer but no spectral constraint.
    A(c) is an unconstrained diagonal — spectral radius can exceed 1 → may diverge.

    Has the same bias term b(c) as ContractiveLinearizer for a fair comparison.
    Only difference: no tanh bounding on eigenvalues.
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
        # Unconstrained diagonal: no tanh bounding, spectral radius may exceed 1
        self.A_net = nn.Sequential(
            nn.Linear(CONTEXT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, state_dim),
        )
        # Same bias term as ContractiveLinearizer (for fair comparison)
        self.b_net = nn.Sequential(
            nn.Linear(CONTEXT_DIM, 256), nn.ReLU(),
            nn.Linear(256, state_dim),
        )

    def forward(self, V: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        c = self.map_encoder(maps)
        z = self.g.encode(V)
        diag = self.A_net(c)
        b = self.b_net(c)
        z_next = diag * z + b
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


# ---------------------------------------------------------------------------
# 5. FullContractiveLinearizer — Ablation: full A(c) instead of diagonal
# ---------------------------------------------------------------------------

class FullContractiveLinearizer(nn.Module):
    """
    Same as ContractiveLinearizer but with a full (non-diagonal) operator A(c).

    A(c) ∈ R^{d×d} with ‖A(c)‖_2 < spectral_bound, parameterized via spectral
    normalization (FullMatrixContractiveOp). This isolates the effect of the
    diagonal restriction — Proposition 1 implies the parameterization is already
    universal with diagonal Λ, so this ablation tests whether the *inductive
    bias* of full A helps in practice.

    Cost: closed-form fixed point requires solving a d×d linear system per
    state instead of element-wise division — O(d^3) per state vs O(d).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        map_channels: int = MAP_CHANNELS,
        spectral_bound: float = 0.9,
        n_coupling_layers: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.spectral_bound = spectral_bound

        self.map_encoder = MapEncoder(map_channels, out_dim=CONTEXT_DIM)
        self.g = AffineCouplingNet(dim=state_dim, n_coupling_layers=n_coupling_layers,
                                   hidden_dim=hidden_dim)
        self.A = FullMatrixContractiveOp(context_dim=CONTEXT_DIM, latent_dim=state_dim,
                                         spectral_bound=spectral_bound)
        self.b_net = nn.Sequential(
            nn.Linear(CONTEXT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim),
        )

    def forward(self, V: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        """One step: V_{k+1} = g^{-1}(A(c) g(V) + b(c))."""
        c = self.map_encoder(maps)
        z = self.g.encode(V)
        b = self.b_net(c)
        z_next = self.A.apply(z, c) + b
        return self.g.decode(z_next)

    def iterate_loop(self, V0: torch.Tensor, maps: torch.Tensor, K: int) -> torch.Tensor:
        V = V0
        for _ in range(K):
            V = self.forward(V, maps)
        return V

    def iterate_fast(self, V0: torch.Tensor, maps: torch.Tensor, K: int) -> torch.Tensor:
        """
        For affine T(z) = Az + b with ‖A‖_2 < 1, the K-step iterate is:
          z_K = A^K (z_0 - z*) + z*,  z* = (I - A)^{-1} b.

        We compute z* via linear solve (closed form for K = ∞) and then add
        a residual A^K (z_0 - z*) for finite K. For K large or unspecified,
        we just return z*.
        """
        c = self.map_encoder(maps)
        z0 = self.g.encode(V0)
        b = self.b_net(c)
        z_star = self.A.fixed_point(b, c)        # (B, d)
        if K is None or K >= 200:
            return self.g.decode(z_star)
        # Compute A^K (z0 - z*) via repeated multiplication
        A = self.A.get_matrix(c)                 # (B, d, d)
        delta = (z0 - z_star).unsqueeze(-1)      # (B, d, 1)
        for _ in range(K):
            delta = A @ delta
        z_K = z_star + delta.squeeze(-1)
        return self.g.decode(z_K)
