"""
Models for Contractive Linearizer RL experiments.

Four models:
1. ContractiveLinearizer  — diagonal contractive A(c) = diag(tanh(s(c))), spectral radius < 1
2. UnconstrainedLinearizer — same arch but A(c) is unconstrained MLP output
3. MLPBaseline            — direct MLP value predictor
4. VINBaseline            — convolutional value iteration network

Grid size: 10x10 = 100 states
Latent dim: 64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


GRID_SIZE = 10
STATE_DIM = GRID_SIZE * GRID_SIZE      # 100
MAP_CHANNELS = 3                        # obstacle, goal, free
LATENT_DIM = 64


# ---------------------------------------------------------------------------
# Shared encoder/decoder for the Linearizer models
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """g: R^100 -> R^64  (2-layer MLP with tanh)"""

    def __init__(self, state_dim=STATE_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """g^{-1}: R^64 -> R^100  (2-layer MLP with tanh hidden)"""

    def __init__(self, state_dim=STATE_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, state_dim),
        )

    def forward(self, z):
        return self.net(z)


class MapEncoder(nn.Module):
    """
    Encode the (3, H, W) map into a flat context vector.
    Uses a small conv net then flatten.
    """

    def __init__(self, grid_size=GRID_SIZE, map_channels=MAP_CHANNELS, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(map_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out_dim = 32 * grid_size * grid_size
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, maps):
        # maps: (B, 3, H, W)
        x = self.conv(maps)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 1. ContractiveLinearizer
# ---------------------------------------------------------------------------

class ContractiveLinearizer(nn.Module):
    """
    T(x, c) = g^{-1}( A(c) * g(x) )

    A(c) = diag( scale * tanh(s(c)) )
    Guaranteed spectral radius < scale < 1

    Fast iteration:
        z_K = A(c)^K * z_0  (element-wise power since A is diagonal)
        x_K = g^{-1}(z_K)
    """

    def __init__(
        self,
        state_dim=STATE_DIM,
        latent_dim=LATENT_DIM,
        grid_size=GRID_SIZE,
        map_channels=MAP_CHANNELS,
        spectral_scale=0.99,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.spectral_scale = spectral_scale

        self.encoder = Encoder(state_dim, latent_dim)
        self.decoder = Decoder(state_dim, latent_dim)
        self.map_encoder = MapEncoder(grid_size, map_channels, out_dim=128)

        # MLP that produces diagonal of A(c): 128 -> 64
        self.A_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def get_diagonal(self, map_context):
        """
        Compute diagonal of A(c).

        Returns:
            diag: (B, latent_dim) in (-spectral_scale, spectral_scale)
        """
        ctx = self.map_encoder(map_context)
        s = self.A_net(ctx)
        return self.spectral_scale * torch.tanh(s)

    def forward(self, x, map_context):
        """
        One application of T.

        Args:
            x: (B, state_dim) value map
            map_context: (B, 3, H, W) map encoding

        Returns:
            x_next: (B, state_dim)
        """
        z = self.encoder(x)
        diag = self.get_diagonal(map_context)  # (B, latent_dim)
        z_next = diag * z
        return self.decoder(z_next)

    def iterate_loop(self, x0, map_context, K):
        """Apply T K times using a loop."""
        x = x0
        for _ in range(K):
            x = self.forward(x, map_context)
        return x

    def iterate_fast(self, x0, map_context, K):
        """
        Apply T^K using the fast diagonal power trick:
            z_K = diag^K * z_0
        """
        z0 = self.encoder(x0)
        diag = self.get_diagonal(map_context)  # (B, latent_dim)
        diag_K = diag ** K
        z_K = diag_K * z0
        return self.decoder(z_K)


# ---------------------------------------------------------------------------
# 2. UnconstrainedLinearizer
# ---------------------------------------------------------------------------

class UnconstrainedLinearizer(nn.Module):
    """
    Same as ContractiveLinearizer but A(c) is an unconstrained full matrix
    (or unconstrained diagonal — we use unconstrained diagonal for fair comparison).
    """

    def __init__(
        self,
        state_dim=STATE_DIM,
        latent_dim=LATENT_DIM,
        grid_size=GRID_SIZE,
        map_channels=MAP_CHANNELS,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(state_dim, latent_dim)
        self.decoder = Decoder(state_dim, latent_dim)
        self.map_encoder = MapEncoder(grid_size, map_channels, out_dim=128)

        # Unconstrained: no tanh, no scale
        self.A_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def get_diagonal(self, map_context):
        ctx = self.map_encoder(map_context)
        return self.A_net(ctx)  # unconstrained

    def forward(self, x, map_context):
        z = self.encoder(x)
        diag = self.get_diagonal(map_context)
        z_next = diag * z
        return self.decoder(z_next)

    def iterate_loop(self, x0, map_context, K):
        x = x0
        for _ in range(K):
            x = self.forward(x, map_context)
        return x


# ---------------------------------------------------------------------------
# 3. MLPBaseline
# ---------------------------------------------------------------------------

class MLPBaseline(nn.Module):
    """
    Direct MLP: map_encoding -> V*

    Does not iterate — single forward pass predicts V* directly from the map.
    Used as a non-iterative baseline.
    """

    def __init__(
        self,
        state_dim=STATE_DIM,
        grid_size=GRID_SIZE,
        map_channels=MAP_CHANNELS,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.map_encoder = MapEncoder(grid_size, map_channels, out_dim=256)

        self.value_net = nn.Sequential(
            nn.Linear(256 + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim),
        )

    def forward(self, x, map_context):
        """
        Args:
            x: (B, state_dim) current value estimate
            map_context: (B, 3, H, W)

        Returns:
            (B, state_dim) updated value
        """
        ctx = self.map_encoder(map_context)
        inp = torch.cat([ctx, x], dim=-1)
        return self.value_net(inp)

    def iterate_loop(self, x0, map_context, K):
        """Apply T K times."""
        x = x0
        for _ in range(K):
            x = self.forward(x, map_context)
        return x


# ---------------------------------------------------------------------------
# 4. VINBaseline (Value Iteration Network)
# ---------------------------------------------------------------------------

class VINBaseline(nn.Module):
    """
    Convolutional Value Iteration Network.

    Implements K iterations of conv-based Bellman updates.
    Architecture follows VIN (Tamar et al., 2016) in spirit:
    - Reward network: map -> R
    - Q network: R + V -> Q (conv)
    - Value: max over actions

    For simplicity, we run this as an iterative model like the others —
    forward() = one step of VIN update.
    """

    def __init__(
        self,
        grid_size=GRID_SIZE,
        map_channels=MAP_CHANNELS,
        state_dim=STATE_DIM,
        k_vin=10,
        hidden_dim=64,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.state_dim = state_dim

        # Reward prediction: map -> reward map
        self.reward_net = nn.Sequential(
            nn.Conv2d(map_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        # Q-value convolutional layer: [V, R] -> Q (one per action)
        # Input: V (1 channel) + R (1 channel) = 2 channels
        self.q_conv = nn.Conv2d(2, 4, kernel_size=3, padding=1)  # 4 actions

        # Final value readout
        self.value_conv = nn.Conv2d(4, 1, kernel_size=1)

    def _one_step(self, V_grid, R_grid):
        """
        One VIN update step.

        Args:
            V_grid: (B, 1, H, W) value grid
            R_grid: (B, 1, H, W) reward grid

        Returns:
            V_next: (B, 1, H, W)
        """
        inp = torch.cat([V_grid, R_grid], dim=1)  # (B, 2, H, W)
        Q = self.q_conv(inp)                       # (B, 4, H, W)
        V_next = Q.max(dim=1, keepdim=True)[0]     # (B, 1, H, W)
        return V_next

    def forward(self, x, map_context):
        """
        One step of VIN update.

        Args:
            x: (B, state_dim) flattened value map
            map_context: (B, 3, H, W)

        Returns:
            (B, state_dim) updated flattened value map
        """
        B = x.size(0)
        V_grid = x.view(B, 1, self.grid_size, self.grid_size)
        R_grid = self.reward_net(map_context)
        V_next = self._one_step(V_grid, R_grid)
        return V_next.view(B, self.state_dim)

    def iterate_loop(self, x0, map_context, K):
        """Apply T K times."""
        x = x0
        for _ in range(K):
            x = self.forward(x, map_context)
        return x


# ---------------------------------------------------------------------------
# Ablation variants of ContractiveLinearizer
# ---------------------------------------------------------------------------

def make_contractive_with_scale(scale, **kwargs):
    """Create a ContractiveLinearizer with given spectral scale."""
    return ContractiveLinearizer(spectral_scale=scale, **kwargs)
