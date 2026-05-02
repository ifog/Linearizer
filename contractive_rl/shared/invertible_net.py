"""
Truly invertible network for 1D vectors using affine coupling layers (NICE/RealNVP style).

Each coupling layer splits z into (z1, z2), then:
  forward:  y1 = z1,  y2 = z2 * exp(s(z1)) + t(z1)
  inverse:  z2 = (y2 - t(y1)) * exp(-s(y1)),  z1 = y1

Alternating which half is transformed gives a fully invertible network.
encode()/decode() are exact inverses to machine precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _STNet(nn.Module):
    """Small MLP producing (scale, translation) from half of the vector."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim * 2),  # s and t stacked
        )
        # Small (not zero) init: zero weight kills gradients in earlier layers
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        out = self.net(x)
        half = out.shape[-1] // 2
        s = out[..., :half]
        t = out[..., half:]
        return s, t


class AffineCouplingNet(nn.Module):
    """
    Truly invertible MLP via stacked affine coupling layers.

    At each layer, split z into (z1, z2), compute:
      forward:  y1 = z1,  y2 = z2 * exp(s(z1)) + t(z1)
      inverse:  z2 = (y2 - t(y1)) * exp(-s(y1)),  z1 = y1
    Alternate which half is transformed each layer.

    This gives exact encode() / decode() with no approximation error.
    decode(encode(x)) == x to machine precision.

    Args:
        dim: input/output dimension
        n_coupling_layers: number of coupling layers (must be even for symmetry)
        hidden_dim: width of each coupling MLP
    """

    def __init__(self, dim: int, n_coupling_layers: int = 6, hidden_dim: int = 128):
        super().__init__()
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")

        self.dim = dim
        self.n_coupling_layers = n_coupling_layers

        # Split sizes: first half and second half
        self.split1 = dim // 2
        self.split2 = dim - self.split1  # handles odd dims

        # Build ST networks for each layer
        # Even layers: transform z2 using z1 as conditioning
        # Odd layers: transform z1 using z2 as conditioning
        self.st_nets = nn.ModuleList()
        for i in range(n_coupling_layers):
            if i % 2 == 0:
                # z1 conditions z2: in_dim=split1, out_dim=split2
                st = _STNet(self.split1, hidden_dim, self.split2)
            else:
                # z2 conditions z1: in_dim=split2, out_dim=split1
                st = _STNet(self.split2, hidden_dim, self.split1)
            self.st_nets.append(st)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x -> z.

        Args:
            x: (..., dim) tensor

        Returns:
            z: (..., dim) tensor
        """
        z = x
        for i, st in enumerate(self.st_nets):
            z1 = z[..., : self.split1]
            z2 = z[..., self.split1 :]

            if i % 2 == 0:
                # Transform z2 conditioned on z1
                s_raw, t = st(z1)
                # Clamp s to [-2, 2] via tanh to prevent exp overflow;
                # preserves invertibility since we apply the same clamp in decode.
                s = torch.tanh(s_raw)
                z2_new = z2 * torch.exp(s) + t
                z = torch.cat([z1, z2_new], dim=-1)
            else:
                # Transform z1 conditioned on z2
                s_raw, t = st(z2)
                s = torch.tanh(s_raw)
                z1_new = z1 * torch.exp(s) + t
                z = torch.cat([z1_new, z2], dim=-1)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass: z -> x. Exact inverse of encode().

        Args:
            z: (..., dim) tensor

        Returns:
            x: (..., dim) tensor
        """
        x = z
        # Reverse the layers
        for i in reversed(range(self.n_coupling_layers)):
            st = self.st_nets[i]
            x1 = x[..., : self.split1]
            x2 = x[..., self.split1 :]

            if i % 2 == 0:
                # Undo: z2_new = z2 * exp(s(z1)) + t(z1)  (with s = tanh(s_raw))
                # => z2 = (z2_new - t(z1)) * exp(-s(z1))
                s_raw, t = st(x1)
                s = torch.tanh(s_raw)
                x2_orig = (x2 - t) * torch.exp(-s)
                x = torch.cat([x1, x2_orig], dim=-1)
            else:
                # Undo: z1_new = z1 * exp(s(z2)) + t(z2)  (with s = tanh(s_raw))
                # => z1 = (z1_new - t(z2)) * exp(-s(z2))
                s_raw, t = st(x2)
                s = torch.tanh(s_raw)
                x1_orig = (x1 - t) * torch.exp(-s)
                x = torch.cat([x1_orig, x2], dim=-1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for encode()."""
        return self.encode(x)
