"""
Diagonal contractive linear operator with guaranteed spectral radius < alpha < 1.

A(c) = diag(alpha * tanh(MLP(c)))

The spectral radius is guaranteed to be < alpha since all eigenvalues satisfy:
  |lambda_i| = alpha * |tanh(h_i)| < alpha < 1

For fast iteration: A(c)^K is element-wise power of the diagonal vector.
"""

import torch
import torch.nn as nn


class DiagonalContractiveOp(nn.Module):
    """
    A(c) = diag(alpha * tanh(MLP(c)))
    Spectral radius guaranteed < alpha < 1.

    For fast iteration: A(c)^K is element-wise power of the diagonal.

    Args:
        context_dim: dimension of context vector c
        latent_dim: dimension of the latent space (size of diagonal)
        spectral_bound: alpha, must be in (0, 1). Spectral radius < alpha.
    """

    def __init__(self, context_dim: int, latent_dim: int, spectral_bound: float = 0.99):
        super().__init__()
        if not (0.0 < spectral_bound < 1.0):
            raise ValueError(f"spectral_bound must be in (0,1), got {spectral_bound}")

        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.spectral_bound = spectral_bound

        # MLP that produces raw (unconstrained) values; tanh+scale is applied after
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        # Initialize to small values so initial eigenvalues are near 0
        nn.init.uniform_(self.mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def get_eigenvalues(self, c: torch.Tensor) -> torch.Tensor:
        """
        Compute eigenvalues of A(c).

        Args:
            c: (B, context_dim) context tensor

        Returns:
            eigs: (B, latent_dim) values in (-spectral_bound, spectral_bound)
        """
        raw = self.mlp(c)
        return self.spectral_bound * torch.tanh(raw)

    def apply(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute A(c) @ z  (element-wise since A is diagonal).

        Args:
            z: (B, latent_dim)
            c: (B, context_dim)

        Returns:
            (B, latent_dim)
        """
        eigs = self.get_eigenvalues(c)
        return eigs * z

    def apply_power(self, z: torch.Tensor, c: torch.Tensor, K: int) -> torch.Tensor:
        """
        Compute A(c)^K @ z  (element-wise z * eig^K since A is diagonal).

        This is O(1) in K — no matrix multiplications needed.

        Args:
            z: (B, latent_dim)
            c: (B, context_dim)
            K: number of steps (non-negative integer)

        Returns:
            (B, latent_dim)
        """
        eigs = self.get_eigenvalues(c)
        eigs_K = eigs ** K
        return eigs_K * z
