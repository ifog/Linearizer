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


class FullMatrixContractiveOp(nn.Module):
    """
    Full (non-diagonal) contractive linear operator for the ablation study
    against DiagonalContractiveOp.

      A(c) = alpha * tanh(scale(c)) * A_raw(c) / sigma(A_raw(c))

    where A_raw ∈ R^{d×d} is produced by an MLP, sigma(·) is the spectral
    norm computed by a few iterations of power iteration, and scale(c) is a
    per-state scalar in (-1, 1) via tanh. This guarantees the spectral norm
    ‖A(c)‖₂ = alpha * |tanh(scale(c))| < alpha < 1 for every c, so A(c) is a
    contraction with rate < alpha.

    Closed-form fixed point of T(z; c) = A(c) z + b(c):
        z* = (I - A(c))^{-1} b(c)
    requires solving a d×d linear system per state.
    """

    def __init__(self, context_dim: int, latent_dim: int,
                 spectral_bound: float = 0.9, n_power_iter: int = 3,
                 hidden_dim: int = 128):
        super().__init__()
        if not (0.0 < spectral_bound < 1.0):
            raise ValueError(f"spectral_bound must be in (0,1), got {spectral_bound}")

        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.spectral_bound = spectral_bound
        self.n_power_iter = n_power_iter

        # Raw matrix entries
        self.matrix_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * latent_dim),
        )
        nn.init.uniform_(self.matrix_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.matrix_mlp[-1].bias)

        # Per-state scalar that bounds spectral norm via tanh
        self.scale_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.scale_mlp[-1].weight)
        nn.init.zeros_(self.scale_mlp[-1].bias)

    @staticmethod
    def _spectral_norm(A: torch.Tensor, n_iter: int) -> torch.Tensor:
        """
        Power iteration for the spectral norm of a batch of matrices.

        Args:
            A: (B, d, d)
            n_iter: number of power iteration steps
        Returns:
            sigma: (B,) — approximate largest singular value of each A[b]
        """
        B, d, _ = A.shape
        u = torch.randn(B, d, device=A.device, dtype=A.dtype)
        u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        for _ in range(n_iter):
            v = torch.einsum('bji,bj->bi', A, u)
            v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            u = torch.einsum('bij,bj->bi', A, v)
            u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sigma = torch.einsum('bi,bij,bj->b', u, A, v).abs()
        return sigma.clamp(min=1e-8)

    def get_matrix(self, c: torch.Tensor) -> torch.Tensor:
        """
        Return the full state-conditioned contractive matrix A(c) of shape
        (B, latent_dim, latent_dim). ‖A(c)‖_2 < spectral_bound.
        """
        B = c.shape[0]
        A_raw = self.matrix_mlp(c).view(B, self.latent_dim, self.latent_dim)
        sigma = self._spectral_norm(A_raw, self.n_power_iter)  # (B,)
        scale = self.spectral_bound * torch.tanh(self.scale_mlp(c)).squeeze(-1)  # (B,)
        # A = (scale / sigma) * A_raw  →  ‖A‖_2 = |scale| < spectral_bound
        coef = (scale / sigma).view(B, 1, 1)
        return coef * A_raw

    def apply(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute A(c) @ z."""
        A = self.get_matrix(c)               # (B, d, d)
        return torch.einsum('bij,bj->bi', A, z)

    def fixed_point(self, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Closed-form fixed point: z* = (I - A(c))^{-1} b.

        Args:
            b: (B, latent_dim)
            c: (B, context_dim)
        Returns:
            z*: (B, latent_dim)
        """
        A = self.get_matrix(c)               # (B, d, d)
        I = torch.eye(self.latent_dim, device=A.device, dtype=A.dtype).expand_as(A)
        return torch.linalg.solve(I - A, b.unsqueeze(-1)).squeeze(-1)
