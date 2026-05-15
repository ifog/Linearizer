"""Smoke tests for tropical ops: identity, associativity, distributivity, gradients."""

from __future__ import annotations

import torch

from tropical_linearizer.tropical.ops import (
    NEG_INF,
    soft_trop_matmul,
    soft_trop_matvec,
    trop_add,
    trop_identity,
    trop_matmul,
    trop_matvec,
    trop_zero,
)


def test_identity_matvec() -> None:
    n = 5
    I = trop_identity(n)
    x = torch.randn(n)
    y = trop_matvec(I, x)
    assert torch.allclose(x, y, atol=1e-6), (x, y)


def test_identity_matmul() -> None:
    n = 4
    I = trop_identity(n)
    A = torch.randn(n, n)
    assert torch.allclose(trop_matmul(I, A), A, atol=1e-6)
    assert torch.allclose(trop_matmul(A, I), A, atol=1e-6)


def test_associativity() -> None:
    torch.manual_seed(0)
    A = torch.randn(4, 5)
    B = torch.randn(5, 6)
    C = torch.randn(6, 3)
    left = trop_matmul(trop_matmul(A, B), C)
    right = trop_matmul(A, trop_matmul(B, C))
    assert torch.allclose(left, right, atol=1e-5)


def test_distributivity_matvec() -> None:
    # A ⊗ (x ⊕ y) = (A ⊗ x) ⊕ (A ⊗ y) for any x, y.
    torch.manual_seed(1)
    A = torch.randn(3, 4)
    x = torch.randn(4)
    y = torch.randn(4)
    left = trop_matvec(A, trop_add(x, y))
    right = trop_add(trop_matvec(A, x), trop_matvec(A, y))
    assert torch.allclose(left, right, atol=1e-5)


def test_zero_absorbs() -> None:
    # NEG_INF + anything ≈ NEG_INF (well, very small), and the row stays small
    # under matmul.
    n = 3
    A = trop_zero(n, n)
    x = torch.randn(n)
    y = trop_matvec(A, x)
    # All outputs should be in the very-negative range.
    assert (y < NEG_INF / 2).all().item()


def test_matvec_grad() -> None:
    A = torch.randn(3, 4, requires_grad=True)
    x = torch.randn(4, requires_grad=True)
    y = trop_matvec(A, x).sum()
    y.backward()
    # Each row of A picks exactly one argmax; the gradient should be a 0/1 mask
    # summing to 1 along the last axis.
    assert A.grad is not None
    grad = A.grad
    per_row = grad.sum(dim=-1)
    assert torch.allclose(per_row, torch.ones_like(per_row))


def test_soft_approaches_hard() -> None:
    torch.manual_seed(2)
    A = torch.randn(4, 5)
    x = torch.randn(5)
    hard = trop_matvec(A, x)
    soft = soft_trop_matvec(A, x, beta=1e3)
    assert torch.allclose(hard, soft, atol=1e-2), (hard, soft)


def test_soft_matmul_approaches_hard() -> None:
    torch.manual_seed(3)
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    hard = trop_matmul(A, B)
    soft = soft_trop_matmul(A, B, beta=1e3)
    assert torch.allclose(hard, soft, atol=1e-2)


def test_batch_matvec() -> None:
    # A is (m, n), x is (B, n) - should broadcast across B.
    A = torch.randn(3, 4)
    x = torch.randn(2, 4)
    y = trop_matvec(A, x)
    assert y.shape == (2, 3)
    # Spot-check against the loop.
    for b in range(2):
        yb = trop_matvec(A, x[b])
        assert torch.allclose(y[b], yb, atol=1e-6)


if __name__ == "__main__":
    test_identity_matvec()
    test_identity_matmul()
    test_associativity()
    test_distributivity_matvec()
    test_zero_absorbs()
    test_matvec_grad()
    test_soft_approaches_hard()
    test_soft_matmul_approaches_hard()
    test_batch_matvec()
    print("OK: tropical ops tests passed.")
