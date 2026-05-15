"""Verify the vector flow is exactly invertible: g(g^{-1}(z)) = z, g^{-1}(g(x)) = x."""

from __future__ import annotations

import torch

from tropical_linearizer.flows.vector_flow import VectorFlow


def test_round_trip() -> None:
    torch.manual_seed(0)
    flow = VectorFlow(dim=16, n_blocks=4, hidden=32)
    flow.eval()
    x = torch.randn(8, 16)
    # Run a forward pass with the flow in train mode once to data-init ActNorm.
    flow.train()
    _ = flow(x)
    flow.eval()
    z = flow(x)
    x_rec = flow.inverse(z)
    assert torch.allclose(x, x_rec, atol=1e-4), torch.max(torch.abs(x - x_rec)).item()


def test_round_trip_other_direction() -> None:
    torch.manual_seed(1)
    flow = VectorFlow(dim=12, n_blocks=4, hidden=32)
    flow.train()
    # Initialize ActNorm.
    _ = flow(torch.randn(16, 12))
    flow.eval()
    z = torch.randn(8, 12)
    x = flow.inverse(z)
    z_rec = flow(x)
    assert torch.allclose(z, z_rec, atol=1e-4), torch.max(torch.abs(z - z_rec)).item()


if __name__ == "__main__":
    test_round_trip()
    test_round_trip_other_direction()
    print("OK: invertibility tests passed.")
