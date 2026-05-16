"""Sanity-check that the re-exported VectorFlow round-trips."""

from __future__ import annotations

import torch

from monotone_linearizer.flows import VectorFlow


def test_round_trip() -> None:
    torch.manual_seed(0)
    flow = VectorFlow(dim=16, n_blocks=4, hidden=32)
    flow.train()
    _ = flow(torch.randn(16, 16))  # ActNorm warm-up.
    flow.eval()
    x = torch.randn(8, 16)
    z = flow(x)
    x_rec = flow.inverse(z)
    assert torch.allclose(x, x_rec, atol=1e-4), (x - x_rec).abs().max().item()


if __name__ == "__main__":
    test_round_trip()
    print("OK: invertibility test passed.")
