"""For a KleeneTropicalCore, f(f(x)) = f(x) by construction (Lemma 5)."""

from __future__ import annotations

import torch

from tropical_linearizer.flows.vector_flow import VectorFlow
from tropical_linearizer.models.tropical_linearizer import TropicalLinearizer
from tropical_linearizer.tropical.core import KleeneTropicalCore


def test_idempotency_at_init() -> None:
    torch.manual_seed(0)
    n = 8
    flow = VectorFlow(dim=n, n_blocks=2, hidden=32)
    core = KleeneTropicalCore(n=n)
    model = TropicalLinearizer(flow_x=flow, core=core)
    # Warm up ActNorm.
    model.train()
    _ = model.gx(torch.randn(16, n))
    model.eval()

    x = torch.randn(4, n)
    y = model(x)
    yy = model(y)
    diff = torch.max(torch.abs(y - yy)).item()
    assert diff < 1e-3, f"Idempotency error too large: {diff}"


if __name__ == "__main__":
    test_idempotency_at_init()
    print("OK: idempotency test passed.")
