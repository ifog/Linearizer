"""End-to-end forward + backward for the full MonotoneLinearizer, both with
g = identity (monDEQ baseline) and with g = VectorFlow."""

from __future__ import annotations

import torch

from monotone_linearizer.models.monotone_linearizer import IdentityFlow, build_default


def _train_step(model, x, y) -> float:
    pred = model(x)
    loss = (pred - y).pow(2).mean()
    loss.backward()
    return float(loss.item())


def test_with_flow() -> None:
    torch.manual_seed(0)
    model = build_default(in_dim=8, latent_dim=16, out_dim=4, m=0.2,
                          flow_blocks=2, flow_hidden=32, use_flow=True)
    # ActNorm warm-up.
    model.train()
    with torch.no_grad():
        _ = model.flow(torch.randn(32, 16))
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    loss = _train_step(model, x, y)
    # Any finite, positive loss is fine; we mostly care the backward pass
    # produced gradients without crashing.
    assert loss > 0 and torch.isfinite(torch.tensor(loss))
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is None or torch.isfinite(p.grad).all()


def test_g_identity_recovers_monDEQ() -> None:
    """With use_flow=False the flow is identity and the model is exactly monDEQ."""
    torch.manual_seed(1)
    model = build_default(in_dim=8, latent_dim=16, out_dim=4, m=0.2,
                          use_flow=False)
    assert isinstance(model.flow, IdentityFlow)
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    loss = _train_step(model, x, y)
    assert loss > 0 and torch.isfinite(torch.tensor(loss))


if __name__ == "__main__":
    test_g_identity_recovers_monDEQ()
    test_with_flow()
    print("OK: end-to-end tests passed.")
