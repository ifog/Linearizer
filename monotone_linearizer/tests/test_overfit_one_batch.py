"""Training-loop sanity test: can the model memorise 16 samples?

This is the single most-revealing sanity test for any neural-net training
pipeline. If the model can't drive train loss near zero on a tiny batch in
a few hundred steps, the gradient flow, optimiser hookup, loss formulation,
or numerical conditioning is broken — and you'd waste hours debugging
downstream effects.

We run the test for both legs of the A1 ablation (use_flow=True and
use_flow=False) so we catch regressions in either path. The test must hit
train_acc >= 0.99 within step_budget on both legs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from monotone_linearizer.models.vision import VisionMonotoneLinearizer, warmup_actnorm


def _make_synthetic_batch(B: int, in_channels: int, img_size: int, n_classes: int,
                          *, seed: int = 0, device: str = "cpu"):
    """Random images + random labels — we don't need a real dataset, just
    enough signal to memorise."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, in_channels, img_size, img_size, generator=g)
    y = torch.randint(0, n_classes, (B,), generator=g)
    return x.to(device), y.to(device)


def _overfit(
    *,
    use_flow: bool,
    device: str,
    in_channels: int = 1,
    img_size: int = 28,
    n_classes: int = 5,
    batch_size: int = 16,
    latent_dim: int = 64,
    encoder_base: int = 16,
    flow_blocks: int = 4,
    flow_hidden: int = 64,
    m: float = 0.1,
    lr: float = 3e-3,
    step_budget: int = 300,
) -> tuple[float, list[float]]:
    torch.manual_seed(0)
    x, y = _make_synthetic_batch(batch_size, in_channels, img_size, n_classes, device=device)

    model = VisionMonotoneLinearizer(
        in_channels=in_channels, img_size=img_size, n_classes=n_classes,
        latent_dim=latent_dim, encoder_base=encoder_base,
        m=m, use_flow=use_flow,
        flow_blocks=flow_blocks, flow_hidden=flow_hidden,
        solver="forward_backward",
        solver_max_iter=30, solver_tol=1e-3, solver_step=0.8,
    ).to(device)

    if use_flow:
        # ActNorm data-init using the single batch.
        from monotone_linearizer.monotone.solvers import fixed_point_solve
        model.train()
        with torch.no_grad():
            c = model.encoder(x)
            w_star, _ = fixed_point_solve(model.core, c, method="forward_backward",
                                           max_iter=30, tol=1e-3, step=0.8)
            _ = model.flow(w_star)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses: list[float] = []
    for step in range(step_budget):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        acc = float((model(x).argmax(-1) == y).float().mean().item())
    return acc, losses


def _assert_overfit(use_flow: bool) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    acc, losses = _overfit(use_flow=use_flow, device=device)
    msg = (f"use_flow={use_flow}: final train_acc={acc:.3f}  "
           f"loss[0]={losses[0]:.3f}  loss[-1]={losses[-1]:.3f}")
    assert acc >= 0.99, f"OVERFIT FAILED: {msg}"
    print("  " + msg)


def test_overfit_noflow() -> None:
    """g = Id  (= monDEQ) should memorise 16 random samples in <300 steps."""
    _assert_overfit(use_flow=False)


def test_overfit_flow() -> None:
    """g = learned flow (= Monotone Linearizer) should also memorise."""
    _assert_overfit(use_flow=True)


if __name__ == "__main__":
    test_overfit_noflow()
    test_overfit_flow()
    print("OK: overfit-one-batch test passed.")
