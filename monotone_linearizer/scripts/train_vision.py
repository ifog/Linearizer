"""Train the Monotone Linearizer (or its g = Id ablation) on an image-classification dataset.

The same script supports both legs of the A1 ablation: pass `--use_flow 1`
for the full Monotone Linearizer, `--use_flow 0` to recover monDEQ exactly.

Logs each epoch's train loss, test accuracy, mean equilibrium error, and mean
solver iterations. Writes history.json + model.pt under --out.

Usage (Fashion-MNIST CPU smoke):
    PYTHONPATH=/home/nvidia/Linearizer:/tmp/claude/pylib \\
        /home/nvidia/anaconda3/bin/python monotone_linearizer/scripts/train_vision.py \\
        --dataset fashion_mnist --epochs 5 --subset_size 5000 \\
        --use_flow 1 --out monotone_linearizer/results/fmnist_flow

Usage (CIFAR-10 GPU):
    PYTHONPATH=/home/nvidia/Linearizer:/tmp/claude/pylib \\
        /home/nvidia/anaconda3/bin/python monotone_linearizer/scripts/train_vision.py \\
        --dataset cifar10 --epochs 30 --device cuda --use_flow 1 \\
        --out monotone_linearizer/results/cifar10_flow
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from monotone_linearizer.data import get_loaders
from monotone_linearizer.models.vision import VisionMonotoneLinearizer, warmup_actnorm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="fashion_mnist",
                   choices=["mnist", "fashion_mnist", "cifar10", "cifar100"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--subset_size", type=int, default=None,
                   help="Use only this many training samples (None = full).")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--encoder_base", type=int, default=32)
    p.add_argument("--m", type=float, default=0.1)
    p.add_argument("--use_flow", type=int, default=1)
    p.add_argument("--flow_blocks", type=int, default=4)
    p.add_argument("--flow_hidden", type=int, default=128)
    p.add_argument("--solver", type=str, default="forward_backward",
                   choices=["naive", "forward_backward", "peaceman_rachford"])
    p.add_argument("--solver_max_iter", type=int, default=30)
    p.add_argument("--solver_tol", type=float, default=1e-3)
    p.add_argument("--solver_step", type=float, default=0.8)
    p.add_argument("--jac_reg", type=float, default=1e-3,
                   help="L2 penalty on the unconstrained core parameter A; "
                        "controls operator Lipschitz (Bai-Koltun-Kolter ICML 2021).")
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--log_every", type=int, default=100)
    return p.parse_args()


@torch.no_grad()
def evaluate(model: VisionMonotoneLinearizer, loader, device: str) -> dict:
    model.eval()
    n, correct, loss_sum = 0, 0, 0.0
    eq_err_sum, iter_sum, n_info = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, info = model(x, return_info=True)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += float(loss.item())
        correct += int((logits.argmax(-1) == y).sum().item())
        n += x.shape[0]
        eq_err_sum += float(info.final_err)
        iter_sum += int(info.iters)
        n_info += 1
    return {
        "loss": loss_sum / n,
        "acc": correct / n,
        "eq_err": eq_err_sum / max(n_info, 1),
        "iters": iter_sum / max(n_info, 1),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = get_loaders(
        args.dataset,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print(f"[setup] dataset={args.dataset} n_classes={loaders.n_classes} "
          f"in_ch={loaders.in_channels} img={loaders.img_size} "
          f"train_batches={len(loaders.train)} use_flow={args.use_flow} "
          f"device={args.device}", flush=True)

    model = VisionMonotoneLinearizer(
        in_channels=loaders.in_channels,
        img_size=loaders.img_size,
        n_classes=loaders.n_classes,
        latent_dim=args.latent_dim,
        encoder_base=args.encoder_base,
        m=args.m,
        use_flow=bool(args.use_flow),
        flow_blocks=args.flow_blocks,
        flow_hidden=args.flow_hidden,
        solver=args.solver,
        solver_max_iter=args.solver_max_iter,
        solver_tol=args.solver_tol,
        solver_step=args.solver_step,
    ).to(args.device)

    # ActNorm data-init for the flow (no-op when use_flow=False).
    warmup_actnorm(model, loaders.train, n_batches=1, device=args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[setup] params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"args": vars(args), "epochs": []}
    t0 = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_run, n_run, correct_run = 0.0, 0, 0
        for batch_i, (x, y) in enumerate(loaders.train):
            step += 1
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            logits = model(x)
            loss_ce = F.cross_entropy(logits, y)
            loss = loss_ce
            if args.jac_reg > 0:
                # Penalize ||A||_F^2 (Jacobian-norm proxy: ||I - W||_F^2 = ||mI + A^TA||_F^2,
                # easier knob is just ||A||_F^2).
                loss = loss + args.jac_reg * model.core.A.pow(2).sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()
            loss_run += float(loss.item()) * x.shape[0]
            correct_run += int((logits.argmax(-1) == y).sum().item())
            n_run += x.shape[0]
            if step % args.log_every == 0:
                print(
                    f"  step {step:6d}  ep {epoch} batch {batch_i+1}/{len(loaders.train)}  "
                    f"loss {loss.item():.4f}  acc-running {correct_run / n_run:.4f}",
                    flush=True,
                )
        eval_stats = evaluate(model, loaders.test, args.device)
        elapsed = time.time() - t0
        ep_record = {
            "epoch": epoch,
            "train_loss": loss_run / n_run,
            "train_acc": correct_run / n_run,
            **{f"test_{k}": v for k, v in eval_stats.items()},
            "elapsed_s": elapsed,
        }
        history["epochs"].append(ep_record)
        print(
            f"[epoch {epoch}/{args.epochs}] train_loss {ep_record['train_loss']:.4f}  "
            f"train_acc {ep_record['train_acc']:.4f}  test_acc {eval_stats['acc']:.4f}  "
            f"eq_err {eval_stats['eq_err']:.4f}  iters {eval_stats['iters']:.1f}  "
            f"({elapsed:.0f}s)",
            flush=True,
        )

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, out_dir / "model.pt")
    print(f"\nSaved history -> {out_dir / 'history.json'}")
    print(f"Saved model   -> {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
