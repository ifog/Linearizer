"""Summarise the A1 ablation pilot: print and plot test accuracy for
use_flow=0 (monDEQ) vs use_flow=1 (Monotone Linearizer) on each dataset.

Per the blueprint, the decision gate is: gap >= 1% -> proceed to CIFAR-100 GPU
run; gap < 1% -> halt and pivot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p: Path) -> dict | None:
    f = p / "history.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, default=None,
                    help="Path for the comparison figure (default: <root>/a1_comparison.png)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "a1_comparison.png"

    datasets = sorted([p.name for p in root.iterdir() if p.is_dir()])
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)

    print(f"\n{'dataset':<16} | {'monDEQ (g=Id)':>15} | {'M-Linearizer':>15} | {'gap':>10}")
    print("-" * 64)
    summary = {}
    for ax, ds in zip(axes[0], datasets):
        noflow = load(root / ds / "noflow")
        flow = load(root / ds / "flow")
        if noflow is None or flow is None:
            print(f"{ds:<16} | missing data")
            continue
        nf_acc = [e["test_acc"] for e in noflow["epochs"]]
        fl_acc = [e["test_acc"] for e in flow["epochs"]]
        ax.plot(range(1, len(nf_acc) + 1), nf_acc, marker="o", label="monDEQ (g=Id)  [A1]")
        ax.plot(range(1, len(fl_acc) + 1), fl_acc, marker="s", label="Monotone Linearizer (g=flow)")
        ax.set_title(ds)
        ax.set_xlabel("epoch")
        ax.set_ylabel("test accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        best_nf = max(nf_acc)
        best_fl = max(fl_acc)
        gap = best_fl - best_nf
        summary[ds] = {"monDEQ_best": best_nf, "MLin_best": best_fl, "gap": gap}
        print(f"{ds:<16} | {best_nf:>15.4f} | {best_fl:>15.4f} | {gap:>+10.4f}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    (root / "a1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved figure  -> {out}")
    print(f"Saved summary -> {root / 'a1_summary.json'}")


if __name__ == "__main__":
    main()
