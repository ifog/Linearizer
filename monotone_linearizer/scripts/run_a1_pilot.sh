#!/usr/bin/env bash
# Week-3 A1 ablation pilot on the GPU we have.
# Each leg: use_flow=1 (Monotone Linearizer) and use_flow=0 (monDEQ exact).
#
# Output: monotone_linearizer/results/a1_pilot/<dataset>/<flow|noflow>/{history.json, model.pt}
#
# Usage:  bash monotone_linearizer/scripts/run_a1_pilot.sh

set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/nvidia/Linearizer:/tmp/claude/pylib
PY=/home/nvidia/anaconda3/bin/python

ROOT=monotone_linearizer/results/a1_pilot
mkdir -p $ROOT

run_one () {
  local ds=$1; local subset=$2; local epochs=$3; local lr=$4; local m=$5
  local lat=$6; local base=$7; local blocks=$8; local hid=$9; local fl=${10}
  local tag=$([ "$fl" = "1" ] && echo "flow" || echo "noflow")
  local out=$ROOT/${ds}/${tag}
  mkdir -p $out
  echo "=== ${ds}  use_flow=${fl}  -> ${out} ==="
  CUDA_VISIBLE_DEVICES=0 $PY -u monotone_linearizer/scripts/train_vision.py \
    --dataset $ds ${subset:+--subset_size $subset} \
    --epochs $epochs --lr $lr --m $m \
    --weight_decay 1e-3 --jac_reg 1e-3 --grad_clip 0.5 \
    --latent_dim $lat --encoder_base $base \
    --flow_blocks $blocks --flow_hidden $hid \
    --use_flow $fl --solver forward_backward --solver_max_iter 40 --solver_tol 1e-3 --solver_step 0.8 \
    --batch_size 128 --num_workers 2 \
    --device cuda --out $out \
    2>&1 | tee $out/train.log
}

# Fashion-MNIST: full set, 15 epochs (more headroom under the lower LR).
run_one fashion_mnist "" 15 5e-4 0.2 128 32 4 128 1
run_one fashion_mnist "" 15 5e-4 0.2 128 32 4 128 0

# CIFAR-10: full set, 20 epochs.
run_one cifar10       "" 20 5e-4 0.2 128 32 4 128 1
run_one cifar10       "" 20 5e-4 0.2 128 32 4 128 0

echo
echo "=== DONE ==="
echo "Compare with: python monotone_linearizer/scripts/summarize_a1.py --root $ROOT"
