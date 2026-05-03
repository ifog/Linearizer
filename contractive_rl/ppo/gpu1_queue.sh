#!/bin/bash
# GPU 1 queue: Hopper (running, PID 2021049) -> Breakout -> Walker2d
GPU=1
HOPPER_PID=2021049
STATE_LOG=/home/nvidia/Linearizer/contractive_rl/logs/gpu1_queue.log
ROOT=/home/nvidia/Linearizer
ENV_PREFIX="env CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp PYTHONPATH=/home/nvidia/.local/lib/python3.8/site-packages"
PY=/home/nvidia/anaconda3/envs/rlgpu/bin/python

cd "$ROOT"

echo "[gpu1_queue] start at $(date) — waiting for Hopper PID $HOPPER_PID" >> "$STATE_LOG"
while kill -0 "$HOPPER_PID" 2>/dev/null; do sleep 60; done
echo "[gpu1_queue] Hopper finished at $(date) — launching Breakout" >> "$STATE_LOG"

$ENV_PREFIX $PY -u contractive_rl/dqn/run_atari_dqn.py --env ALE/Breakout-v5 \
  > /home/nvidia/Linearizer/contractive_rl/logs/breakout_dqn.log 2>&1
echo "[gpu1_queue] Breakout finished at $(date) — launching Walker2d" >> "$STATE_LOG"

$ENV_PREFIX $PY -u contractive_rl/ppo/run_mujoco_ppo.py --env Walker2d-v4 \
  > /home/nvidia/Linearizer/contractive_rl/logs/walker_ppo.log 2>&1
echo "[gpu1_queue] Walker2d finished at $(date) — queue done" >> "$STATE_LOG"
