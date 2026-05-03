#!/bin/bash
# GPU 0 queue: HalfCheetah (running, PID 2020966) -> Pong -> SpaceInvaders
GPU=0
HC_PID=2020966
STATE_LOG=/home/nvidia/Linearizer/contractive_rl/logs/gpu0_queue.log
ROOT=/home/nvidia/Linearizer
ENV_PREFIX="env CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp PYTHONPATH=/home/nvidia/.local/lib/python3.8/site-packages"
PY=/home/nvidia/anaconda3/envs/rlgpu/bin/python

cd "$ROOT"

echo "[gpu0_queue] start at $(date) — waiting for HalfCheetah PID $HC_PID" >> "$STATE_LOG"
while kill -0 "$HC_PID" 2>/dev/null; do sleep 60; done
echo "[gpu0_queue] HalfCheetah finished at $(date) — launching Pong" >> "$STATE_LOG"

$ENV_PREFIX $PY -u contractive_rl/dqn/run_atari_dqn.py --env ALE/Pong-v5 \
  > /home/nvidia/Linearizer/contractive_rl/logs/pong_dqn.log 2>&1
echo "[gpu0_queue] Pong finished at $(date) — launching SpaceInvaders" >> "$STATE_LOG"

$ENV_PREFIX $PY -u contractive_rl/dqn/run_atari_dqn.py --env ALE/SpaceInvaders-v5 \
  > /home/nvidia/Linearizer/contractive_rl/logs/space_invaders_dqn.log 2>&1
echo "[gpu0_queue] SpaceInvaders finished at $(date) — queue done" >> "$STATE_LOG"
