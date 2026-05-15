#!/bin/bash
HC_PID=2020966
HOPPER_PID=2021049

while kill -0 "$HC_PID" 2>/dev/null && kill -0 "$HOPPER_PID" 2>/dev/null; do
  sleep 60
done

if ! kill -0 "$HC_PID" 2>/dev/null; then
  GPU=0
  FINISHED="halfcheetah"
else
  GPU=1
  FINISHED="hopper"
fi

echo "[watcher] $FINISHED finished — launching Walker2d on GPU $GPU at $(date)" \
  >> /home/nvidia/Linearizer/contractive_rl/logs/walker_watcher.log

cd /home/nvidia/Linearizer
exec env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp \
  PYTHONPATH=/home/nvidia/.local/lib/python3.8/site-packages \
  /home/nvidia/anaconda3/envs/rlgpu/bin/python -u \
  contractive_rl/ppo/run_mujoco_ppo.py --env Walker2d-v4 \
  > /home/nvidia/Linearizer/contractive_rl/logs/walker_ppo.log 2>&1
