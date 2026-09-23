#!/usr/bin/env bash
# X5 replicate on a fresh pod: serve the uploaded task-1 LoRA checkpoint, run fixed / primitive / memory
# arms on NEW seeds (101-151) with videos. Expects /workspace/ckpt/7999/{params,assets} uploaded from the Mac.
set -u
export DEBIAN_FRONTEND=noninteractive
cd /workspace
[ -d memory_robotics/.git ] || git clone -q https://github.com/Phillips-Ugo/memory_robotics.git
cd memory_robotics && git pull -q origin main
bash scripts/setup_gpu_box.sh > /workspace/setup.log 2>&1 || { echo "[SETUP FAILED]"; tail -20 /workspace/setup.log; exit 1; }
. /workspace/env.sh
python3 scripts/patch_openpi_config.py --repo-id belu/rma_task1
python3 scripts/patch_rma_adapter_hooks.py
echo "[SETUP OK $(date -u +%H:%M)]"
for i in $(seq 1 200); do [ -f /workspace/ckpt/UPLOAD_DONE ] && break; sleep 15; done
[ -f /workspace/ckpt/UPLOAD_DONE ] || { echo "[NO CHECKPOINT UPLOAD]"; exit 1; }
cd /workspace/memory_robotics/vendor/openpi
(nohup uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_rma_lora --policy.dir=/workspace/ckpt/7999 > /workspace/server.log 2>&1 < /dev/null &)
for i in $(seq 1 180); do grep -q "server listening" /workspace/server.log && break; sleep 5; done
grep -q "server listening" /workspace/server.log || { echo "[SERVER FAILED]"; tail -20 /workspace/server.log; exit 1; }
echo "[SERVER OK $(date -u +%H:%M)]"
cd /workspace/memory_robotics/vendor/RoboMemArena/evaluation_benchmark
SEED=${SEED:-101}
for ARM in fixed primitive memory; do
  OUT=/workspace/memory_robotics/outputs/x5r_$ARM; rm -rf $OUT /workspace/x5r_$ARM.db /workspace/x5r_$ARM.jsonl
  EXTRA=""; [ "$ARM" = memory ] && EXTRA=", \"scope\": \"episode\""
  echo "[ARM $ARM start $(date -u +%H:%M)]"
  X5_LOG=/workspace/x5r_$ARM.jsonl MUJOCO_GL=egl PYTHONUNBUFFERED=1 ../../rma-venv/bin/python scripts/eval_task1_only.py \
    --adapter-spec /workspace/memory_robotics/scripts/03_rma_memory_adapter.py:build_adapter \
    --adapter-kwargs "{\"mode\": \"$ARM\", \"db\": \"/workspace/x5r_$ARM.db\"$EXTRA}" \
    --num-trials-per-task 51 --seed $SEED --video-out-path $OUT 2>&1 | grep --line-buffered "Episode\|Final result\|Traceback\|Error"
  echo "[ARM $ARM exit ${PIPESTATUS[0]} $(date -u +%H:%M)]"
done
echo "[X5R DONE]"
