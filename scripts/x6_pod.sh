#!/usr/bin/env bash
# X6 on a fresh pod: the learned stage detector (Jev-R v0) replaces the oracle inside the X5 memory loop.
# Needs /workspace/ckpt/7999 uploaded (task-1 LoRA params+assets) — see scripts/x5_replicate_pod.sh for the upload wait.
# Arms on NEW seeds (default 151–201): memory+oracle (reference), memory+learned, primitive+learned.
# Launch+auto-stop in one step: this script stops the pod itself when done (RUNPOD_API_KEY must be in /workspace/token.sh).
set -u
export DEBIAN_FRONTEND=noninteractive
cd /workspace
[ -d memory_robotics/.git ] || git clone -q https://github.com/Phillips-Ugo/memory_robotics.git
cd memory_robotics && git pull -q origin main
bash scripts/setup_gpu_box.sh > /workspace/setup.log 2>&1 || { echo "[SETUP FAILED]"; tail -20 /workspace/setup.log; exit 1; }
. /workspace/env.sh; . /workspace/token.sh 2>/dev/null
python3 scripts/patch_openpi_config.py --repo-id belu/rma_task1
python3 scripts/patch_rma_adapter_hooks.py
echo "[SETUP OK $(date -u +%H:%M)]"
for i in $(seq 1 240); do [ -f /workspace/ckpt/UPLOAD_DONE ] && break; sleep 15; done
[ -f /workspace/ckpt/UPLOAD_DONE ] || { echo "[NO CHECKPOINT UPLOAD]"; exit 1; }
# detector deps in the harness venv (torch is there; dinov2 via torch.hub needs network once)
vendor/rma-venv/bin/python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vits14', verbose=False); print('dinov2 cached')"
cd /workspace/memory_robotics/vendor/openpi
(nohup uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_rma_lora --policy.dir=/workspace/ckpt/7999 > /workspace/server.log 2>&1 < /dev/null &)
for i in $(seq 1 180); do grep -q "server listening" /workspace/server.log && break; sleep 5; done
grep -q "server listening" /workspace/server.log || { echo "[SERVER FAILED]"; tail -20 /workspace/server.log; exit 1; }
echo "[SERVER OK $(date -u +%H:%M)]"
cd /workspace/memory_robotics/vendor/RoboMemArena/evaluation_benchmark
SEED=${SEED:-151}; TRIALS=${TRIALS:-51}
run_arm () {  # name, kwargs-json
  OUT=/workspace/memory_robotics/outputs/x6_$1; rm -rf $OUT /workspace/x6_$1.db /workspace/x6_$1.jsonl
  echo "[ARM $1 start $(date -u +%H:%M)]"
  X5_LOG=/workspace/x6_$1.jsonl MUJOCO_GL=egl PYTHONUNBUFFERED=1 ../../rma-venv/bin/python scripts/eval_task1_only.py \
    --adapter-spec /workspace/memory_robotics/scripts/04_rma_jevr_adapter.py:build_adapter \
    --adapter-kwargs "$2" --num-trials-per-task $TRIALS --seed $SEED --video-out-path $OUT 2>&1 | grep --line-buffered "Episode\|Final result\|Traceback\|Error"
  echo "[ARM $1 exit ${PIPESTATUS[0]} $(date -u +%H:%M)]"
}
run_arm memory_oracle  "{\"mode\": \"memory\", \"scope\": \"episode\", \"detector\": \"oracle\",  \"db\": \"/workspace/x6_memory_oracle.db\"}"
run_arm memory_learned "{\"mode\": \"memory\", \"scope\": \"episode\", \"detector\": \"learned\", \"db\": \"/workspace/x6_memory_learned.db\"}"
run_arm primitive_learned "{\"mode\": \"primitive\", \"detector\": \"learned\", \"db\": \"/workspace/x6_primitive_learned.db\"}"
echo "[X6 DONE $(date -u +%H:%M)]"
# stop the pod from inside via the API (runpodctl has no config in a detached shell)
vendor_py=/workspace/memory_robotics/vendor/openpi/.venv/bin/python
cd /workspace/memory_robotics/vendor/openpi && uv pip install -q runpod >/dev/null 2>&1
uv run python -c "import runpod,os; runpod.api_key=os.environ['RUNPOD_API_KEY']; runpod.stop_pod(os.environ['RUNPOD_POD_ID']); print('[POD STOP REQUESTED]')" || echo "[POD STOP FAILED — stop it from the Mac]"
