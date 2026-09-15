#!/usr/bin/env bash
# M2b, end to end, unattended: setup -> GPU check -> checkpoint -> data -> LeRobot
# conversion -> LoRA config -> norm stats -> 20-step smoke train.
# Expects /workspace/token.sh with `export HF_TOKEN=hf_...`. Logs to /workspace/m2b.log.
#   curl -sSL https://raw.githubusercontent.com/Phillips-Ugo/memory_robotics/main/scripts/m2b_pipeline.sh -o /workspace/m2b.sh && nohup bash /workspace/m2b.sh > /workspace/m2b.log 2>&1 &
set -e
cd /workspace
[ -d memory_robotics/.git ] || git clone -q https://github.com/Phillips-Ugo/memory_robotics.git
cd memory_robotics && git pull -q
bash scripts/setup_gpu_box.sh > setup.log 2>&1
grep -q HF_TOKEN /workspace/env.sh || cat /workspace/token.sh >> /workspace/env.sh
. /workspace/env.sh
cd vendor/openpi && uv pip install h5py
echo "== JAX devices:"; uv run python -c "import jax; print(jax.devices())"
uv run python /workspace/memory_robotics/scripts/download_pi05.py
uv run python /workspace/memory_robotics/scripts/download_rma_data.py --tasks 1 --out /workspace/rma_data
uv run python /workspace/memory_robotics/scripts/convert_rma_to_lerobot.py --data-root /workspace/rma_data --repo-id belu/rma_task1
(cd /workspace/memory_robotics && python3 scripts/patch_openpi_config.py --repo-id belu/rma_task1)
uv run scripts/compute_norm_stats.py --config-name pi05_rma_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_rma_lora --exp-name smoke --overwrite --num-train-steps 20
echo "[M2B PIPELINE OK THROUGH SMOKE TRAIN]"
