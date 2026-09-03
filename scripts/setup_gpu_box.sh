#!/usr/bin/env bash
# Set up a rented GPU box (Ubuntu 22.04, single RTX 4090-class) for Phase 1 M2:
# pi-0.5 inference through the RoboMemArena harness.
#
# Usage: clone this repo on the box, then run  bash scripts/setup_gpu_box.sh
# Idempotent — safe to rerun. Everything lands in gitignored vendor/.
#
# Architecture (two venvs, one socket — they never share dependencies):
#   [openpi venv: JAX + pi05_libero]  <-- ws://localhost:8000 -->  [rma-venv: robosuite/mujoco harness]
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

# ---- system deps (rented boxes are usually root; add sudo if not) ----------
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl ffmpeg libegl1 libgl1 libosmesa6 tmux > /dev/null

# ---- uv --------------------------------------------------------------------
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# openpi caches the checkpoint (several GB) under ~/.cache by default — on RunPod that is
# the small container disk. Keep it on the /workspace volume, for this shell and future ones.
export OPENPI_DATA_HOME=/workspace/openpi_cache
grep -q OPENPI_DATA_HOME ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'RC'
export PATH="$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME=/workspace/openpi_cache
RC

# ---- A) openpi server env (official repo; serves pi05_libero) --------------
[ -d vendor/openpi ] || git clone https://github.com/Physical-Intelligence/openpi vendor/openpi
(cd vendor/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync)

# ---- B) RoboMemArena harness env (same recipe as setup_rma_env.sh, Linux) --
[ -d vendor/RoboMemArena ] || git clone --depth 1 https://github.com/OpenHelix-Team/RoboMemArena vendor/RoboMemArena

uv venv vendor/rma-venv --python 3.11
# mujoco MUST be 2.3.7: robosuite 1.4.x breaks on mujoco 3.x joint indexing
uv pip install --python vendor/rma-venv/bin/python \
    robosuite==1.4.1 mujoco==2.3.7 bddl easydict "gym==0.26.2" future termcolor \
    pyyaml opencv-python "imageio[ffmpeg]" tqdm torch numpy matplotlib \
    websockets msgpack typing_extensions  # last three: openpi-client deps for the pi05 adapter

# LIBERO asks about a dataset folder on first import; accept defaults non-interactively
printf 'N\n' | vendor/rma-venv/bin/python -c "import sys; sys.path.insert(0, 'vendor/RoboMemArena/evaluation_benchmark/libero_fork'); import libero" || true

cat <<EOF

Done. M2 runbook (use tmux so nothing dies with your ssh session):

  # (new shells: source ~/.bashrc first so uv and OPENPI_DATA_HOME are set)
  # step 0 — verified checkpoint download (openpi's own downloader corrupts on restart):
  cd $REPO_ROOT/vendor/openpi && uv run python $REPO_ROOT/scripts/download_pi05.py

  # pane 1 — policy server (finds the checkpoint in OPENPI_DATA_HOME, no re-download):
  cd $REPO_ROOT/vendor/openpi && nohup uv run scripts/serve_policy.py --env LIBERO > $REPO_ROOT/server.log 2>&1 &
  tail -f $REPO_ROOT/server.log      # until it reports listening on :8000

  # pane 2 — smoke test (1 trial, then scale up):
  cd $REPO_ROOT/vendor/RoboMemArena/evaluation_benchmark
  MUJOCO_GL=egl PYTHONUNBUFFERED=1 ../../rma-venv/bin/python scripts/eval_task1_only.py \\
      --adapter-spec $REPO_ROOT/scripts/02_rma_pi05_adapter.py:build_adapter \\
      --num-trials-per-task 1 \\
      --video-out-path $REPO_ROOT/outputs/rma_pi05_smoke

  # official protocol for the real number: 51 trials/task, seed 50, max 2500 steps
  # (paper: pi-0.5 20.0% TSR / 42.8% CSR on task 1's category; stock checkpoint = 0/51)

M2b runbook — LoRA fine-tune on task 1 (needs >=40 GB VRAM, >=120 GB volume; see docs/phase1-plan.md):
  cd $REPO_ROOT/vendor/openpi
  uv run python $REPO_ROOT/scripts/download_rma_data.py --tasks 1 --out /workspace/rma_data
  uv run python $REPO_ROOT/scripts/convert_rma_to_lerobot.py --data-root /workspace/rma_data --repo-id belu/rma_task1
  (cd $REPO_ROOT && python3 scripts/patch_openpi_config.py --repo-id belu/rma_task1)
  uv run scripts/compute_norm_stats.py --config-name pi05_rma_lora
  # smoke: 20 steps, then the real run in the background
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_rma_lora --exp-name smoke --overwrite --num-train-steps 20
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 nohup uv run scripts/train.py pi05_rma_lora --exp-name t1 --overwrite > $REPO_ROOT/train_t1.log 2>&1 &
  # then serve the fine-tuned checkpoint instead of --env LIBERO:
  uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_rma_lora --policy.dir=checkpoints/pi05_rma_lora/t1/8000

If MUJOCO_GL=egl fails on the box's driver stack, fall back to MUJOCO_GL=osmesa
(slower, CPU rendering — fine for a smoke test, not for 51-trial runs).
EOF
