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
  # (paper baseline to beat/match: pi-0.5 ~21.5% avg TSR / see paper table for CSR)

If MUJOCO_GL=egl fails on the box's driver stack, fall back to MUJOCO_GL=osmesa
(slower, CPU rendering — fine for a smoke test, not for 51-trial runs).
EOF
