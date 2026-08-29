#!/usr/bin/env bash
# Recreate the RoboMemArena eval environment from scratch (macOS).
# Everything lands in gitignored vendor/ — safe to delete and rerun.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d vendor/RoboMemArena ] || git clone --depth 1 https://github.com/OpenHelix-Team/RoboMemArena vendor/RoboMemArena

uv venv vendor/rma-venv --python 3.11
# mujoco MUST be 2.3.7: robosuite 1.4.x breaks on mujoco 3.x joint indexing
uv pip install --python vendor/rma-venv/bin/python \
    robosuite==1.4.1 mujoco==2.3.7 bddl easydict "gym==0.26.2" future termcolor \
    pyyaml opencv-python "imageio[ffmpeg]" tqdm torch numpy matplotlib

# mujoco 2.3.7 hardcodes a pre-Sequoia OpenGL framework path; point it at the real one
sed -i '' "s|'/System/Library/OpenGL.framework/OpenGL'|'/System/Library/Frameworks/OpenGL.framework/OpenGL'|" \
    vendor/rma-venv/lib/python3.11/site-packages/mujoco/cgl/cgl.py

# LIBERO asks about a dataset folder on first import; accept defaults non-interactively
printf 'N\n' | vendor/rma-venv/bin/python -c "import sys; sys.path.insert(0, 'vendor/RoboMemArena/evaluation_benchmark/libero_fork'); import libero" || true

cat <<'EOF'
Done. Smoke test:
  cd vendor/RoboMemArena/evaluation_benchmark
  MUJOCO_GL=glfw ../../rma-venv/bin/python scripts/eval_task1_only.py \
    --adapter-spec "$PWD/../../../scripts/01_rma_dummy_adapter.py:build_adapter" \
    --num-trials-per-task 1 --max-steps 60 \
    --video-out-path ../../../outputs/rma_smoke
(MUJOCO_GL=glfw overrides the harness's Linux-only egl default on macOS)
EOF
