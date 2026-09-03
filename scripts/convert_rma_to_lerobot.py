"""Convert RoboMemArena subtask HDF5 episodes into a LeRobot dataset for openpi.

Modeled on openpi's examples/libero/convert_libero_data_to_lerobot.py but reads the
HDF5 files directly (no RLDS/TFDS detour). Same features openpi's LIBERO configs
expect: image, wrist_image (256x256x3), state (8 = ee_states 6 + gripper 2),
actions (7), task string. The instruction is derived from the filename exactly as
RoboMemArena's own RLDS builder does ("pick_cookies_0_seed100_task1.hdf5" ->
"pick cookies"), i.e. we train on subtask primitives like the paper's baseline.

Run from the openpi env on the GPU box (lerobot + h5py are installed there):
    cd vendor/openpi && uv run python /workspace/memory_robotics/scripts/convert_rma_to_lerobot.py \
        --data-root /workspace/rma_data --repo-id belu/rma_task1
Then: uv run scripts/compute_norm_stats.py --config-name pi05_rma_lora
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


def instruction_from_filename(path: str) -> str:
    parts = Path(path).stem.split("_")
    words = parts[:-3]  # strip <subtask_order>, seed<...>, task<...>
    if words and words[-1].endswith("."):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dir containing */subtask_data/*.hdf5 (any depth)")
    ap.add_argument("--repo-id", default="belu/rma_task1")
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    files = sorted(glob.glob(os.path.join(args.data_root, "**", "subtask_data", "*.hdf5"), recursive=True))
    if not files:
        raise SystemExit(f"no subtask_data/*.hdf5 under {args.data_root}")
    print(f"{len(files)} hdf5 files under {args.data_root}", flush=True)

    out = HF_LEROBOT_HOME / args.repo_id
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=args.fps,
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    n_ep = n_frames = 0
    for path in files:
        task = instruction_from_filename(path)
        with h5py.File(path, "r") as f:
            for demo in sorted(f["data"].keys()):
                g = f["data"][demo]
                actions = g["actions"][()].astype(np.float32)
                ee = g["obs"]["ee_states"][()].astype(np.float32)
                grip = g["obs"]["gripper_states"][()].astype(np.float32)
                imgs = g["obs"]["agentview_rgb"][()]
                wrist = g["obs"]["eye_in_hand_rgb"][()]
                state = np.concatenate([ee, grip], axis=-1)
                assert state.shape[1] == 8 and actions.shape[1] == 7, (path, state.shape, actions.shape)
                assert imgs.shape[1:] == (256, 256, 3), (path, imgs.shape)
                for i in range(actions.shape[0]):
                    ds.add_frame(
                        {
                            "image": imgs[i],
                            "wrist_image": wrist[i],
                            "state": state[i],
                            "actions": actions[i],
                            "task": task,
                        }
                    )
                ds.save_episode()
                n_ep += 1
                n_frames += actions.shape[0]
                if n_ep % 20 == 0:
                    print(f"  {n_ep} episodes, {n_frames} frames  (last: {task!r}, {actions.shape[0]} steps)", flush=True)
                if args.max_episodes and n_ep >= args.max_episodes:
                    break
        if args.max_episodes and n_ep >= args.max_episodes:
            break

    print(f"done: {n_ep} episodes, {n_frames} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
