"""Download selected RoboMemArena tasks' subtask_data from Hugging Face (resumable).

The full dataset is ~1 TB across 26 tasks; task 1 alone is ~27 GB / 400 files.
    cd vendor/openpi && uv run python /workspace/memory_robotics/scripts/download_rma_data.py \
        --tasks 1 --out /workspace/rma_data
Re-run to resume. Note the README says the June-20 refresh landed on ModelScope
first with the HF mirror "coming soon" — check huggingface.co/datasets/RoboMemArenaBenchmark/RoboMemArena
for the update before training on tasks 1-3 (their scenes changed: two identical baskets).
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

REPO = "RoboMemArenaBenchmark/RoboMemArena"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="1", help="comma-separated task ids, e.g. 1,2,3")
    ap.add_argument("--out", default="/workspace/rma_data")
    ap.add_argument("--full-trajectory", action="store_true", help="also fetch full_trajectory/ files")
    args = ap.parse_args()

    patterns = []
    for t in args.tasks.split(","):
        t = t.strip()
        patterns.append(f"*/{t}_*_dataset/subtask_data/*.hdf5")
        if args.full_trajectory:
            patterns.append(f"*/{t}_*_dataset/full_trajectory/*")
    print(f"downloading {patterns} from {REPO} -> {args.out}", flush=True)
    path = snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=args.out,
        allow_patterns=patterns,
        max_workers=8,
    )
    print(f"done -> {path}", flush=True)


if __name__ == "__main__":
    main()
