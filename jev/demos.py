"""Stage labels + DINOv2 features from RoboMemArena's per-subtask demo HDF5 files (no rollouts needed).

A task's demo for one seed is k files `<primitive>_<k>_seed<S>_task<T>.hdf5` (one demo each). For the basket
tasks (1-3) the 4 primitives are pick_A_0, place_A_1, pick_B_2, place_B_3 and the harness stages are
"A in basket" (stage 1) and "B in basket" (stage 2). Labels per frame:
    stage_done_1 = 1 for files 2,3 and the last `tail` frames of file 1 (object already in the basket, about to release)
    stage_done_2 = 1 for the last `tail` frames of file 3
    current_stage = 0 / 1 / 2 as in jev/frames.py;  task = T

    uv run python -m jev.demos --root outputs/rma_demos --every 2 --tail 10 --out outputs/jev_demos
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import h5py
import numpy as np

FILE_RE = re.compile(r"(?P<prim>.+)_(?P<k>\d)_seed(?P<seed>\d+)_task(?P<task>\d+)\.hdf5$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/rma_demos")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--tail", type=int, default=10)
    ap.add_argument("--out", default="outputs/jev_demos")
    ap.add_argument("--features", default="dinov2_vits14")
    ap.add_argument("--label", default="release", choices=["release", "tail"],
                    help="release: a place stage is done from the first frame the demo commands the gripper open (actions[:,6] < 0); tail: last --tail frames")
    ap.add_argument("--index-only", action="store_true", help="rewrite index.jsonl without re-extracting features")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{args.root}/**/subtask_data/*.hdf5", recursive=True))
    seqs: dict[tuple[int, int], dict[int, str]] = {}
    for f in files:
        m = FILE_RE.search(Path(f).name); seqs.setdefault((int(m["task"]), int(m["seed"])), {})[int(m["k"])] = f
    rows, agent, wrist = [], [], []
    for (task, seed), parts in sorted(seqs.items()):
        if sorted(parts) != [0, 1, 2, 3]:
            print("skipping incomplete", task, seed, sorted(parts)); continue
        t_global = 0
        for k in range(4):
            with h5py.File(parts[k]) as h:
                g = h["data"][list(h["data"].keys())[0]]
                acts = g["actions"][()]; n = acts.shape[0]
                rel = int(np.argmax(acts[:, 6] < 0)) if (acts[:, 6] < 0).any() else n  # first open command
                done_from = rel if args.label == "release" else n - args.tail
                idx = list(range(0, n, args.every))
                A = g["obs"]["agentview_rgb"][idx]; W = g["obs"]["eye_in_hand_rgb"][idx]; grip = g["obs"]["gripper_states"][idx]
                for j, i in enumerate(idx):
                    d1 = k >= 2 or (k == 1 and i >= done_from)
                    d2 = k == 3 and i >= done_from
                    rows.append({"task": task, "seed": seed, "k": k, "prim": Path(parts[k]).name.rsplit("_", 3)[0], "i": i, "n": n,
                                 "t": t_global + i, "stage_done_1": int(d1), "stage_done_2": int(d2),
                                 "current_stage": 2 if d2 else (1 if d1 else 0), "gripper": grip[j].tolist()})
                agent.append(A); wrist.append(W); t_global += n
    A = np.concatenate(agent); W = np.concatenate(wrist)
    (out / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} frames from {len(seqs)} (task, seed) demos; stage1-done {sum(r['stage_done_1'] for r in rows)}, stage2-done {sum(r['stage_done_2'] for r in rows)}")
    if args.index_only:
        return
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", args.features, verbose=False).eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    def feats(imgs):
        outp = []
        with torch.no_grad():
            for i in range(0, len(imgs), 64):
                x = torch.from_numpy(imgs[i:i + 64]).to(dev).permute(0, 3, 1, 2).float() / 255.0
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                outp.append(model((x - mean) / std).float().cpu().numpy())
        return np.concatenate(outp)
    np.save(out / f"features_{args.features}.npy", feats(A)); np.save(out / f"features_{args.features}_wrist.npy", feats(W))
    print("features written to", out)


if __name__ == "__main__":
    main()
