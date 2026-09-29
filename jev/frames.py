"""Build the frame-level "what stage are we in?" dataset from RoboMemArena harness recordings.

Every harness episode ships a 10 fps agent-view mp4 (one frame per env step) and results.json with the
step at which each sequential stage check first passed. That is free supervision for the questions a
fast decision model should answer:

    stage_done_1  : has stage 1 (cookies in basket) been completed?         (boolean)
    stage_done_2  : has stage 2 (tomato sauce in basket) been completed?    (boolean)
    current_stage : 0 = working on stage 1, 1 = working on stage 2, 2 = task complete   (choice)

    uv run python -m jev.frames --every 10 --out outputs/jev_frames   # ~40k frames from 204 episodes

Writes <out>/index.jsonl (one row per sampled frame: source dir, episode, seed, step, labels, video path)
and, with --features <backbone>, <out>/features_<backbone>.npy aligned to index.jsonl.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np

SOURCES = {  # dir -> (seed set, prompting strategy) — keep for split design
    "outputs/rma_pi05_ft_task1": ("A", "fixed"),
    "outputs/x5r_fixed": ("B", "fixed"),
    "outputs/x5r_primitive": ("B", "primitive"),
    "outputs/x5r_memory": ("B", "memory"),
}
S1, S2 = "01_Place_Cookies_Basket", "02_Place_Tomato_Basket"


def episode_rows(src: str, every: int) -> list[dict]:
    r = json.loads(Path(src, "results.json").read_text())
    rows = []
    for e in r["episodes"]:
        vids = [v for v in glob.glob(f"{src}/task1_*_ep{e['ep']}_seed{e['seed']}.mp4") if "wrist" not in v]
        if not vids:
            continue
        n = e["total_steps"]
        s1 = e["stage_steps"].get(S1); s2 = e["stage_steps"].get(S2)
        for t in range(0, n, every):
            d1 = s1 is not None and t >= s1
            d2 = s2 is not None and t >= s2
            rows.append({"src": src, "seedset": SOURCES[src][0], "strategy": SOURCES[src][1], "ep": e["ep"], "seed": e["seed"],
                         "step": t, "n": n, "stage_done_1": int(d1), "stage_done_2": int(d2),
                         "current_stage": 2 if d2 else (1 if d1 else 0), "video": vids[0]})
    return rows


def load_frames(rows: list[dict]) -> np.ndarray:
    """Read the sampled frames video by video (frames are 1:1 with env steps)."""
    out = np.zeros((len(rows), 256, 256, 3), dtype=np.uint8)
    by_video: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_video.setdefault(r["video"], []).append(i)
    for v, idxs in by_video.items():
        want = {rows[i]["step"]: i for i in idxs}
        for t, frame in enumerate(iio.imiter(v)):
            if t in want:
                out[want[t]] = frame
            if t > max(want):
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--out", default="outputs/jev_frames")
    ap.add_argument("--features", default="", help="dinov2_vits14 | dinov2_vitb14 | '' (index only)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--view", default="agent", choices=["agent", "wrist"], help="which camera's mp4 to featurize")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = [r for src in SOURCES for r in episode_rows(src, args.every)]
    if args.view == "wrist":
        for r in rows:
            r["video"] = r["video"].replace(".mp4", "_wrist.mp4")
    else:
        (out / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} frames from {len({(r['src'], r['ep']) for r in rows})} episodes -> {out/'index.jsonl'}")
    if not args.features:
        return
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", args.features, verbose=False).eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    feats = []
    frames = load_frames(rows)
    with torch.no_grad():
        for i in range(0, len(frames), args.batch):
            x = torch.from_numpy(frames[i:i + args.batch]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            feats.append(model((x - mean) / std).float().cpu().numpy())
            if (i // args.batch) % 50 == 0:
                print(f"  {i}/{len(frames)}", flush=True)
    tag = args.features + ("_wrist" if args.view == "wrist" else "")
    F = np.concatenate(feats); np.save(out / f"features_{tag}.npy", F)
    print("features", F.shape, "->", out / f"features_{tag}.npy")


if __name__ == "__main__":
    main()
