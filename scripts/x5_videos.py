"""Comparison videos for M2b/X5 from the harness mp4s (10 fps, 256x256 agent view).

    uv run python scripts/x5_videos.py
Writes docs/figures/x5_seed55_three_way.mp4, x5_fixed_failure_gallery.mp4, x5_fixed_success_vs_failure.mp4
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FIG = Path("docs/figures")
F = Path("outputs/rma_pi05_ft_task1")
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 15)
    FONT_S = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
except OSError:
    FONT = FONT_S = ImageFont.load_default()


def frames(path, every=1):
    return [f for i, f in enumerate(iio.imiter(path)) if i % every == 0]


def label(frame, title, sub, ok=None, t=None, stages=None):
    im = Image.fromarray(frame).resize((384, 384), Image.BILINEAR)
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, 384, 44], fill=(20, 24, 30, 200))
    d.text((8, 4), title, font=FONT, fill=(255, 255, 255))
    d.text((8, 24), sub, font=FONT_S, fill=(200, 210, 220))
    if stages is not None and t is not None:
        y = 384 - 40
        d.rectangle([0, y, 384, 384], fill=(20, 24, 30, 190))
        s1 = stages.get("01_Place_Cookies_Basket"); s2 = stages.get("02_Place_Tomato_Basket")
        c1 = (110, 220, 150) if s1 is not None and t >= s1 else (150, 150, 150)
        c2 = (110, 220, 150) if s2 is not None and t >= s2 else (150, 150, 150)
        d.text((8, y + 4), ("DONE " if c1[1] > 200 else "...  ") + "stage 1: cookies → basket", font=FONT_S, fill=c1)
        d.text((8, y + 21), ("DONE " if c2[1] > 200 else "...  ") + "stage 2: tomato sauce → basket", font=FONT_S, fill=c2)
        d.text((300, y + 12), f"t={t}", font=FONT_S, fill=(200, 210, 220))
    if ok is not None:
        d.rectangle([384 - 96, 4, 384 - 6, 24], fill=(40, 140, 80, 230) if ok else (170, 50, 40, 230))
        d.text((384 - 90, 6), "SUCCESS" if ok else "FAILED", font=FONT_S, fill=(255, 255, 255))
    return np.asarray(im)


def tile(clips, cols, speed, titles, subs, oks, stage_list, out, fps=10):
    n = max(len(c) for c in clips)
    rows = (len(clips) + cols - 1) // cols
    writer = iio.imopen(out, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=fps)
    for i in range(0, n, speed):
        panels = []
        for c, ti, su, ok, st in zip(clips, titles, subs, oks, stage_list):
            j = min(i, len(c) - 1)
            done = j == len(c) - 1
            panels.append(label(c[j], ti, su, ok if done else None, t=j, stages=st))
        while len(panels) < rows * cols:
            panels.append(np.zeros_like(panels[0]))
        grid = np.vstack([np.hstack(panels[r * cols:(r + 1) * cols]) for r in range(rows)])
        writer.write_frame(grid)
    writer.close()
    print("wrote", out, Path(out).stat().st_size // 1024, "KB")


def ep(path, seed):
    return [e for e in json.load(open(path))["episodes"] if e["seed"] == seed][0]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    # 1) same seed, three prompt strategies
    fx, mm, pr = ep(F / "results.json", 55), ep("outputs/x5_memory/results.json", 55), ep("outputs/x5_primitive/results.json", 55)
    tile([frames(F / "task1_success_ep5_seed55.mp4"), frames("outputs/x5_frames/memory_seed55.mp4"), frames("outputs/x5_frames/primitive_seed55.mp4")],
         3, 3, ["Fixed full-task prompt", "Memory, per-stage switch", "Primitive prompts (oracle)"],
         ["no memory · 15/51 overall", "switches only stage 2 · 10/51", "trained primitives every stage · 30/51"],
         [fx["TSR"] > 0, mm["TSR"] > 0, pr["TSR"] > 0], [fx["stage_steps"], mm["stage_steps"], pr["stage_steps"]],
         FIG / "x5_seed55_three_way.mp4")
    # 2) fixed-prompt failure gallery: stage 1 done, stage 2 never
    seeds = [50, 53, 54, 56]
    eps = [ep(F / "results.json", s) for s in seeds]
    tile([frames(F / f"task1_failure_ep{s-50}_seed{s}.mp4") for s in seeds], 2, 4,
         [f"seed {s} · fixed prompt" for s in seeds], ["stage 1 done, then re-grasps or stalls"] * 4,
         [False] * 4, [e["stage_steps"] for e in eps], FIG / "x5_fixed_failure_gallery.mp4")
    # 3) fixed prompt: success vs failure
    a, b = ep(F / "results.json", 51), ep(F / "results.json", 50)
    tile([frames(F / "task1_success_ep1_seed51.mp4"), frames(F / "task1_failure_ep0_seed50.mp4")], 2, 2,
         ["seed 51 · fixed prompt", "seed 50 · fixed prompt"], ["both stages", "stage 1 only"],
         [True, False], [a["stage_steps"], b["stage_steps"]], FIG / "x5_fixed_success_vs_failure.mp4")


if __name__ == "__main__":
    main()
