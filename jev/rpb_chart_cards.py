"""Card-style benchmark figure: one card per question family, grayscale bars, ours in black, values in monospace.
    uv run python -m jev.rpb_chart_cards -> docs/figures/rpb_gm100_cards.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

RPB = Path("data/RoboProcessBench"); OUT = Path("docs/figures")
FAM = [("T1", "Phase recognition", "Which phase is the robot in?"), ("T5", "Progress estimation", "How far along is this step?"),
       ("T2", "Contact detection", "Is the gripper in contact?")]
INK, GRAY, TRACK, SUB, PAGE, CARD, EDGE = "#111111", "#b5b5b5", "#efefed", "#6b6b6b", "#f4f4f2", "#ffffff", "#e3e3e0"


def main():
    ours = json.load(open("outputs/rpb/results.json"))["per_task"]; qwen = json.load(open(RPB / "qwen_lora_per_task.json"))
    sonnet = json.load(open("outputs/rpb_vlm/results_claude-sonnet-5.json"))["per_task"]; haiku = json.load(open("outputs/rpb_vlm/results_claude-haiku-4-5.json"))["per_task"]
    models = [("ipau\n23M", lambda t: ours[t]["ours"], True), ("Qwen2.5-VL\n7B, FT", lambda t: qwen[t]["gm100"][0] / qwen[t]["gm100"][1], False),
              ("Sonnet 5\nzero-shot", lambda t: sonnet[t]["acc"], False), ("Haiku 4.5\nzero-shot", lambda t: haiku[t]["acc"], False)]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    fig = plt.figure(figsize=(16, 8), dpi=120); fig.patch.set_facecolor(PAGE)
    n = len(FAM); cw = 1.0 / n
    for ci, (t, title, sub) in enumerate(FAM):
        ax = fig.add_axes([ci * cw + 0.012, 0.04, cw - 0.024, 0.92]); ax.set_facecolor(CARD); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0,rounding_size=0.03", linewidth=1.2, edgecolor=EDGE, facecolor=CARD, transform=ax.transAxes, clip_on=False))
        ax.text(0.08, 0.88, title, fontsize=20, color=INK, va="center", ha="left")
        ax.text(0.08, 0.815, sub, fontsize=13, color=SUB, va="center", ha="left")
        vals = sorted(((name, f(t), hi) for name, f, hi in models), key=lambda x: -x[1])
        top = max(v for _, v, _ in vals)
        y0, h = 0.19, 0.50; k = len(vals); bw = 0.17; gap = (0.84 - k * bw) / (k - 1)
        for i, (name, v, hi) in enumerate(vals):
            x = 0.08 + i * (bw + gap)
            ax.add_patch(Rectangle((x, y0), bw, h * (top / top), facecolor=TRACK, edgecolor="none"))  # light track = best score
            ax.add_patch(Rectangle((x, y0), bw, h * (v / top), facecolor=INK if hi else GRAY, edgecolor="none"))
            ax.text(x + bw / 2, y0 + h + 0.035, f"{100*v:.1f}%", ha="center", va="bottom", fontsize=16, color=INK if hi else SUB, family="monospace", fontweight="bold" if hi else "normal")
            ax.text(x + bw / 2, y0 - 0.03, name, ha="center", va="top", fontsize=12.5, color=INK if hi else SUB, fontweight="bold" if hi else "normal", linespacing=1.25)
    fig.text(0.012, 0.012, "RoboProcessBench, GM-100 split, 2,643 held-out items. Chance: 25% phase, 33% progress, 50% contact.", fontsize=11, color=SUB, va="bottom")
    out = OUT / "rpb_gm100_cards.png"; fig.savefig(out, dpi=120, facecolor=PAGE); print("wrote", out)


if __name__ == "__main__":
    main()
