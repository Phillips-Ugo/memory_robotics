"""Concise, publishable version of the RoboProcessBench comparison: four question families, three models, chance tick.
    uv run python -m jev.rpb_chart_concise -> docs/figures/rpb_gm100_concise.png (+ _dark)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RPB = Path("data/RoboProcessBench"); OUT = Path("docs/figures")
FAM = [("T1", "Which phase\nis the robot in?"), ("T5", "How far along\nis this step?"), ("T2", "Is the gripper\nin contact?"), ("T6", "Is the arm\nmoving or still?")]
PAL = {"light": {"ours": "#2a78d6", "qwen": "#eb6834", "sonnet": "#eda100", "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
       "dark": {"ours": "#3987e5", "qwen": "#d95926", "sonnet": "#c98500", "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"}}


def main():
    ours = json.load(open("outputs/rpb/results.json"))["per_task"]; qwen = json.load(open(RPB / "qwen_lora_per_task.json"))
    sonnet = json.load(open("outputs/rpb_vlm/results_claude-sonnet-5.json"))["per_task"]
    dist = {l.split(",")[0]: float(l.strip().split(",")[4]) for l in open(RPB / "metadata/task_distribution.csv") if l.startswith("T")}
    series = [("ours", "Ours · 0.7M-param judge on a frozen 22M encoder", {t: ours[t]["ours"] for t in ours}),
              ("qwen", "Qwen2.5-VL-7B · fine-tuned on this benchmark", {t: qwen[t]["gm100"][0] / qwen[t]["gm100"][1] for t in qwen if qwen[t]["gm100"]}),
              ("sonnet", "Claude Sonnet 5 · zero-shot", {t: sonnet[t]["acc"] for t in sonnet})]
    for mode in ("light", "dark"):
        c = PAL[mode]
        plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
        fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160); fig.patch.set_facecolor(c["surface"]); ax.set_facecolor(c["surface"])
        n = len(series); w = 0.72 / n
        for si, (key, _, vals) in enumerate(series):
            for xi, (t, _) in enumerate(FAM):
                v = vals[t]; x = xi - 0.36 + w * (si + 0.5)
                ax.bar(x, 100 * v, width=w * 0.9, color=c[key], linewidth=0, zorder=3)
                ax.text(x, 100 * max(v, dist[t]) + 1.5, f"{100*v:.0f}", ha="center", va="bottom", fontsize=12, color=c["ink2"], zorder=6)
        for xi, (t, _) in enumerate(FAM):
            ax.plot([xi - 0.38, xi + 0.38], [100 * dist[t]] * 2, color=c["ink"], lw=1.1, ls=(0, (2, 2)), zorder=5)
        ax.set_xticks(range(len(FAM))); ax.set_xticklabels([f for _, f in FAM], fontsize=12.5, color=c["ink"])
        ax.set_ylim(0, 92); ax.set_yticks([0, 25, 50, 75]); ax.set_yticklabels(["0", "25", "50", "75%"], fontsize=10.5, color=c["muted"])
        ax.yaxis.grid(True, color=c["grid"], lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(c["axis"]); ax.tick_params(axis="both", length=0)
        ax.set_title("Robot process questions, answered by a model 300× smaller", loc="left", fontsize=18, color=c["ink"], fontweight="semibold", pad=40)
        ax.text(0, 1.045, "RoboProcessBench, GM-100 split · 2,643 held-out items · accuracy per question family", transform=ax.transAxes, fontsize=11.5, color=c["ink2"], va="bottom")
        handles = [Patch(color=c[k], label=l) for k, l, _ in series] + [Line2D([0], [0], color=c["ink"], lw=1.1, ls=(0, (2, 2)), label="Random guessing")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=11, labelcolor=c["ink"], ncol=2, columnspacing=1.6, handlelength=1.4)
        ax.text(0, -0.16, "Ours: one forward pass, no text generation, ~18 ms per answer on a laptop GPU.   Qwen: 7B parameters, generates tokens.   Sonnet: frontier API, seconds per answer.",
                transform=ax.transAxes, fontsize=10, color=c["muted"], va="top")
        fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.2)
        out = OUT / f"rpb_gm100_concise{'_dark' if mode == 'dark' else ''}.png"; fig.savefig(out, dpi=160, facecolor=c["surface"]); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
