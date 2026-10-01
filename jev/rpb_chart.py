"""Publishable chart: RoboProcessBench (GM-100 source) accuracy per task family, ours vs fine-tuned and zero-shot VLMs.

    uv run python -m jev.rpb_chart    -> docs/figures/rpb_gm100_compare.png (+ _dark.png)
Palette: validated categorical slots (dataviz reference palette), chance marked as a tick, direct labels on bars.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

RPB = Path("data/RoboProcessBench"); OUT = Path("docs/figures")
FAMILIES = [("T1", "Phase\nrecognition"), ("T5", "Progress\n(early/mid/late)"), ("T2", "Contact\ndetection"), ("T6", "Motion\nstate"),
            ("T3", "Motion\ndirection"), ("T4", "Bimanual\ncoordination"), ("T9", "Temporal\npriority"), ("T8", "Temporal\nordering")]
PAL = {"light": {"ours": "#2a78d6", "qwen": "#eb6834", "haiku": "#1baf7a", "sonnet": "#eda100", "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
                 "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
       "dark": {"ours": "#3987e5", "qwen": "#d95926", "haiku": "#199e70", "sonnet": "#c98500", "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
                "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"}}


def load():
    ours = json.load(open("outputs/rpb/results.json"))["per_task"]
    qwen = json.load(open(RPB / "qwen_lora_per_task.json"))
    dist = {l.split(",")[0]: l.strip().split(",") for l in open(RPB / "metadata/task_distribution.csv") if l.startswith("T")}
    series = [("ours", "Ours: 22M frozen encoder + 0.7M head (no generation)", {t: ours[t]["ours"] for t in ours}),
              ("qwen", "Qwen2.5-VL-7B, fine-tuned on the benchmark", {t: qwen[t]["gm100"][0] / qwen[t]["gm100"][1] for t in qwen if qwen[t]["gm100"]})]
    for key, label, f in (("haiku", "Claude Haiku 4.5, zero-shot", "outputs/rpb_vlm/results_claude-haiku-4-5.json"),
                          ("sonnet", "Claude Sonnet 5, zero-shot", "outputs/rpb_vlm/results_claude-sonnet-5.json")):
        if Path(f).exists():
            r = json.load(open(f))["per_task"]; series.append((key, label, {t: r[t]["acc"] for t in r}))
    chance = {t: float(dist[t][4]) for t in dist}
    return series, chance


def draw(mode: str, series, chance, latency: dict) -> Path:
    c = PAL[mode]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120); fig.patch.set_facecolor(c["surface"]); ax.set_facecolor(c["surface"])
    n = len(series); w = 0.8 / n; xs = range(len(FAMILIES))
    for si, (key, label, vals) in enumerate(series):
        for xi, (t, _) in enumerate(FAMILIES):
            v = vals.get(t)
            if v is None:
                continue
            x = xi - 0.4 + w * (si + 0.5)
            ax.bar(x, 100 * v, width=w * 0.92, color=c[key], linewidth=0, zorder=3)
            ax.text(x, 100 * max(v, chance[t]) + 1.4, f"{100*v:.0f}", ha="center", va="bottom", fontsize=10, color=c["ink2"], zorder=6)
    for xi, (t, _) in enumerate(FAMILIES):  # chance tick spanning the group
        ax.plot([xi - 0.42, xi + 0.42], [100 * chance[t]] * 2, color=c["ink"], lw=1.2, ls=(0, (2, 2)), zorder=5)
    handles = [Patch(color=c[k], label=l) for k, l, _ in series] + [Line2D([0], [0], color=c["ink"], lw=1.2, ls=(0, (2, 2)), label="Random guessing")]
    ax.set_xticks(list(xs)); ax.set_xticklabels([f for _, f in FAMILIES], fontsize=11, color=c["ink2"])
    ax.set_ylim(0, 108); ax.set_yticks(range(0, 101, 25)); ax.set_yticklabels([f"{v}%" for v in range(0, 101, 25)], fontsize=10, color=c["muted"])
    ax.yaxis.grid(True, color=c["grid"], lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(c["axis"]); ax.tick_params(axis="both", length=0)
    ax.set_title("Robot process understanding, RoboProcessBench (GM-100 split), accuracy by question family", loc="left", fontsize=17, color=c["ink"], pad=44, fontweight="semibold")
    ax.text(0, 1.03, "Same training and evaluation episodes for every model. The fine-tuned baseline is the benchmark authors' released model; zero-shot models use the benchmark's own prompt.",
            transform=ax.transAxes, fontsize=11, color=c["ink2"], va="bottom")
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10.5, labelcolor=c["ink"], ncol=1, bbox_to_anchor=(1.0, 1.0))
    # latency / size strip
    txt = "   ".join(f"{k}: {v}" for k, v in latency.items())
    ax.text(0, -0.17, "Cost per answer:  " + txt, transform=ax.transAxes, fontsize=10.5, color=c["ink2"], va="top")
    ax.text(0, -0.215, "Phase and progress are the two questions a robot controller asks every step. Ours answers them in one forward pass, with calibrated probabilities, no text generation.",
            transform=ax.transAxes, fontsize=10.5, color=c["muted"], va="top")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.2)
    out = OUT / f"rpb_gm100_compare{'_dark' if mode == 'dark' else ''}.png"; fig.savefig(out, dpi=120, facecolor=c["surface"]); plt.close(fig)
    return out


def main():
    series, chance = load()
    ours = json.load(open("outputs/rpb/results.json"))
    latency = {"Ours": f"{ours['scorer_params']/1e6:.1f}M trainable + 22M frozen params, ~{ours['scorer_ms'] + 7*2.5:.0f} ms on a laptop GPU",
               "Qwen2.5-VL-7B": "7B params, generates tokens", "Claude (zero-shot)": "frontier API, seconds per answer"}
    for mode in ("light", "dark"):
        print("wrote", draw(mode, series, chance, latency))


if __name__ == "__main__":
    main()
