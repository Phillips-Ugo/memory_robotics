"""X5 analysis: experience curves of the fine-tuned pi05 under fixed / primitive / memory prompting.

    uv run python scripts/x5_analyze.py            # reads outputs/rma_pi05_ft_task1 (fixed) + outputs/x5_*/results.json
Writes docs/figures/x5_experience_curves.png and prints a table with Wilson CIs, paired-by-seed counts and AUC.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("fixed", "outputs/rma_pi05_ft_task1/results.json", "full-task prompt (no memory)"),
        ("primitive", "outputs/x5_primitive/results.json", "primitive prompts every stage (oracle planner)"),
        ("memory", "outputs/x5_memory/results.json", "memory, stage-scope switch (v1)"),
        ("memory_v3", "outputs/x5_memory_v3/results.json", "memory, episode-scope switch")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def running(xs, w=10):
    return [sum(xs[max(0, i - w + 1):i + 1]) / len(xs[max(0, i - w + 1):i + 1]) for i in range(len(xs))]


def main():
    data = {}
    for name, path, label in ARMS:
        if Path(path).exists():
            r = json.loads(Path(path).read_text())
            data[name] = ({e["seed"]: e for e in r["episodes"]}, label)
    print(f"{'arm':10s} {'TSR':>8s} {'95% CI':>12s} {'CSR':>6s} {'stage2':>7s} {'AUC':>5s}  paired vs fixed (won/lost)")
    fixed = data["fixed"][0]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, (eps, label) in data.items():
        seeds = sorted(eps); succ = [int(eps[s]["TSR"] > 0) for s in seeds]
        k, n = sum(succ), len(succ); lo, hi = wilson(k, n)
        csr = sum(eps[s]["CSR"] for s in seeds) / n
        s2 = sum(eps[s]["stage_done"].get("02_Place_Tomato_Basket", False) for s in seeds)
        auc = sum(succ) / n
        won = sum(1 for s in seeds if eps[s]["TSR"] > 0 and fixed[s]["TSR"] == 0)
        lost = sum(1 for s in seeds if eps[s]["TSR"] == 0 and fixed[s]["TSR"] > 0)
        print(f"{name:10s} {k:3d}/{n:<3d} {100*k/n:4.1f}% {100*lo:5.1f}–{100*hi:4.1f}% {csr:5.1f}% {s2:3d}/{n:<3d} {auc:5.2f}  {won:2d}/{lost:<2d}")
        ax.plot(range(n), running(succ), label=f"{label}  ({k}/{n})", lw=2)
    ax.set_xlabel("episode index (same seeds in every arm)"); ax.set_ylabel("task success, running mean of 10")
    ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("X5 — fine-tuned π₀.₅ on RoboMemArena task 1: what the prompt policy learns across episodes")
    out = Path("docs/figures/x5_experience_curves.png"); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150); print("wrote", out)


if __name__ == "__main__":
    main()
