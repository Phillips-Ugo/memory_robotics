"""Experiment X2: does any memory beat none? does structure beat raw retrieval?
And (X4 preview): what happens at a change event?

    uv run python -m bench.run --worlds 30 --episodes 50 --seeds 3 --change-at 25

Outputs outputs/bench_v0/{curves.png, results.json}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import env as _env
from .env import STEP_BUDGET, SkillEnv
from .memory import BASELINES, Memory
from .planner import run_planner
from .world import make_world


def relevant_episodes(history: list, task, props) -> set[int]:
    """Ground truth: past episodes carrying direct evidence about this task's secrets
    under the CURRENT world properties (a jam on its drawer if that drawer is sticky
    now, a drop of its object if that object is heavy now)."""
    rel = set()
    for log in history:
        for e in log.events:
            if task.kind == "put" and props.is_sticky(task.drawer) and e.skill == "open" and e.target == task.drawer and e.outcome == "jam":
                rel.add(log.episode_idx)
            if props.is_heavy(task.obj) and e.skill == "pick" and e.target == task.obj and e.outcome == "drop":
                rel.add(log.episode_idx)
            if task.kind == "fetch" and e.skill == "look_in" and e.outcome == "found" and log.task.obj == task.obj \
                    and e.target == props.location_of(task.obj):
                rel.add(log.episode_idx)
            if task.kind == "put_any" and e.skill == "open" and e.outcome == "ok" and e.steps <= 2 and props.is_fast(e.target):
                rel.add(log.episode_idx)
    return rel


def run_sequence(memory: Memory, world_id: int, seed: int, episodes: int, change_at: int,
                 extra_changes: int = 0, property_types=None, task_kinds=None) -> dict:
    kw = {}
    if property_types:
        kw["property_types"] = tuple(property_types)
    if task_kinds:
        kw["task_kinds"] = tuple(task_kinds)
    world = make_world(world_id, seed, **kw)
    change_eps = {change_at}
    if extra_changes:  # further events at random episodes after the main one
        change_eps |= set(int(x) for x in world.rng.choice(range(change_at + 5, episodes - 3), size=extra_changes, replace=False))
    rows = []
    history = []
    for ep in range(episodes):
        if ep in change_eps:
            world.apply_change_event()
        task = world.sample_task()
        beliefs = memory.recall(task, initial_obs={"task": task.text})
        rel = relevant_episodes(history, task, world.props)
        surf = set(memory.surfaced(task))
        prec = len(rel & surf) / len(surf) if surf else None
        recl = len(rel & surf) / len(rel) if rel else None
        env = SkillEnv(world, task, ep)
        run_planner(env, beliefs)
        memory.observe(env.log)
        history.append(env.log)
        rows.append(
            {
                "ep": ep,
                "success": int(env.log.success),
                "steps": env.log.steps,
                "stale": env.log.stale_actions,
                "failures": sum(e.outcome in ("jam", "drop") for e in env.log.events),
                "ret_precision": prec,
                "ret_recall": recl,
                "wasted_looks": env.log.wasted_looks,
                "kind": task.kind,
            }
        )
    return {"rows": rows, "bytes": memory.bytes_stored()}


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


def summarize(runs: list[dict], episodes: int, change_at: int) -> dict:
    succ = np.array([[r["success"] for r in run["rows"]] for run in runs])  # [n_runs, episodes]
    steps = np.array([[r["steps"] for r in run["rows"]] for run in runs])
    stale = np.array([[r["stale"] for r in run["rows"]] for run in runs])
    n = succ.shape[0]

    pre = succ[:, max(0, change_at - 10) : change_at].mean()  # steady state before change
    post = succ[:, change_at:]
    # episodes-to-recovery: first post-change index whose 5-ep rolling mean is back within 5 pts of pre
    recovery = None
    for i in range(post.shape[1] - 4):
        if post[:, i : i + 5].mean() >= pre - 0.05:
            recovery = i
            break

    def _mean(key):
        vals = [r[key] for run in runs for r in run["rows"] if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "n_runs": n,
        "retrieval_precision": _mean("ret_precision"),
        "retrieval_recall": _mean("ret_recall"),
        "curve_success": succ.mean(0).tolist(),
        "curve_steps": steps.mean(0).tolist(),
        "curve_stale": stale.mean(0).tolist(),
        "auc_success": float(succ.mean()),
        "success_pre_change": float(pre),
        "success_pre_change_ci": wilson(float(pre), n * 10),
        "success_post_change_first10": float(post[:, :10].mean()),
        "episodes_to_recovery": recovery,
        "stale_actions_post_change": float(stale[:, change_at:].sum(1).mean()),
        "bytes_stored_mean": float(np.mean([r["bytes"] for r in runs])),
    }


def plot(results: dict, episodes: int, change_at: int, out: Path, budget_label: str | None = None,
         title: str = "Cross-episode memory benchmark v0") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(episodes)
    for name, s in results.items():
        axes[0].plot(x, s["curve_success"], label=name)
        axes[1].plot(x, s["curve_steps"], label=name)
        axes[2].plot(x, np.cumsum(s["curve_stale"]), label=name)
    for ax, title, ylabel in zip(
        axes,
        ["Experience curve", "Steps per episode", "Cumulative stale actions"],
        ["success rate", f"steps ({budget_label or f'budget {STEP_BUDGET}'})", "stale actions"],
    ):
        ax.axvline(change_at, color="k", ls="--", lw=0.8)
        ax.text(change_at + 0.5, ax.get_ylim()[1] * 0.95, "change event", fontsize=8, va="top")
        ax.set_title(title)
        ax.set_xlabel("episode index in this world")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8)
    n = next(iter(results.values()))["n_runs"]
    fig.suptitle(f"{title} — {n} runs (worlds × seeds) per curve", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--change-at", type=int, default=25)
    ap.add_argument("--slack", type=int, default=None, help="override budget slack (default 11; budget = nominal(kind) + slack)")
    ap.add_argument("--extra-changes", type=int, default=0, help="additional random change events after --change-at")
    ap.add_argument("--properties", default=",".join(("sticky", "heavy", "location", "fast")),
                    help="property types in the world (v0 was sticky,heavy)")
    ap.add_argument("--kinds", default="put,put_any,fetch", help="task kinds (v0 was put)")
    ap.add_argument("--out", default="outputs/bench_v0")
    args = ap.parse_args()
    if args.slack is not None:
        _env.SLACK = args.slack

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, factory in BASELINES.items():
        runs = [
            run_sequence(factory(), w, s, args.episodes, args.change_at, args.extra_changes,
                         args.properties.split(","), args.kinds.split(","))
            for s in range(args.seeds)
            for w in range(args.worlds)
        ]
        results[name] = summarize(runs, args.episodes, args.change_at)
        r = results[name]
        lo, hi = r["success_pre_change_ci"]
        print(
            f"{name:14s} AUC={r['auc_success']:.3f}  pre-change={r['success_pre_change']:.2f} "
            f"[{lo:.2f},{hi:.2f}]  post-change(first10)={r['success_post_change_first10']:.2f}  "
            f"recovery={r['episodes_to_recovery']}  stale(post)={r['stale_actions_post_change']:.1f}  "
            f"bytes={r['bytes_stored_mean']:.0f}  "
            + (f"ret P/R={r['retrieval_precision']:.2f}/{r['retrieval_recall']:.2f}" if r['retrieval_precision'] is not None else "ret P/R=n/a")
        )

    (out / "results.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    plot(results, args.episodes, args.change_at, out / "curves.png")
    print(f"wrote {out/'curves.png'} and {out/'results.json'}")


if __name__ == "__main__":
    main()
