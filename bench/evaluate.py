"""Evaluate a memory module on the standard protocol and add it to the leaderboard.

Entrants submit ONLY a memory: a factory returning an object with
    observe(episode_log) -> None
    recall(task, initial_obs) -> bench.memory.Beliefs
(optionally surfaced(task) -> [episode indices] for retrieval P/R, bytes_stored()).

    uv run python -m bench.evaluate --memory memlayer:MemoryLayer --name memlayer-L2
    uv run python -m bench.evaluate --memory /abs/path/my_memory.py:build --name my-memory

Standard protocol (abstract env, Phase 3 world): 30 worlds x 50 episodes x 3 seeds,
properties sticky,heavy,location,fast,preference; task kinds put,put_any,fetch;
change event at 25 + 2 random. Writes outputs/leaderboard/<name>.json and
regenerates docs/leaderboard.md.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from .memory import BASELINES
from .run import run_sequence, summarize

PROTOCOL = dict(worlds=30, episodes=50, seeds=3, change_at=25, extra_changes=2,
                property_types=("sticky", "heavy", "location", "fast", "preference"),
                task_kinds=("put", "put_any", "fetch"))
LB_DIR = Path("outputs/leaderboard")
LB_MD = Path("docs/leaderboard.md")


def load_factory(spec: str):
    mod, _, attr = spec.rpartition(":")
    if mod.endswith(".py"):
        s = importlib.util.spec_from_file_location("entrant", mod)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, attr)


def evaluate(factory, name: str) -> dict:
    t0 = time.time()
    P = PROTOCOL
    runs = [run_sequence(factory(), w, s, P["episodes"], P["change_at"], P["extra_changes"], P["property_types"], P["task_kinds"])
            for s in range(P["seeds"]) for w in range(P["worlds"])]
    r = summarize(runs, P["episodes"], P["change_at"])
    rows = [x for run in runs for x in run["rows"]]
    by_kind = {k: float(np.mean([x["success"] for x in rows if x["kind"] == k])) for k in P["task_kinds"]}
    r.update(name=name, date=str(date.today()), by_kind=by_kind, mean_steps=float(np.mean([x["steps"] for x in rows])),
             wall_s=round(time.time() - t0, 1))
    r.pop("curve_success"); r.pop("curve_steps"); r.pop("curve_stale")
    return r


def write_leaderboard() -> None:
    entries = [json.loads(p.read_text()) for p in sorted(LB_DIR.glob("*.json"))]
    entries.sort(key=lambda e: -e["auc_success"])
    lines = ["# Leaderboard — cross-episode memory benchmark (abstract env, Phase 3 protocol)", "",
             f"Protocol: {PROTOCOL['worlds']} worlds × {PROTOCOL['episodes']} episodes × {PROTOCOL['seeds']} seeds; "
             f"properties {', '.join(PROTOCOL['property_types'])}; tasks {', '.join(PROTOCOL['task_kinds'])}; "
             f"change at {PROTOCOL['change_at']} + {PROTOCOL['extra_changes']} random. "
             "Higher AUC is better; read stale actions and steps next to it (a success-only ranking rewards never revising).", "",
             "| memory | AUC | pre-change [95% CI] | post-change (10) | stale (post) | steps | put | put_any | fetch | bytes | ret P/R | date |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in entries:
        lo, hi = e["success_pre_change_ci"]
        pr = f"{e['retrieval_precision']:.2f}/{e['retrieval_recall']:.2f}" if e.get("retrieval_precision") is not None else "—"
        k = e["by_kind"]
        lines.append(f"| {e['name']} | **{e['auc_success']:.3f}** | {e['success_pre_change']:.2f} [{lo:.2f}, {hi:.2f}] | "
                     f"{e['success_post_change_first10']:.2f} | {e['stale_actions_post_change']:.1f} | {e['mean_steps']:.1f} | "
                     f"{k['put']:.2f} | {k['put_any']:.2f} | {k['fetch']:.2f} | {e['bytes_stored_mean']:.0f} | {pr} | {e['date']} |")
    lines += ["", "Submit: `uv run python -m bench.evaluate --memory <module_or_file>:<factory> --name <name>` "
              "(see `bench/evaluate.py` for the memory interface)."]
    LB_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", help="module:factory or /path/file.py:factory")
    ap.add_argument("--name")
    ap.add_argument("--baselines", action="store_true", help="(re)evaluate the built-in baselines")
    args = ap.parse_args()
    LB_DIR.mkdir(parents=True, exist_ok=True)
    todo = []
    if args.baselines:
        todo += [(n, f) for n, f in BASELINES.items()]
    if args.memory:
        todo.append((args.name or args.memory, load_factory(args.memory)))
    for name, f in todo:
        r = evaluate(f, name)
        (LB_DIR / f"{name}.json").write_text(json.dumps(r, indent=1))
        print(f"{name:22s} AUC={r['auc_success']:.3f} pre={r['success_pre_change']:.2f} post10={r['success_post_change_first10']:.2f} "
              f"stale={r['stale_actions_post_change']:.1f} steps={r['mean_steps']:.1f} kinds={ {k: round(v, 2) for k, v in r['by_kind'].items()} } ({r['wall_s']}s)")
    write_leaderboard()
    print(f"wrote {LB_MD}")


if __name__ == "__main__":
    main()
