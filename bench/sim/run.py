"""Experiment X2 in physics: the abstract benchmark's protocol with SimSkillEnv.

    MUJOCO_GL=glfw caffeinate -i vendor/rma-venv/bin/python -m bench.sim.run \
        --worlds 5 --episodes 30 --seeds 1 --change-at 15 --budget-slack 260

Episodes are ~5-10 s each, so results stream to outputs/bench_sim/episodes.jsonl
(resumable: finished (memory, seed, world) sequences are skipped on re-run) and the
summary/chart reuse bench.run's summarize/plot.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import re

from ..memory import BASELINES
from ..planner import run_planner
from ..run import plot, summarize
from ..world import make_world
from .skill_env import DRAWERS, OBJECTS, SimProps, SimSkillEnv


def load_nominal_steps(calib_log: str) -> dict[tuple[str, str], int]:
    """Median no-secret, no-belief step count per (object, drawer) from a calibrate.py log."""
    import statistics

    samples: dict[tuple[str, str], list[int]] = {}
    for line in Path(calib_log).read_text().splitlines():
        m = re.match(r"(\S+)\s+(\S+)\s+sticky=0 heavy=0 belief=none\s+steps=\s*(\d+)", line)
        if m:
            samples.setdefault((m[1], m[2]), []).append(int(m[3]))
    nominal = {k: int(statistics.median(v)) for k, v in samples.items()}
    if not nominal:
        raise SystemExit(f"no nominal rows in {calib_log}")
    return nominal


def run_sequence(memory, world_id: int, seed: int, episodes: int, change_at: int, budget, log_f,
                 property_types=("sticky", "heavy"), task_kinds=("put",)) -> dict:
    world = make_world(world_id, seed, drawers=tuple(DRAWERS), objects=tuple(OBJECTS),
                       property_types=tuple(property_types), task_kinds=tuple(task_kinds))
    rows = []
    for ep in range(episodes):
        if ep == change_at:
            world.apply_change_event()
        task = world.sample_task()
        beliefs = memory.recall(task, initial_obs={"task": task.text})
        step_budget = budget(task) if callable(budget) else budget
        env = SimSkillEnv(SimProps.from_world(world.props), task, ep,
                          step_budget=step_budget, seed=seed * 1000 + world_id * 100 + ep)
        t0 = time.time()
        run_planner(env, beliefs)
        memory.observe(env.log)
        row = {
            "memory": memory.name, "seed": seed, "world": world_id, "ep": ep,
            "success": int(env.log.success), "steps": env.log.steps, "budget": step_budget, "stale": env.log.stale_actions,
            "failures": sum(e.outcome in ("jam", "drop") for e in env.log.events),
            "task": task.text, "kind": task.kind,
            "props": [world.props.sticky_drawer, world.props.heavy_object, world.props.hidden_object,
                      world.props.hidden_in, world.props.fast_drawer],
            "wasted_looks": env.log.wasted_looks,
            "events": [f"{e.skill}({e.target})->{e.outcome}" for e in env.log.events],
            "wall": round(time.time() - t0, 1),
        }
        env.shutdown()
        rows.append(row)
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()
        print(f"[{memory.name:13s} s{seed} w{world_id} ep{ep:2d}] {'OK ' if row['success'] else 'FAIL'} "
              f"{row['steps']:4d} steps stale={row['stale']} {row['wall']:4.1f}s  {task.text}", flush=True)
    return {"rows": rows, "bytes": memory.bytes_stored()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--change-at", type=int, default=15)
    ap.add_argument("--budget-slack", type=int, default=260,
                    help="per-task budget = nominal (no-secret) steps + slack. Calibrated 2026-09-03: knowing both "
                         "secrets costs up to +211, a drop recovery ~+110, a jam recovery >= +350 -> slack in "
                         "[211, 350) makes jams decisive and everything else survivable; 260 leaves ~50 margin")
    ap.add_argument("--calib-log", default="outputs/sim_calibrate3.log")
    ap.add_argument("--memories", default=",".join(BASELINES))
    ap.add_argument("--properties", default="sticky,heavy", help="e.g. sticky,heavy,location,fast")
    ap.add_argument("--kinds", default="put", help="e.g. put,put_any,fetch")
    ap.add_argument("--fetch-slack", type=int, default=400,
                    help="fetch budget = 360 (known-location cost) + this; one wasted look (~330) survives, two do not")
    ap.add_argument("--out", default="outputs/bench_sim")
    args = ap.parse_args()

    nominal = load_nominal_steps(args.calib_log)
    import statistics
    put_median = int(statistics.median(nominal.values()))

    def budget(task):
        if task.kind == "fetch":
            return 360 + args.fetch_slack  # measured: known-location fetch ~358-361 from any drawer
        if task.kind == "put_any":
            return put_median + args.budget_slack
        return nominal[(task.obj, task.drawer)] + args.budget_slack
    print("per-task budgets (put):", {f"{o}->{d}": nominal[(o, d)] + args.budget_slack for (o, d) in nominal}, flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "episodes.jsonl"
    done: dict[tuple, list] = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            done.setdefault((r["memory"], r["seed"], r["world"]), []).append(r)

    results = {}
    with log_path.open("a") as log_f:
        for name in args.memories.split(","):
            factory = BASELINES[name]
            runs = []
            for s in range(args.seeds):
                for w in range(args.worlds):
                    prev = done.get((factory().name, s, w))
                    if prev and len(prev) >= args.episodes:
                        # a world rerun after a crash appends a fresh full sequence after the
                        # partial one: the LAST `episodes` rows are the complete run
                        runs.append({"rows": prev[-args.episodes :], "bytes": 0})
                        continue
                    runs.append(run_sequence(factory(), w, s, args.episodes, args.change_at, budget, log_f,
                                             args.properties.split(","), args.kinds.split(",")))
            results[name] = summarize(runs, args.episodes, args.change_at)
            r = results[name]
            print(f"== {name:14s} AUC={r['auc_success']:.3f} pre={r['success_pre_change']:.2f} "
                  f"post10={r['success_post_change_first10']:.2f} stale(post)={r['stale_actions_post_change']:.1f}", flush=True)

    (out / "results.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    plot(results, args.episodes, args.change_at, out / "curves.png",
         budget_label=f"budget = nominal + {args.budget_slack}", title="Cross-episode memory benchmark v0.5 (robosuite physics)")
    print(f"wrote {out/'curves.png'}")


if __name__ == "__main__":
    main()
