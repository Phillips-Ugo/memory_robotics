"""Experiment X2 in physics: the abstract benchmark's protocol with SimSkillEnv.

    MUJOCO_GL=glfw caffeinate -i vendor/rma-venv/bin/python -m bench.sim.run \
        --worlds 5 --episodes 30 --seeds 1 --change-at 15 --budget <from calibrate>

Episodes are ~5-10 s each, so results stream to outputs/bench_sim/episodes.jsonl
(resumable: finished (memory, seed, world) sequences are skipped on re-run) and the
summary/chart reuse bench.run's summarize/plot.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..memory import BASELINES
from ..planner import run_planner
from ..run import plot, summarize
from ..world import make_world
from .skill_env import DRAWERS, OBJECTS, SimProps, SimSkillEnv


def run_sequence(memory, world_id: int, seed: int, episodes: int, change_at: int, budget: int, log_f) -> dict:
    world = make_world(world_id, seed, drawers=tuple(DRAWERS), objects=tuple(OBJECTS))
    rows = []
    for ep in range(episodes):
        if ep == change_at:
            world.apply_change_event()
        task = world.sample_task()
        beliefs = memory.recall(task, initial_obs={"task": task.text})
        env = SimSkillEnv(SimProps(world.props.sticky_drawer, world.props.heavy_object), task, ep,
                          step_budget=budget, seed=seed * 1000 + world_id * 100 + ep)
        t0 = time.time()
        run_planner(env, beliefs)
        memory.observe(env.log)
        row = {
            "memory": memory.name, "seed": seed, "world": world_id, "ep": ep,
            "success": int(env.log.success), "steps": env.log.steps, "stale": env.log.stale_actions,
            "failures": sum(e.outcome in ("jam", "drop") for e in env.log.events),
            "task": task.text, "props": [world.props.sticky_drawer, world.props.heavy_object],
            "events": [f"{e.skill}({e.target})->{e.outcome}" for e in env.log.events],
            "wall": round(time.time() - t0, 1),
        }
        env.close()
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
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--memories", default=",".join(BASELINES))
    ap.add_argument("--out", default="outputs/bench_sim")
    args = ap.parse_args()

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
                        runs.append({"rows": prev[: args.episodes], "bytes": 0})
                        continue
                    runs.append(run_sequence(factory(), w, s, args.episodes, args.change_at, args.budget, log_f))
            results[name] = summarize(runs, args.episodes, args.change_at)
            r = results[name]
            print(f"== {name:14s} AUC={r['auc_success']:.3f} pre={r['success_pre_change']:.2f} "
                  f"post10={r['success_post_change_first10']:.2f} stale(post)={r['stale_actions_post_change']:.1f}", flush=True)

    (out / "results.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    plot(results, args.episodes, args.change_at, out / "curves.png")
    print(f"wrote {out/'curves.png'}")


if __name__ == "__main__":
    main()
