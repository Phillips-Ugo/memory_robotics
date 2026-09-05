"""Calibrate the physics benchmark's step budget.

Runs the fixed planner under every combination of {task} x {no secret, sticky drawer,
heavy object, both} x {beliefs: none, correct, stale} and prints step counts, so the
budget can be set where a single unrecovered failure decides the episode (v0 rule).

    MUJOCO_GL=glfw caffeinate -i vendor/rma-venv/bin/python -m bench.sim.calibrate --seeds 3

The nominal (no-secret, no-belief) rows are what bench.sim.run keys budgets on, so
they are run over --seeds placements and reported per seed; run.py takes the median.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ..memory import Beliefs
from ..planner import run_planner
from ..world import Task
from .skill_env import DRAWERS, OBJECTS, SimProps, SimSkillEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1, help="placements to sample for the nominal rows")
    args = ap.parse_args()
    rows = []
    t0 = time.time()
    tasks = [Task(o, d) for o in OBJECTS for d in DRAWERS]
    for task in tasks:
        for sticky, heavy in ((False, False), (True, False), (False, True), (True, True)):
            props = SimProps(task.drawer if sticky else "none", task.obj if heavy else "none")
            seeds = range(args.seeds) if (not sticky and not heavy) else [0]
            for belief, seed in ((b, sd) for b in ("none", "correct", "stale") for sd in (seeds if b == "none" else [0])):
                b = Beliefs()
                if belief == "correct":
                    if sticky:
                        b.sticky_drawers.add(task.drawer)
                    if heavy:
                        b.heavy_objects.add(task.obj)
                elif belief == "stale":  # believes both secrets; wrong wherever they're absent
                    b.sticky_drawers.add(task.drawer)
                    b.heavy_objects.add(task.obj)
                env = SimSkillEnv(props, task, 0, step_budget=5000, seed=seed)
                run_planner(env, b)
                rows.append((task.text, sticky, heavy, belief, env.log.steps, env.log.success, env.log.stale_actions,
                             ";".join(f"{e.skill}->{e.outcome}" for e in env.log.events)))
                print(f"{task.obj:17s} {task.drawer:6s} sticky={int(sticky)} heavy={int(heavy)} belief={belief:7s} "
                      f"steps={env.log.steps:4d} success={int(env.log.success)} stale={env.log.stale_actions} | {rows[-1][-1]}",
                      flush=True)
                env.shutdown()

    steps = np.array([r[4] for r in rows])
    succ = np.array([r[5] for r in rows])
    print(f"\n{len(rows)} episodes in {(time.time()-t0)/60:.1f} min; placement success (no budget): {succ.mean():.2f}")
    for cond in ("none", "correct", "stale"):
        sel = [r for r in rows if r[3] == cond and r[5]]
        if sel:
            s = np.array([r[4] for r in sel])
            print(f"belief={cond:7s}: steps mean {s.mean():.0f}  min {s.min()}  max {s.max()}  (n={len(s)})")
    opt = [r[4] for r in rows if r[3] == "correct" and r[5]]
    rec = [r[4] for r in rows if r[3] == "none" and (r[1] or r[2]) and r[5]]
    if opt and rec:
        print(f"\nsuggested budget: between max optimal ({max(opt)}) and min single-recovery ({min(rec)}) "
              f"-> try {int((max(opt)+min(rec))/2)}")


if __name__ == "__main__":
    main()
