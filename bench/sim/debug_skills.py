"""Run individual physical skills and save a filmstrip for visual debugging.

    MUJOCO_GL=glfw vendor/rma-venv/bin/python -m bench.sim.debug_skills open top
    MUJOCO_GL=glfw vendor/rma-venv/bin/python -m bench.sim.debug_skills pick butter --heavy butter
    MUJOCO_GL=glfw vendor/rma-venv/bin/python -m bench.sim.debug_skills full butter top --sticky top
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..world import Task
from .skill_env import SimProps, SimSkillEnv


def strip(frames: list[np.ndarray], n: int = 8) -> np.ndarray:
    if not frames:
        return np.zeros((128, 128, 3), np.uint8)
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    return np.concatenate([frames[i] for i in idx], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("skill", choices=["open", "pull_hard", "pick", "pick_firm", "place", "full"])
    ap.add_argument("target", nargs="+")
    ap.add_argument("--sticky", default="none")
    ap.add_argument("--heavy", default="none")
    ap.add_argument("--out", default="outputs/sim_debug")
    args = ap.parse_args()

    obj = next((t for t in args.target if t in ("cream_cheese", "butter", "chocolate_pudding")), "butter")
    drawer = next((t for t in args.target if t in ("top", "middle", "bottom")), "top")
    task = Task(obj, drawer)
    env = SimSkillEnv(SimProps(args.sticky, args.heavy), task, 0, render=True)
    t0 = time.time()
    if args.skill in ("open", "pull_hard"):
        ev = getattr(env, args.skill)(drawer)
        print(ev, "drawer qpos:", round(env._drawer_qpos(drawer), 3))
    elif args.skill in ("pick", "pick_firm"):
        ev = getattr(env, args.skill)(obj)
        print(ev, "obj z:", round(env._obj_pos(obj)[2], 3), "eef z:", round(env._eef()[2], 3))
    elif args.skill == "place":
        print(env.open(drawer)); print(env.pick(obj)); print(env.place(obj, drawer))
    else:
        from ..memory import Beliefs
        from ..planner import run_planner
        run_planner(env, Beliefs())
        print(env.log.text)
    print(f"steps={env.log.steps} success={env.log.success} stale={env.log.stale_actions} "
          f"wall={time.time()-t0:.1f}s")

    import imageio
    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = Path(args.out) / f"{args.skill}_{'_'.join(args.target)}.png"
    imageio.imwrite(p, strip(env.frames))
    print("filmstrip ->", p)
    env.close()


if __name__ == "__main__":
    main()
