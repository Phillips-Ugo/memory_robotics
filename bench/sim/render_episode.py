"""Render one physics episode to mp4.

    MUJOCO_GL=glfw vendor/rma-venv/bin/python -m bench.sim.render_episode butter bottom --sticky bottom --heavy butter
"""
from __future__ import annotations
import argparse
from pathlib import Path
import imageio
from ..memory import Beliefs
from ..planner import run_planner
from ..world import Task
from .skill_env import SimProps, SimSkillEnv

ap = argparse.ArgumentParser()
ap.add_argument("obj"); ap.add_argument("drawer")
ap.add_argument("--sticky", default="none"); ap.add_argument("--heavy", default="none")
ap.add_argument("--believe", action="store_true", help="planner already knows the secrets")
ap.add_argument("--out", default="outputs/sim_videos")
a = ap.parse_args()
b = Beliefs()
if a.believe:
    b.sticky_drawers.add(a.sticky); b.heavy_objects.add(a.heavy)
env = SimSkillEnv(SimProps(a.sticky, a.heavy), Task(a.obj, a.drawer), 0, step_budget=5000, render=True, cam_size=384, render_every=2)
run_planner(env, b)
print(env.log.text)
Path(a.out).mkdir(parents=True, exist_ok=True)
p = Path(a.out) / f"{a.obj}_{a.drawer}_sticky-{a.sticky}_heavy-{a.heavy}{'_known' if a.believe else ''}.mp4"
imageio.mimwrite(p, env.frames, fps=20)
print(f"{len(env.frames)} frames -> {p}")
env.close()
