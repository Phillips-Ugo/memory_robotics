"""Render one physics episode to mp4 (optionally zoomed camera + slow-motion on failures).

    MUJOCO_GL=glfw vendor/rma-venv/bin/python -m bench.sim.render_episode butter bottom \
        --sticky bottom --heavy butter [--believe] [--zoom] [--slowmo]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np

from ..memory import Beliefs
from ..planner import run_planner
from ..world import Task
from .skill_env import SimProps, SimSkillEnv

ap = argparse.ArgumentParser()
ap.add_argument("obj"); ap.add_argument("drawer")
ap.add_argument("--sticky", default="none"); ap.add_argument("--heavy", default="none")
ap.add_argument("--believe", action="store_true", help="planner already knows the secrets")
ap.add_argument("--zoom", action="store_true", help="tighter agentview: closer, narrower field of view")
ap.add_argument("--slowmo", action="store_true", help="3x slow-motion around jams/drops")
ap.add_argument("--size", type=int, default=480)
ap.add_argument("--out", default="outputs/sim_videos")
a = ap.parse_args()

b = Beliefs()
if a.believe:
    b.sticky_drawers.add(a.sticky); b.heavy_objects.add(a.heavy)
env = SimSkillEnv(SimProps(a.sticky, a.heavy), Task(a.obj, a.drawer), 0, step_budget=5000,
                  render=True, cam_size=a.size, render_every=2)
if a.zoom:
    m = env.sim.model
    cam = m.camera_name2id("agentview")
    m.cam_pos[cam] = np.array([0.70, 0.05, 1.60])  # closer, lower, shifted toward the cabinet (default 0.86, 0, 1.81)
    m.cam_fovy[cam] = 40.0                          # narrower than 45: objects larger on screen
    env.sim.forward()
run_planner(env, b)
print(env.log.text)

frames = env.frames
if a.slowmo:
    # frame index of each event boundary (render_every=2 -> frame = step // 2)
    out, cursor = [], 0
    bounds, t = [], 0
    for e in env.log.events:
        t += e.steps
        bounds.append((t // 2, e.outcome))
    slow = set()
    for f_end, outcome in bounds:
        if outcome in ("jam", "drop"):
            slow.update(range(max(0, f_end - 45), f_end + 5))
    for i, f in enumerate(frames):
        out.extend([f] * (3 if i in slow else 1))
    frames = out

Path(a.out).mkdir(parents=True, exist_ok=True)
p = Path(a.out) / f"{a.obj}_{a.drawer}_sticky-{a.sticky}_heavy-{a.heavy}{'_known' if a.believe else ''}{'_zoom' if a.zoom else ''}{'_slow' if a.slowmo else ''}.mp4"
imageio.mimwrite(p, frames, fps=24, quality=8)
print(f"{len(frames)} frames -> {p}")
env.close()
