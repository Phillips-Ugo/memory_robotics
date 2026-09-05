"""Render a whole episode sequence (one world, one memory) into a single mp4 with an
overlay: episode number, task, what the memory currently believes, outcome, and the
change event. A "day in the life" of the robot.

    MUJOCO_GL=glfw caffeinate -dis vendor/rma-venv/bin/python -m bench.sim.render_sequence \
        --memory consolidated --world 2 --episodes 40 --change-at 20 --out outputs/sim_videos/day_in_the_life.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..memory import BASELINES
from ..planner import run_planner
from ..world import make_world
from .skill_env import DRAWERS, OBJECTS, SimProps, SimSkillEnv

ap = argparse.ArgumentParser()
ap.add_argument("--memory", default="consolidated", choices=list(BASELINES))
ap.add_argument("--world", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--episodes", type=int, default=40)
ap.add_argument("--change-at", type=int, default=20)
ap.add_argument("--properties", default="sticky,heavy,location,fast")
ap.add_argument("--kinds", default="put,put_any,fetch")
ap.add_argument("--budget-slack", type=int, default=260)
ap.add_argument("--render-every", type=int, default=2)
ap.add_argument("--out", default="outputs/sim_videos/day_in_the_life.mp4")
a = ap.parse_args()

W = H = 480
font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
BAR = 150


def beliefs_text(b) -> list[str]:
    lines = []
    if b.sticky_drawers:
        lines.append("sticky drawer: " + ", ".join(sorted(b.sticky_drawers)))
    if b.heavy_objects:
        lines.append("heavy: " + ", ".join(sorted(b.heavy_objects)))
    if b.object_in:
        lines.append("location: " + ", ".join(f"{o} in {d}" for o, d in b.object_in.items()))
    if b.fast_drawers:
        lines.append("fast drawer: " + ", ".join(sorted(b.fast_drawers)))
    if b.probe_drawers or b.probe_objects:
        lines.append("re-checking: " + ", ".join(sorted(b.probe_drawers | b.probe_objects)))
    return lines or ["(nothing yet)"]


def overlay(frame, ep, task, blines, status, changed):
    c = Image.new("RGB", (W, H + BAR), (16, 16, 16))
    c.paste(Image.fromarray(frame), (0, BAR))
    d = ImageDraw.Draw(c)
    d.text((12, 8), f"Task {ep + 1}: {task}", fill=(235, 235, 235), font=font)
    d.text((12, 38), "memory believes:", fill=(140, 140, 140), font=small)
    for i, l in enumerate(blines[:3]):
        d.text((150, 38 + 20 * i), l, fill=(110, 225, 140), font=small)
    if changed:
        d.text((12, 100), "! the world just changed", fill=(255, 190, 70), font=small)
    if status:
        d.text((12, 122), status, fill=(255, 110, 90) if "FAIL" in status else (110, 225, 140), font=small)
    return np.asarray(c)


mem = BASELINES[a.memory]()
world = make_world(a.world, a.seed, drawers=tuple(DRAWERS), objects=tuple(OBJECTS),
                   property_types=tuple(a.properties.split(",")), task_kinds=tuple(a.kinds.split(",")))
print("world secrets:", world.props, flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
writer = imageio.get_writer(a.out, fps=24, quality=8, macro_block_size=1)
changed_type = None
log = []
for ep in range(a.episodes):
    if ep == a.change_at:
        changed_type = world.apply_change_event()
        print(f"change event: {changed_type} -> {world.props}", flush=True)
    task = world.sample_task()
    b = mem.recall(task, initial_obs={"task": task.text})
    blines = beliefs_text(b)
    nominal = {"put": 350, "put_any": 350, "fetch": 360}[task.kind]
    env = SimSkillEnv(SimProps.from_world(world.props), task, ep, step_budget=nominal + (400 if task.kind == "fetch" else a.budget_slack),
                      render=True, seed=a.seed * 1000 + a.world * 100 + ep, cam_size=W, render_every=a.render_every)
    m = env.sim.model
    cam = m.camera_name2id("agentview"); m.cam_pos[cam] = np.array([0.70, 0.05, 1.60]); m.cam_fovy[cam] = 40.0
    env.frames.clear()
    env.rng = world.rng
    run_planner(env, b)
    mem.observe(env.log)
    status = f"{'OK' if env.log.success else 'FAIL'} in {env.log.steps} steps  |  " + "; ".join(f"{e.skill}->{e.outcome}" for e in env.log.events)
    frames = env.frames
    for i, f in enumerate(frames):
        writer.append_data(overlay(f, ep, task.text, blines, status if i > len(frames) - 30 else "", ep == a.change_at and i < 60))
    for _ in range(24):  # 1 s hold on the result
        writer.append_data(overlay(frames[-1], ep, task.text, blines, status, False))
    log.append({"ep": ep, "task": task.text, "success": env.log.success, "steps": env.log.steps, "beliefs": blines})
    print(f"[{ep:2d}] {'OK ' if env.log.success else 'FAIL'} {env.log.steps:4d}  {task.text:45s} | {' / '.join(blines)}", flush=True)
    env.shutdown()
writer.close()
Path(a.out).with_suffix(".json").write_text(json.dumps(log, indent=1))
print("wrote", a.out, flush=True)
