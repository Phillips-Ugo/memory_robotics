"""Probe: create a LIBERO env from a BDDL file, list the sim handles we need for
scripted skills (drawer joints, object bodies, gripper), and time raw stepping.

    MUJOCO_GL=glfw vendor/rma-venv/bin/python bench/sim/probe_env.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor/RoboMemArena/evaluation_benchmark/libero_fork"))

import numpy as np
from libero.libero.envs import OffScreenRenderEnv

bddl = (
    ROOT
    / "vendor/RoboMemArena/evaluation_benchmark/libero_fork/libero/bddl_files/libero_10"
    / "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl"
)
if not bddl.exists():
    cands = sorted((bddl.parent).glob("*drawer*"))
    print("candidates:", *[c.name for c in cands], sep="\n  ")
    bddl = cands[0]
print("BDDL:", bddl.name)

env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
obs = env.reset()
sim = env.sim
m = sim.model

print("\n-- obs keys:", sorted(obs.keys()))
print("-- joints containing 'level' or 'cabinet':")
for j in m.joint_names:
    if "level" in j or "cabinet" in j:
        print("   ", j, "qpos_addr", m.get_joint_qpos_addr(j), "frictionloss", m.dof_frictionloss[m.jnt_dofadr[m.joint_name2id(j)]])
print("-- bodies (non-robot, first 40):")
print("   ", [b for b in m.body_names if not b.startswith("robot") and not b.startswith("gripper")][:40])
print("-- object names known to env:", list(env.env.objects_dict.keys()) if hasattr(env.env, "objects_dict") else "n/a")
print("-- eef pos:", obs.get("robot0_eef_pos"), "gripper qpos:", obs.get("robot0_gripper_qpos"))

# how far does one full-scale action move the end effector? (controller scaling)
p0 = obs["robot0_eef_pos"].copy()
for _ in range(10):
    obs, *_ = env.step(np.array([1.0, 0, 0, 0, 0, 0, -1.0]))
print("-- +x for 10 steps moved eef by", np.round(obs["robot0_eef_pos"] - p0, 4))

t0 = time.time()
n = 100
for _ in range(n):
    env.step(np.zeros(7))
print(f"-- {n} steps in {time.time()-t0:.2f}s -> {(time.time()-t0)/n*1000:.1f} ms/step")
env.close()
