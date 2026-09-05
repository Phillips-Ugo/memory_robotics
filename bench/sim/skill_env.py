"""Physics version of bench.env.SkillEnv: the same skills, executed by scripted
controllers in a LIBERO/robosuite scene (Panda + 3-drawer cabinet + 3 objects).

Grasping model (documented simplification): a "magnetic" grasp — when the closed
gripper is within reach of a handle/object we activate a MuJoCo weld constraint
between the end effector and that body. Every grasp has a FORCE LIMIT: if the weld
has to transmit more than the limit, it breaks (the object drops / the hook slips).
  gentle hook / light grip : limit ~15 N, cheap
  firm hook / firm grip    : limit ~80 N, costs extra steps (squeeze + slow motion)
Hidden properties are physical:
  sticky drawer -> joint frictionloss 40 N on that drawer: a gentle hook slips (jam),
                   a firm pull opens it
  heavy object  -> mass 2 kg (~20 N): a light grip breaks on lift (drop), a firm grip holds
Why not real finger grasps: the cabinet's handle slot is 1.6 cm deep and the Panda's
closed fingertips are ~1.7 cm — a millimetre grasp-tuning problem that has nothing to
do with what this benchmark measures (memory). The policy is fixed and dumb on purpose.

Skills use privileged sim state (object poses). Outcomes mirror the abstract env:
ok | jam | drop; success = object inside the target drawer. Step costs are real sim
steps, so the step budget is a real quantity (calibrate with bench/sim/calibrate.py).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor/RoboMemArena/evaluation_benchmark/libero_fork"))

import mujoco  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from ..env import EpisodeLog, SkillEvent  # noqa: E402
from ..world import Task  # noqa: E402

BDDL = ROOT / "bench/sim/bddl/drawer_world.bddl"

DRAWERS = ("top", "middle", "bottom")
OBJECTS = {"cream_cheese": "cream_cheese_1", "butter": "butter_1", "chocolate_pudding": "chocolate_pudding_1"}
DRAWER_BODY = {d: f"white_cabinet_1_cabinet_{d}" for d in DRAWERS}
OBJ_BODY = {o: f"{n}_main" for o, n in OBJECTS.items()}
EEF_BODY = "gripper0_eef"

# handle bar centre in the cabinet's local frame (from white_cabinet.xml); drawers slide out along -y
HANDLE_LOCAL = {"top": np.array([-0.003, -0.1016, 0.184]),
                "middle": np.array([-0.003, -0.1016, 0.1175]),
                "bottom": np.array([-0.003, -0.1016, 0.051])}
OPEN_DIR_LOCAL = np.array([0.0, -1.0, 0.0])
# hook point relative to the bar: (stand-off in front, height above). The bottom drawer
# needs a higher hook so the wrist clears the middle handle on the way down.
HOOK_OFFSET = {"top": (0.05, 0.0), "middle": (0.05, 0.0), "bottom": (0.06, 0.04)}

STICKY_FRICTIONLOSS = 40.0
FAST_DAMPING = 5.0  # default drawer damping is 50; the gentle pull opens a fast drawer in ~1/3 the steps
HEAVY_MASS = 0.5  # kg -> ~5 N on the grasp; default boxes ~0.1 N. Heavier made the firm carry physically slow (controller sags)
GENTLE_LIMIT = 20.0  # N (measured: normal drawer needs 1-4 N sustained, sticky ~32 N)
FIRM_LIMIT = 80.0  # N
GRIP_LIGHT_LIMIT = 3.0  # N: light grip breaks under the heavy object (~5 N), holds the light ones (~0.1 N)
REACH = 0.035  # m: max eef-to-target distance for the magnetic grasp to engage


@dataclass
class SimProps:
    sticky_drawer: str
    heavy_object: str
    hidden_object: str | None = None  # starts inside `hidden_in` (teleported at reset)
    hidden_in: str | None = None
    fast_drawer: str | None = None  # low damping: the gentle pull finishes in fewer steps

    @classmethod
    def from_world(cls, props) -> "SimProps":
        return cls(props.sticky_drawer, props.heavy_object, props.hidden_object, props.hidden_in, props.fast_drawer)


class SimSkillEnv:
    """Same surface as bench.env.SkillEnv, backed by MuJoCo."""

    drawer_names = tuple(DRAWERS)
    object_names = tuple(OBJECTS)

    def __init__(self, props: SimProps, task: Task, episode_idx: int, step_budget: int = 700,
                 render: bool = False, seed: int = 0, cam_size: int = 128, render_every: int = 3) -> None:
        self.render_every = render_every
        self.props, self.task = props, task
        self.step_budget = step_budget
        self.log = EpisodeLog(episode_idx=episode_idx, task=task)
        self.done = False
        self.frames: list[np.ndarray] = []
        self.render = render
        self.env = OffScreenRenderEnv(bddl_file_name=str(BDDL), camera_heights=cam_size, camera_widths=cam_size,
                                      horizon=20000)  # robosuite's default 1000-step horizon would cut long episodes
        self.env.seed(seed)
        self.obs = self.env.reset()
        self._install_welds()
        self.sim = self.env.sim
        self.holding: str | None = None
        self._active_eq: int | None = None
        self._limit = GENTLE_LIMIT
        self._over = 0  # consecutive steps the weld force exceeded the limit
        self.open_drawers: set[str] = set()
        self._inject_props()
        self._yaw_to_x()  # once per episode: hand's wide axis along the handle bars

    def _finger_axis(self) -> np.ndarray:
        m, d = self.sim.model, self.sim.data
        v = d.body_xpos[m.body_name2id("gripper0_rightfinger")] - d.body_xpos[m.body_name2id("gripper0_leftfinger")]
        return v / np.linalg.norm(v)

    def _yaw_to_x(self, max_steps: int = 60) -> None:
        """Rotate the gripper 90° about z (holding position) so the Panda hand's wide
        axis runs along the drawer handles instead of into the handle above."""
        p0 = self._eef().copy()
        for _ in range(max_steps):
            if abs(self._finger_axis()[0]) > 0.98:
                return
            a = np.zeros(7)
            a[:3] = np.clip(8.0 * (p0 - self._eef()), -1, 1)
            a[5] = 0.4
            a[6] = -1.0
            self._step(a)

    # ---- model surgery ----------------------------------------------------
    def _install_welds(self) -> None:
        """Reload the model with one inactive weld per graspable body, keeping state."""
        rs = self.env.env  # robosuite env
        state = rs.sim.get_state()
        xml = rs.sim.model.get_xml()
        welds = "".join(
            f'<weld name="grasp_{b}" body1="{EEF_BODY}" body2="{b}" active="false" solref="0.002 1"/>'
            for b in list(DRAWER_BODY.values()) + list(OBJ_BODY.values())
        )
        if "<equality>" in xml:
            xml = xml.replace("<equality>", "<equality>" + welds, 1)
        else:
            xml = re.sub(r"</mujoco>\s*$", f"<equality>{welds}</equality></mujoco>", xml)
        rs.reset_from_xml_string(xml)
        rs.sim.set_state(state)
        rs.sim.forward()
        self.obs = rs._get_observations(force_update=True)
        raw = rs.sim.model._model  # robosuite wrapper -> mujoco.MjModel
        self.eq_ids = {
            b: mujoco.mj_name2id(raw, int(mujoco.mjtObj.mjOBJ_EQUALITY), f"grasp_{b}")
            for b in list(DRAWER_BODY.values()) + list(OBJ_BODY.values())
        }
        assert min(self.eq_ids.values()) >= 0, self.eq_ids

    def _inject_props(self) -> None:
        m = self.sim.model
        for d in DRAWERS:
            j = m.joint_name2id(f"white_cabinet_1_{d}_level")
            m.dof_frictionloss[m.jnt_dofadr[j]] = STICKY_FRICTIONLOSS if d == self.props.sticky_drawer else 0.0
        for o, body in OBJ_BODY.items():
            if o == self.props.heavy_object:
                m.body_mass[m.body_name2id(body)] = HEAVY_MASS
        if self.props.fast_drawer in DRAWERS:
            j = m.joint_name2id(f"white_cabinet_1_{self.props.fast_drawer}_level")
            m.dof_damping[m.jnt_dofadr[j]] = FAST_DAMPING
        if self.props.hidden_object in OBJECTS and self.props.hidden_in in DRAWERS:
            # teleport the hidden object onto the floor of its drawer (drawer closed)
            jname = f"{OBJECTS[self.props.hidden_object]}_joint0"
            addr = m.get_joint_qpos_addr(jname)
            floor = self._drawer_region_world(self.props.hidden_in)
            _, open_dir = self._handle_world(self.props.hidden_in)
            q = self.sim.data.qpos.copy()
            # front part of the drawer: once opened, a straight lift clears the handle
            # bar of the drawer above and the drawer's own front panel
            q[addr[0]:addr[0] + 3] = floor + open_dir * 0.045 + np.array([0.0, 0.0, 0.03])
            q[addr[0] + 3:addr[0] + 7] = np.array([1.0, 0.0, 0.0, 0.0])
            self.sim.data.qpos[:] = q
            self.sim.data.qvel[:] = 0
        self.sim.forward()
        for _ in range(20):  # let it settle
            self._step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
        self.log.steps = 0
        self.log.events.clear()

    # ---- magnetic grasp ---------------------------------------------------
    def _attach(self, body: str, limit: float) -> bool:
        m, d = self.sim.model, self.sim.data
        b1, b2 = m.body_name2id(EEF_BODY), m.body_name2id(body)
        eq = self.eq_ids[body]
        # relpose of body2 in body1's frame, from the current configuration
        p1, p2 = d.body_xpos[b1], d.body_xpos[b2]
        R1 = d.body_xmat[b1].reshape(3, 3)
        rel_p = R1.T @ (p2 - p1)
        q1 = np.empty(4); mujoco.mju_mat2Quat(q1, d.body_xmat[b1])
        q2 = np.empty(4); mujoco.mju_mat2Quat(q2, d.body_xmat[b2])
        q1inv = np.empty(4); mujoco.mju_negQuat(q1inv, q1)
        rel_q = np.empty(4); mujoco.mju_mulQuat(rel_q, q1inv, q2)
        m.eq_data[eq, :3] = 0.0  # anchor at body2 origin
        m.eq_data[eq, 3:6] = rel_p
        m.eq_data[eq, 6:10] = rel_q
        m.eq_active[eq] = 1
        self._active_eq, self._limit, self._over = eq, limit, 0
        return True

    def _detach(self) -> None:
        if self._active_eq is not None:
            self.sim.model.eq_active[self._active_eq] = 0
            self._active_eq = None

    def _weld_force(self) -> float:
        if self._active_eq is None:
            return 0.0
        d = self.sim.data
        mask = (d.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)) & (d.efc_id == self._active_eq)
        f = d.efc_force[mask]
        return float(np.linalg.norm(f[:3])) if f.size >= 3 else 0.0

    # ---- helpers ----------------------------------------------------------
    def _cab_pose(self):
        b = self.sim.model.body_name2id("white_cabinet_1_main")
        return self.sim.data.body_xpos[b].copy(), self.sim.data.body_xmat[b].reshape(3, 3).copy()

    def _handle_world(self, drawer: str):
        pos, rot = self._cab_pose()
        # the handle rides on the drawer body: add the drawer's current slide offset
        return pos + rot @ (HANDLE_LOCAL[drawer] + OPEN_DIR_LOCAL * -self._drawer_qpos(drawer)), rot @ OPEN_DIR_LOCAL

    def _drawer_qpos(self, drawer: str) -> float:
        return float(self.sim.data.qpos[self.sim.model.get_joint_qpos_addr(f"white_cabinet_1_{drawer}_level")])

    def _drawer_region_world(self, drawer: str) -> np.ndarray:
        return self.sim.data.site_xpos[self.sim.model.site_name2id(f"white_cabinet_1_{drawer}_region")].copy()

    def _obj_pos(self, obj: str) -> np.ndarray:
        return self.sim.data.body_xpos[self.sim.model.body_name2id(OBJ_BODY[obj])].copy()

    def _eef(self) -> np.ndarray:
        return np.asarray(self.obs["robot0_eef_pos"]).copy()

    def _step(self, action: np.ndarray) -> None:
        self.obs, _, _, _ = self.env.step(action)
        self.log.steps += 1
        if self._active_eq is not None:
            # a sustained overload (an object's weight) breaks the grasp; a momentary
            # bump (brushing a drawer wall) does not
            self._over = self._over + 1 if self._weld_force() > self._limit else 0
            if self._over >= 4:
                self._detach()
                self._over = 0
        if self.render and self.log.steps % self.render_every == 0:
            self.frames.append(self.obs["agentview_image"][::-1].copy())
        if self.log.steps > self.step_budget:
            self.done = True

    def _move(self, target, grip: float, gain: float = 8.0, max_steps: int = 120, tol: float = 0.01,
              max_delta: float = 1.0) -> bool:
        for _ in range(max_steps):
            if self.done:
                return False
            err = target - self._eef()
            if np.linalg.norm(err) < tol:
                return True
            a = np.zeros(7)
            a[:3] = np.clip(gain * err, -max_delta, max_delta)
            a[6] = grip
            self._step(a)
        return False

    def _hold(self, grip: float, n: int) -> None:
        for _ in range(n):
            a = np.zeros(7); a[6] = grip
            self._step(a)

    def _record(self, skill: str, target: str, outcome: str, steps: int) -> SkillEvent:
        # a skill cut off by the step budget is not evidence of anything: don't let a
        # budget timeout masquerade as a jam/drop in the memories' episode logs
        if outcome != "ok" and self.log.steps > self.step_budget:
            outcome = "timeout"
        ev = SkillEvent(skill, target, outcome, steps)
        self.log.events.append(ev)
        return ev

    # ---- skills -----------------------------------------------------------
    def _pull(self, drawer: str, firm: bool) -> float:
        handle, open_dir = self._handle_world(drawer)
        # hook point 3.5 cm in FRONT of the bar at bar height; approach with the gripper
        # closed so the fingertips (±1 cm) clear the bar (which starts 0.8 cm behind it)
        front, up = HOOK_OFFSET[drawer]
        hook = handle + open_dir * front + np.array([0, 0, up])
        self._move(hook + np.array([0, 0, 0.10]), grip=1.0, max_steps=150)
        self._move(hook, grip=1.0, max_steps=120, tol=0.008)
        self._hold(1.0, 30 if firm else 6)  # close; firm = brace (robust costs more, but less than a failure)
        if np.linalg.norm(self._eef() - hook) > REACH:
            return 0.0
        self._attach(DRAWER_BODY[drawer], FIRM_LIMIT if firm else GENTLE_LIMIT)
        q0 = self._drawer_qpos(drawer)
        target = self._eef() + open_dir * 0.15
        # force ~ commanded delta (clipped at max_delta): gentle caps at ~16 N, firm reaches ~50 N
        self._move(target, grip=1.0, gain=(8.0 if firm else 4.0), max_steps=(140 if firm else 90), tol=0.01,
                   max_delta=(1.0 if firm else 0.6))
        self._detach()
        self._move(self._eef() + np.array([0, 0, 0.10]), grip=-1.0, max_steps=40)
        if self._drawer_qpos(drawer) < -0.08:
            self.open_drawers.add(drawer)
        return self._drawer_qpos(drawer) - q0

    def open(self, drawer: str) -> SkillEvent:
        s0 = self.log.steps
        moved = self._pull(drawer, firm=False)
        if moved >= -0.08 and not self.done:  # stuck? tug once more before giving up (only jams pay this)
            moved = self._pull(drawer, firm=False)
        return self._record("open", drawer, "ok" if moved < -0.08 else "jam", self.log.steps - s0)

    def pull_hard(self, drawer: str) -> SkillEvent:
        s0 = self.log.steps
        if drawer != self.props.sticky_drawer:
            self.log.stale_actions += 1
        moved = self._pull(drawer, firm=True)
        return self._record("pull_hard", drawer, "ok" if moved < -0.08 else "jam", self.log.steps - s0)

    def _grasp_and_lift(self, obj: str, firm: bool) -> float:
        for d in DRAWERS:  # inside a closed drawer -> unreachable
            if d not in self.open_drawers and self._in_drawer(obj, d):
                return float("nan")
        p = self._obj_pos(obj)
        grasp = p + np.array([0, 0, 0.02])
        self._move(grasp + np.array([0, 0, 0.12]), grip=-1.0, max_steps=150)
        self._move(grasp, grip=-1.0, max_steps=80, tol=0.008)
        self._hold(1.0, 30 if firm else 6)  # close; firm = brace (costs more, but less than a drop)
        if np.linalg.norm(self._eef() - grasp) > REACH:
            return 0.0
        self._attach(OBJ_BODY[obj], FIRM_LIMIT if firm else GRIP_LIGHT_LIMIT)
        z0 = self._obj_pos(obj)[2]
        # lift force ~ commanded delta, so a slow lift cannot raise the heavy object: both
        # variants lift at full gain; firm differs in grasp limit + brace, not in speed
        self._move(self._eef() + np.array([0, 0, 0.15]), grip=1.0, gain=10.0,
                   max_steps=(140 if firm else 40), tol=0.01, max_delta=1.0)
        self._hold(1.0, 5)
        return self._obj_pos(obj)[2] - z0

    def pick(self, obj: str) -> SkillEvent:
        s0 = self.log.steps
        rose = self._grasp_and_lift(obj, firm=False)
        if rose != rose:  # nan: in a closed drawer
            return self._record("pick", obj, "not_here", self.log.steps - s0)
        ok = rose > 0.08 and self._active_eq is not None
        if not ok:
            self._detach()
        self.holding = obj if ok else None
        return self._record("pick", obj, "ok" if ok else "drop", self.log.steps - s0)

    def pick_firm(self, obj: str) -> SkillEvent:
        s0 = self.log.steps
        if obj != self.props.heavy_object:
            self.log.stale_actions += 1
        rose = self._grasp_and_lift(obj, firm=True)
        if rose != rose:
            return self._record("pick_firm", obj, "not_here", self.log.steps - s0)
        ok = rose > 0.08 and self._active_eq is not None
        if not ok:
            self._detach()
        self.holding = obj if ok else None
        return self._record("pick_firm", obj, "ok" if ok else "drop", self.log.steps - s0)

    pick_two_hand = pick_firm  # the shared planner's name for the robust pick

    def _in_drawer(self, obj: str, drawer: str) -> bool:
        p = self._obj_pos(obj)
        r = self._drawer_region_world(drawer)
        return abs(p[0] - r[0]) < 0.11 and abs(p[1] - r[1]) < 0.12 and abs(p[2] - r[2]) < 0.08

    def look_in(self, drawer: str) -> SkillEvent:
        """Open the drawer if needed (gentle), hover over its exposed section, check."""
        s0 = self.log.steps
        if drawer not in self.open_drawers:
            ev = self.open(drawer)
            if ev.outcome != "ok":
                return ev
        handle, open_dir = self._handle_world(drawer)
        spot = handle - open_dir * 0.085
        spot[2] = self._drawer_region_world(drawer)[2] + 0.18
        self._move(spot, grip=-1.0, max_steps=100)
        self._hold(-1.0, 10)  # "looking"
        here = self._in_drawer(self.task.obj, drawer)
        if not here:
            self.log.wasted_looks += 1
        return self._record("look_in", drawer, "found" if here else "empty", self.log.steps - s0)

    def close(self, drawer: str) -> SkillEvent:
        """Push an open drawer shut (magnetic hook on the handle, move it back in)."""
        s0 = self.log.steps
        if drawer not in self.open_drawers:
            return self._record("close", drawer, "ok", 0)
        handle, open_dir = self._handle_world(drawer)
        front, up = HOOK_OFFSET[drawer]
        hook = handle + open_dir * front + np.array([0, 0, up])
        self._move(hook + np.array([0, 0, 0.10]), grip=1.0, max_steps=120)
        self._move(hook, grip=1.0, max_steps=100, tol=0.008)
        self._hold(1.0, 6)
        if np.linalg.norm(self._eef() - hook) <= REACH:
            self._attach(DRAWER_BODY[drawer], FIRM_LIMIT)
            self._move(self._eef() - open_dir * 0.16, grip=1.0, gain=6.0, max_steps=90, tol=0.01, max_delta=0.8)
            self._detach()
        self._move(self._eef() + np.array([0, 0, 0.10]), grip=-1.0, max_steps=40)
        closed = self._drawer_qpos(drawer) > -0.03
        if closed:
            self.open_drawers.discard(drawer)
        return self._record("close", drawer, "ok" if closed else "stuck_open", self.log.steps - s0)

    def place_table(self, obj: str) -> SkillEvent:
        s0 = self.log.steps
        spot = np.array([-0.10, -0.05, 0.99])  # free patch of table in front of the robot
        self._move(spot + np.array([0, 0, 0.15]), grip=1.0, max_steps=150)
        self._move(spot, grip=1.0, max_steps=80, tol=0.01)
        self._detach()
        self._hold(-1.0, 8)
        self._move(self._eef() + np.array([0, 0, 0.12]), grip=-1.0, max_steps=60)
        ev = self._record("place", "table", "ok", self.log.steps - s0)
        p = self._obj_pos(obj)
        on_table = p[2] < 0.96 and not any(self._in_drawer(obj, d) for d in DRAWERS)
        self.log.success = bool(on_table and self.task.kind == "fetch" and self.log.steps <= self.step_budget)
        self.done = True
        return ev

    def place(self, obj: str, drawer: str) -> SkillEvent:
        if drawer == "table":
            return self.place_table(obj)
        s0 = self.log.steps
        # release in the EXPOSED front section of the open drawer (the interior centre is
        # under the drawer above it): ~8.5 cm behind the handle bar, just above the floor
        handle, open_dir = self._handle_world(drawer)
        floor_z = self._drawer_region_world(drawer)[2]
        spot = handle - open_dir * 0.085
        spot[2] = floor_z + 0.045
        self._move(spot + np.array([0, 0, 0.15]), grip=1.0, max_steps=150)
        self._move(spot, grip=1.0, max_steps=80, tol=0.01)
        self._detach()
        self._hold(-1.0, 8)
        self._move(self._eef() + np.array([0, 0, 0.15]), grip=-1.0, max_steps=60)
        ev = self._record("place", drawer, "ok", self.log.steps - s0)
        p = self._obj_pos(obj)
        r = self._drawer_region_world(drawer)
        # inside the drawer's exposed section: within its width in x, near the release
        # spot in y (front-to-back), on the floor, and the drawer actually open
        inside = (abs(p[0] - spot[0]) < 0.10 and abs(p[1] - spot[1]) < 0.07 and abs(p[2] - floor_z) < 0.06
                  and self._drawer_qpos(drawer) < -0.08)
        kind = getattr(self.task, "kind", "put")
        right_drawer = (kind == "put_any") or drawer == self.task.drawer
        self.log.success = bool(inside and right_drawer and self.log.steps <= self.step_budget)
        self.done = True
        return ev

    def shutdown(self) -> None:
        self.env.close()
