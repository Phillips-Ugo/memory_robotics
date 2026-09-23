"""X5: memlayer in the loop of a real VLA on RoboMemArena task 1.

Wraps the pi05 websocket adapter and decides, per stage, which prompt the policy sees:

  mode=fixed      the harness's full-task prompt for the whole episode (today's baseline, 15/51)
  mode=primitive  the training primitives for every stage ("pick cookies" -> "place cookies into
                  basket" -> ...), switching pick->place on gripper closure: the oracle-planner arm
  mode=memory     starts as `fixed`; memlayer ingests every episode's per-stage outcomes and, once a
                  stage is a known `trouble:<verb>@fixed` fact, that stage is prompted with primitives.
                  Facts are keyed by strategy so a fix never erases the evidence that motivated it.

Stage progress inside an episode comes from the harness's own stage checks (adapter.on_stage_done,
see scripts/patch_rma_adapter_hooks.py) — i.e. within-episode progress is given (Problem A), what is
learned across episodes is *whether this stage needs help* (Problem B).

    --adapter-spec scripts/03_rma_memory_adapter.py:build_adapter --adapter-kwargs "mode=memory,db=/workspace/x5_memory.db"
Env: X5_LOG=<path.jsonl> to record the prompt schedule per episode; X5_INNER=dummy for a policy-free plumbing test.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_HARNESS = _REPO_ROOT / "vendor/RoboMemArena/evaluation_benchmark/scripts"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))
try:
    from policy_adapter import BasePolicyAdapter  # the harness insists on a subclass
except ImportError:  # importing outside the harness (tests)
    BasePolicyAdapter = object

from memlayer.core import MemoryLayer  # noqa: E402
from memlayer.strategy import StageOutcome, StrategyMemory  # noqa: E402

# harness stage name -> (pick primitive, place primitive) exactly as the training filenames spell them
PRIMITIVES = {
    "01_Place_Cookies_Basket": ("pick cookies", "place cookies into basket"),
    "02_Place_Tomato_Basket": ("pick tomato sauce", "place tomato into basket"),
}


def _load_inner(host, port):
    if os.environ.get("X5_INNER") == "dummy":
        class Dummy:
            def reset(self): pass
            def infer_actions(self, obs, prompt, resize_size):
                return np.zeros((10, 7), dtype=np.float32)
        return Dummy()
    spec = importlib.util.spec_from_file_location("pi05_adapter", _REPO_ROOT / "scripts/02_rma_pi05_adapter.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.Pi05WebsocketAdapter(host=host, port=port)


class MemoryPromptAdapter(BasePolicyAdapter):
    def __init__(self, mode: str = "memory", db: str = "outputs/x5_memory.db", host=None, port=None,
                 grasp_width: float = 0.065, grasp_hold: int = 5, open_width: float = 0.072,
                 release_gate: bool = True, scope: str = "stage", stage_names: str = "") -> None:
        assert mode in ("fixed", "primitive", "memory"), mode
        self.mode = mode
        self.inner = _load_inner(host, port)
        self.stage_names = [s for s in stage_names.split("|") if s] or list(PRIMITIVES)
        self.grasp_width, self.grasp_hold = float(grasp_width), int(grasp_hold)
        self.open_width, self.release_gate = float(open_width), bool(release_gate)
        assert scope in ("stage", "episode"), scope
        self.scope = scope  # episode: once any stage is a trouble fact, run the whole episode on primitives
        self.sm = StrategyMemory(MemoryLayer(db), default="fixed", fallback="primitive", scope=scope) if mode == "memory" else None
        self.log_path = os.environ.get("X5_LOG")
        self.ep = 0
        self._task_prompt = None
        self._reset_episode_state()

    # ---- harness hooks -------------------------------------------------------------
    def reset(self) -> None:
        self.inner.reset()
        self._reset_episode_state()
        self._plan = self.sm.plan(self.stage_names) if self.sm else {name: self.mode for name in self.stage_names}

    def on_stage_done(self, name: str, t: int) -> None:
        self._done.append(name)
        self._closed = 0
        self._holding = False
        # the stage check fires when the object is in place even if it is still in the gripper
        # (the full-task prompt tends to keep holding): a new grasp only counts after a release
        self._needs_open = bool(self.release_gate)

    def on_episode_end(self, ep_summary: dict) -> None:
        prev_t, outcomes = 0, []
        for name, passed in ep_summary["stage_done"].items():
            steps = None
            if passed and name in ep_summary.get("stage_steps", {}):
                steps = int(ep_summary["stage_steps"][name]) - prev_t; prev_t = int(ep_summary["stage_steps"][name])
            outcomes.append(StageOutcome(name, bool(passed), steps))
        if self.sm is not None:
            self.sm.record(self.ep, outcomes, self._plan, task=self._task_prompt or "task1",
                           total_steps=ep_summary.get("total_steps"))
        if self.log_path:
            rec = {"ep": self.ep, "mode": self.mode, "plan": self._plan, "prompts_used": self._prompts_used,
                   "TSR": ep_summary.get("TSR"), "CSR": ep_summary.get("CSR"), "stage_done": ep_summary.get("stage_done"),
                   "recall": self.sm.explain(self.stage_names) if self.sm else ""}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        self.ep += 1

    def infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        self._task_prompt = prompt
        use = self._choose_prompt(obs, prompt)
        if not self._prompts_used or self._prompts_used[-1][1] != use:
            self._prompts_used.append((len(self._prompts_used), use))
        return self.inner.infer_actions(obs=obs, prompt=use, resize_size=resize_size)

    # ---- policy over prompts -------------------------------------------------------
    def _reset_episode_state(self):
        self._done, self._closed, self._holding, self._prompts_used = [], 0, False, []
        self._needs_open = False
        self._plan = getattr(self, "_plan", {})

    def _current_stage(self) -> str | None:
        for name in self.stage_names:
            if name not in self._done:
                return name
        return None

    def _choose_prompt(self, obs, task_prompt: str) -> str:
        stage = self._current_stage()
        if stage is None or self._plan.get(stage, "fixed") == "fixed" or stage not in PRIMITIVES:
            return task_prompt
        pick, place = PRIMITIVES[stage]
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        width = float(abs(state[6] - state[7])) if state.shape[0] >= 8 else 1.0
        if self._needs_open:
            if width >= self.open_width:
                self._needs_open = False
            return pick
        if width < self.grasp_width:
            self._closed += 1
        else:
            self._closed = 0
            if self._holding:
                self._holding = False  # dropped it: back to picking
        if self._closed >= self.grasp_hold:
            self._holding = True
        return place if self._holding else pick

    def close(self) -> None:
        c = getattr(self.inner, "close", None)
        if callable(c):
            c()


def build_adapter(**kwargs: Any) -> MemoryPromptAdapter:
    return MemoryPromptAdapter(**kwargs)
