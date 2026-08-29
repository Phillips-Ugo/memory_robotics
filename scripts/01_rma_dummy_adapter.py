"""Phase 1, M1: a dummy policy adapter for the RoboMemArena eval harness.

Proves the sim stack + adapter contract end to end with zero model weights:
the "policy" holds the arm still (LIBERO dummy action) with optional small
random jitter. It will score ~0% — the point is that the harness runs, renders,
and scores at all. This same adapter file is where a real policy (and later, a
memory layer) plugs in.

Usage (uses the dedicated vendor/rma-venv, NOT the main .venv):
    cd vendor/RoboMemArena/evaluation_benchmark
    ../../rma-venv/bin/python scripts/eval_task1_only.py \
        --adapter-spec /abs/path/to/scripts/01_rma_dummy_adapter.py:build_adapter \
        --num-trials 2 --seed 0
(see scripts/eval_task1_only.py --help for exact flags; run_all_tasks1_26.py
takes --adapter-spec the same way)
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from policy_adapter import BasePolicyAdapter
except ImportError:  # allow importing this file outside the harness for linting
    BasePolicyAdapter = object


class DummyAdapter(BasePolicyAdapter):
    """Returns near-zero action chunks: 6 delta-pose dims + gripper closed."""

    def __init__(self, horizon: int = 8, jitter: float = 0.0, seed: int = 0) -> None:
        self.horizon = horizon
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        return None

    def infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        chunk = np.zeros((self.horizon, 7), dtype=np.float32)
        chunk[:, 6] = -1.0  # gripper open (LIBERO convention)
        if self.jitter > 0:
            chunk[:, :6] += self.rng.normal(0, self.jitter, size=(self.horizon, 6)).astype(np.float32)
        return chunk


def build_adapter(**kwargs: Any) -> "BasePolicyAdapter":
    return DummyAdapter(**kwargs)
