"""Phase 1, M2: RoboMemArena adapter that queries a pi-0.5 openpi policy server.

Same plug point as the M1 dummy adapter, but instead of zeros it forwards the
harness observation to an openpi websocket policy server (serve_policy.py
--env LIBERO, which loads the pi05_libero checkpoint) and returns the action
chunk. The harness already builds the exact element openpi's LIBERO example
expects — flipud+resized uint8 images and eef_pos(3)+axis_angle(3)+gripper
state — so this adapter is a thin pass-through (see
vendor/RoboMemArena/evaluation_benchmark/scripts/policy_adapter.py,
build_eval26_policy_input).

Server (GPU box, openpi venv):
    cd vendor/openpi && uv run scripts/serve_policy.py --env LIBERO

Client (harness venv):
    cd vendor/RoboMemArena/evaluation_benchmark
    MUJOCO_GL=egl ../../rma-venv/bin/python scripts/eval_task1_only.py \
        --adapter-spec /abs/path/to/scripts/02_rma_pi05_adapter.py:build_adapter \
        --num-trials-per-task 1

Config via env vars: PI05_HOST (default localhost), PI05_PORT (default 8000).
Client deps in the harness venv: websockets, msgpack, typing_extensions
(installed by scripts/setup_gpu_box.sh); openpi-client itself is imported from
the vendored copy inside vendor/RoboMemArena.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Vendored openpi-client (pure client: websockets + msgpack, no JAX).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CLIENT_SRC = (
    _REPO_ROOT
    / "vendor/RoboMemArena/third_party/openpi_minimal/packages/openpi-client/src"
)
if _CLIENT_SRC.exists() and str(_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_CLIENT_SRC))

try:
    from policy_adapter import BasePolicyAdapter
except ImportError:  # allow importing outside the harness for linting
    BasePolicyAdapter = object


class Pi05WebsocketAdapter(BasePolicyAdapter):
    """Forwards harness observations to an openpi policy server over websocket."""

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self.host = host or os.environ.get("PI05_HOST", "localhost")
        self.port = int(port or os.environ.get("PI05_PORT", "8000"))
        self._client = None  # connect lazily so adapter load doesn't block

    def _ensure_client(self):
        if self._client is None:
            from openpi_client import websocket_client_policy

            print(f"[pi05-adapter] connecting to ws://{self.host}:{self.port} ...", flush=True)
            self._client = websocket_client_policy.WebsocketClientPolicy(
                host=self.host, port=self.port
            )
            print(f"[pi05-adapter] connected: {self._client.get_server_metadata()}", flush=True)
        return self._client

    def reset(self) -> None:
        client = self._client
        if client is not None and hasattr(client, "reset"):
            client.reset()

    def infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        element = {
            "observation/image": obs["observation/image"],
            "observation/wrist_image": obs["observation/wrist_image"],
            "observation/state": np.asarray(obs["observation/state"], dtype=np.float32),
            "prompt": str(prompt),
        }
        actions = self._ensure_client().infer(element)["actions"]
        return np.asarray(actions, dtype=np.float32)


def build_adapter(**kwargs: Any) -> "BasePolicyAdapter":
    return Pi05WebsocketAdapter(**kwargs)
