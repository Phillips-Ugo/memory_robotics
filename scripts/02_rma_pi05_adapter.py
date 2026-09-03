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


def _make_client(host: str, port: int):
    """openpi's client with keepalive pings disabled: the first request blocks for
    30-90 s while JAX/XLA compiles, and the default 20 s ping timeout kills the
    connection (same workaround as RoboMemArena's StableWebsocketClientPolicy)."""
    import logging
    import time

    import websockets.sync.client
    from openpi_client import msgpack_numpy, websocket_client_policy

    class _Stable(websocket_client_policy.WebsocketClientPolicy):
        def _wait_for_server(self):
            logging.info(f"Waiting for server at {self._uri}...")
            while True:
                try:
                    conn = websockets.sync.client.connect(
                        self._uri, compression=None, max_size=None,
                        ping_interval=None, ping_timeout=None, close_timeout=30.0,
                    )
                    return conn, msgpack_numpy.unpackb(conn.recv())
                except ConnectionRefusedError:
                    logging.info("Still waiting for server...")
                    time.sleep(5)

    return _Stable(host=host, port=port)


class Pi05WebsocketAdapter(BasePolicyAdapter):
    """Forwards harness observations to an openpi policy server over websocket."""

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self.host = host or os.environ.get("PI05_HOST", "localhost")
        self.port = int(port or os.environ.get("PI05_PORT", "8000"))
        # The harness flips images vertically (its own data-recorder convention);
        # openpi's LIBERO example rotates 180°. PI05_FLIPLR=1 adds the missing
        # horizontal flip so the stock pi05_libero checkpoint sees its training view.
        self.fliplr = os.environ.get("PI05_FLIPLR", "0") == "1"
        self._client = None  # connect lazily so adapter load doesn't block
        self._calls = 0

    def _ensure_client(self):
        if self._client is None:
            print(f"[pi05-adapter] connecting to ws://{self.host}:{self.port} ...", flush=True)
            self._client = _make_client(self.host, self.port)
            print(f"[pi05-adapter] connected: {self._client.get_server_metadata()}", flush=True)
        return self._client

    def reset(self) -> None:
        self._calls = 0
        client = self._client
        if client is not None and hasattr(client, "reset"):
            client.reset()

    def infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        img, wrist = obs["observation/image"], obs["observation/wrist_image"]
        if self.fliplr:
            img, wrist = np.ascontiguousarray(img[:, ::-1]), np.ascontiguousarray(wrist[:, ::-1])
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        element = {
            "observation/image": img,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": str(prompt),
        }
        actions = np.asarray(self._ensure_client().infer(element)["actions"], dtype=np.float32)
        self._calls += 1
        if self._calls in (1, 50, 150):  # diagnostic: is the policy commanding motion at all?
            print(
                f"[pi05-adapter] call {self._calls}: img{img.shape} state={np.round(state, 3)} "
                f"chunk{actions.shape} mean|a[:6]|={np.abs(actions[:, :6]).mean():.3f} "
                f"grip={actions[:, 6].mean():+.2f}",
                flush=True,
            )
        return actions


def build_adapter(**kwargs: Any) -> "BasePolicyAdapter":
    return Pi05WebsocketAdapter(**kwargs)
