"""X6: the learned stage detector (Jev-R v0) replaces the harness's oracle stage checks inside the X5 loop.

Wraps MemoryPromptAdapter (scripts/03_rma_memory_adapter.py). With detector="learned" the adapter ignores the
harness's on_stage_done hook and instead runs DINOv2-S on the agent + wrist views every `every` env steps, keeps a
window of the last 3 samples, applies the exported logistic heads (jev/heads_task1.npz) with their operating
points (p >= thr for k consecutive samples), and fires stage completion itself — sequentially (stage 2 is only
considered after stage 1 fired). The harness still scores TSR/CSR with its own checks, so the number is honest.

    --adapter-spec scripts/04_rma_jevr_adapter.py:build_adapter --adapter-kwargs '{"mode": "memory", "scope": "episode", "detector": "learned"}'
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("mem_adapter", _REPO_ROOT / "scripts/03_rma_memory_adapter.py")
_mem = importlib.util.module_from_spec(spec); spec.loader.exec_module(_mem)

STAGES = ["01_Place_Cookies_Basket", "02_Place_Tomato_Basket"]


class StageDetector:
    """Frozen DINOv2-S (agent + wrist) -> exported logistic heads -> sustained-positive decision."""

    def __init__(self, heads: str, every: int = 10, device: str | None = None) -> None:
        import torch
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval().to(self.dev)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)
        h = np.load(heads)
        self.window = int(h["window"]); self.every = every
        self.f_mean, self.f_scale = h["mean"].astype(np.float32), h["scale"].astype(np.float32)
        self.heads = {q: (h[f"{q}_W"].astype(np.float32), float(h[f"{q}_b"][0]), float(h[f"{q}_thr"]), int(h[f"{q}_k"])) for q in ("stage_done_1", "stage_done_2")}
        self.ms: list[float] = []
        self.reset()

    def reset(self) -> None:
        self.hist: list[np.ndarray] = []
        self.runs = {q: 0 for q in self.heads}
        self.probs = {q: 0.0 for q in self.heads}

    def _embed(self, img: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            x = t.from_numpy(np.stack([img, wrist])).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
            x = t.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            f = self.model((x - self.mean) / self.std).float().cpu().numpy()
        return np.concatenate([f[0], f[1]])

    def step(self, img: np.ndarray, wrist: np.ndarray, stages_done: int) -> str | None:
        """Call every `every` env steps. Returns the name of a stage that just completed, or None."""
        t0 = time.perf_counter()
        z = self._embed(img, wrist)
        self.hist.append(z); self.hist = self.hist[-self.window:]
        pad = [self.hist[0]] * (self.window - len(self.hist)) + self.hist
        x = (np.concatenate(pad) - self.f_mean) / self.f_scale
        fired = None
        q = f"stage_done_{stages_done + 1}"
        if q in self.heads:
            W, b, thr, k = self.heads[q]
            with np.errstate(all="ignore"):  # Accelerate BLAS on macOS emits spurious matmul warnings
                p = 1.0 / (1.0 + np.exp(-(float(np.dot(x, W)) + b)))
            self.probs[q] = float(p)
            self.runs[q] = self.runs[q] + 1 if p >= thr else 0
            if self.runs[q] >= k:
                fired = STAGES[stages_done]; self.runs[q] = 0
        self.ms.append((time.perf_counter() - t0) * 1000)
        return fired


class JevRAdapter(_mem.MemoryPromptAdapter):
    def __init__(self, detector: str = "learned", heads: str = str(_REPO_ROOT / "jev/heads_task1.npz"), every: int = 10, **kw) -> None:
        super().__init__(**kw)
        assert detector in ("learned", "oracle"), detector
        self.detector = detector
        self.det = StageDetector(heads, every) if detector == "learned" else None
        self._t = 0
        self._learned_done: list[str] = []
        self._oracle_events: list[tuple[str, int]] = []
        self._learned_events: list[tuple[str, int]] = []

    def reset(self) -> None:
        super().reset()
        self._t = 0; self._learned_done = []; self._oracle_events = []; self._learned_events = []
        if self.det:
            self.det.reset()

    # the harness's own stage checks: record for diagnostics, but only act on them in oracle mode
    def on_stage_done(self, name: str, t: int) -> None:
        self._oracle_events.append((name, t))
        if self.detector == "oracle":
            super().on_stage_done(name, t)

    def infer_actions(self, obs: dict[str, Any], prompt: str, resize_size: int) -> np.ndarray:
        if self.det and self._t % self.det.every == 0 and len(self._learned_done) < len(STAGES):
            fired = self.det.step(obs["observation/image"], obs["observation/wrist_image"], len(self._learned_done))
            if fired:
                self._learned_done.append(fired); self._learned_events.append((fired, self._t))
                super().on_stage_done(fired, self._t)
        self._t += 1
        return super().infer_actions(obs, prompt, resize_size)

    def on_episode_end(self, ep_summary: dict) -> None:
        ep_summary = dict(ep_summary)
        ep_summary["oracle_events"] = list(self._oracle_events)
        ep_summary["learned_events"] = list(self._learned_events)
        if self.det:
            ep_summary["detector_ms_mean"] = float(np.mean(self.det.ms)) if self.det.ms else None
        super().on_episode_end(ep_summary)
        if self.log_path:  # append the detector diagnostics as a second line
            import json
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"ep": self.ep - 1, "oracle_events": self._oracle_events, "learned_events": self._learned_events,
                                    "detector_ms_mean": ep_summary.get("detector_ms_mean")}) + "\n")


def build_adapter(**kwargs: Any) -> JevRAdapter:
    return JevRAdapter(**kwargs)
