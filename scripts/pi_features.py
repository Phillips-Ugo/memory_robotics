"""Extract π0.5's own prefix (vision-language) features for frames, to test "does the policy's representation
already know phase / progress / subtask-done?" (SAFE-style). Runs in the openpi venv on a GPU box.

    uv run python /workspace/memory_robotics/scripts/pi_features.py --config pi05_rma_lora --ckpt /workspace/ckpt/7999 \
        --mode rollouts --root /workspace/rollouts --every 10 --out /workspace/feats/rollouts_pi05_t1.npz
    uv run python /workspace/memory_robotics/scripts/pi_features.py --config pi05_libero --ckpt <base dir> \
        --mode rpb --rpb /workspace/data/RoboProcessBench --gm /workspace/data/gm100 --out /workspace/feats/rpb_pi05_base.npz

Feature per frame = mean over image tokens of the PaliGemma last-layer hidden states after the prefix pass
(the same pass sample_actions runs before the action expert), plus the mean over language tokens.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def load_policy(config_name: str, ckpt: str):
    from openpi.policies import policy_config
    from openpi.training import config as _config
    cfg = _config.get_config(config_name)
    return policy_config.create_trained_policy(cfg, ckpt)


_JIT = {}


def _get_jit(policy, batch_size: int):
    """A jitted prefix pass (embed_prefix + PaliGemma LLM) with fixed batch size, built once per policy."""
    key = (id(policy), batch_size)
    if key in _JIT:
        return _JIT[key]
    from flax import nnx
    from openpi.models import model as _model
    from openpi.models import pi0 as _pi0
    model = policy._model
    graphdef, state = nnx.split(model)

    def f(state, inputs):
        m = nnx.merge(graphdef, state)
        obs = _model.Observation.from_dict(inputs)
        tokens, mask, ar_mask = m.embed_prefix(obs)
        attn = _pi0.make_attn_mask(mask, ar_mask)
        positions = jnp.cumsum(mask, axis=1) - 1
        (out, _), _ = m.PaliGemma.llm([tokens, None], mask=attn, positions=positions)
        return out, mask

    jf = jax.jit(f)
    _JIT[key] = (jf, state)
    return _JIT[key]


def prefix_features(policy, images: np.ndarray, wrists: np.ndarray, states: np.ndarray, prompts: list[str]) -> np.ndarray:
    """images/wrists: (B,256,256,3) uint8, states (B,8). Returns (B, 2*D) [mean image tokens | mean language tokens]."""
    B = len(images); BS = 16
    batch = []
    for i in range(B):
        obs = {"observation/image": images[i], "observation/wrist_image": wrists[i], "observation/state": states[i].astype(np.float32), "prompt": prompts[i]}
        batch.append(policy._input_transform(obs))
    while len(batch) < BS:  # pad to the jit batch size
        batch.append(batch[-1])
    inputs = {k: (np.stack([b[k] for b in batch]) if not isinstance(batch[0][k], dict) else {kk: np.stack([b[k][kk] for b in batch]) for kk in batch[0][k]}) for k in batch[0]}
    jf, state = _get_jit(policy, BS)
    out, mask = jf(state, jax.tree.map(jnp.asarray, inputs))
    out = np.asarray(out, dtype=np.float32); m = np.asarray(mask)
    n_lang = int(np.asarray(inputs["tokenized_prompt_mask"]).sum(1).max()) if "tokenized_prompt_mask" in inputs else 0
    feats = []
    for b in range(B):
        valid = out[b][m[b].astype(bool)]
        n_l = int(np.asarray(inputs["tokenized_prompt_mask"][b]).sum()) if n_lang else 0
        img = valid[:-n_l] if n_l else valid; lang = valid[-n_l:] if n_l else valid[:0]
        feats.append(np.concatenate([img.mean(0), lang.mean(0) if len(lang) else np.zeros(img.shape[1], np.float32)]))
    return np.stack(feats)


def _make_attn(mask, ar_mask):
    from openpi.models import pi0 as _pi0
    return _pi0.make_attn_mask(mask, ar_mask)


def read_frames(path: str, idxs: set[int]) -> dict[int, np.ndarray]:
    import imageio.v3 as iio
    out = {}
    for t, f in enumerate(iio.imiter(path)):
        if t in idxs:
            out[t] = f
        if t >= max(idxs):
            break
    return out


def run_rollouts(policy_loader, root: str, every: int, prompt: str, out: str, batch: int) -> None:
    S1, S2 = "01_Place_Cookies_Basket", "02_Place_Tomato_Basket"
    rows, imgs, wrs = [], [], []
    for src in sorted(glob.glob(f"{root}/*/results.json")):
        d = Path(src).parent; r = json.load(open(src))
        for e in r["episodes"]:
            vids = [v for v in glob.glob(f"{d}/task1_*_ep{e['ep']}_seed{e['seed']}.mp4") if "wrist" not in v]
            if not vids:
                continue
            n = e["total_steps"]; idx = set(range(0, n, every))
            A = read_frames(vids[0], idx); W = read_frames(vids[0].replace(".mp4", "_wrist.mp4"), idx)
            for t in sorted(idx):
                if t not in A or t not in W:
                    continue
                s1 = e["stage_steps"].get(S1); s2 = e["stage_steps"].get(S2)
                rows.append({"src": d.name, "ep": e["ep"], "seed": e["seed"], "step": t, "stage_done_1": int(s1 is not None and t >= s1), "stage_done_2": int(s2 is not None and t >= s2)})
                imgs.append(A[t]); wrs.append(W[t])
        print(f"  {d.name}: {len(rows)} frames so far", flush=True)
    policy = policy_loader()
    F = []; t0 = time.time()
    for i in range(0, len(rows), batch):
        F.append(prefix_features(policy, np.stack(imgs[i:i + batch]), np.stack(wrs[i:i + batch]), np.zeros((len(imgs[i:i + batch]), 8), np.float32), [prompt] * len(imgs[i:i + batch])))
        if (i // batch) % 20 == 0:
            print(f"  features {i}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
    np.savez(out, F=np.concatenate(F).astype(np.float16), rows=np.array([json.dumps(r) for r in rows])); print("wrote", out, np.concatenate(F).shape)


def run_rpb(policy_loader, rpb: str, gm: str, out: str, batch: int) -> None:
    from collections import defaultdict
    rows = [json.loads(l) for f in ("eval", "sft") for l in open(f"{rpb}/splits/processdata_{f}.jsonl")]
    rows = [r for r in rows if r["source"] == "GM-100"]
    def vp(r): return f"{gm}/{r['source_task_id']}/videos/chunk-000/observation.images.camera_top/{r['source_unit_id'].split('__')[1]}.mp4"
    want = defaultdict(set); qs = {}
    for r in rows:
        for i in json.loads(r["frame_indices_json"]):
            want[vp(r)].add(int(i)); qs.setdefault((vp(r), int(i)), r["question"] or "perform the manipulation task")
    keys, imgs, prompts = [], [], []
    for k, (p, idxs) in enumerate(want.items()):
        if not Path(p).exists():
            continue
        fr = read_frames(p, idxs)
        for t, f in fr.items():
            keys.append((p, t)); imgs.append(f); prompts.append(qs[(p, t)])
        if k % 200 == 0:
            print(f"  frames: {k}/{len(want)} videos, {len(keys)} frames", flush=True)
    policy = policy_loader()
    F = []; t0 = time.time()
    for i in range(0, len(keys), batch):
        im = np.stack([_resize(x) for x in imgs[i:i + batch]])
        F.append(prefix_features(policy, im, np.zeros_like(im), np.zeros((len(im), 8), np.float32), prompts[i:i + batch]))
        if (i // batch) % 20 == 0:
            print(f"  features {i}/{len(keys)} {time.time()-t0:.0f}s", flush=True)
    np.savez(out, F=np.concatenate(F).astype(np.float16), keys=np.array([f"{p}|{t}" for p, t in keys])); print("wrote", out)


def _resize(x: np.ndarray, size: int = 256) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray(x).resize((size, size), Image.BILINEAR))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["rollouts", "rpb"], required=True)
    ap.add_argument("--root", default="/workspace/rollouts"); ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--prompt", default="Pick and place cookies into the basket, then pick and place tomato sauce into the same basket.")
    ap.add_argument("--rpb", default="/workspace/data/RoboProcessBench"); ap.add_argument("--gm", default="/workspace/data/gm100")
    ap.add_argument("--out", required=True); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    loader = lambda: load_policy(a.config, a.ckpt)  # JAX initialises only after all video decoding is done
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    (run_rollouts(loader, a.root, a.every, a.prompt, a.out, a.batch) if a.mode == "rollouts" else run_rpb(loader, a.rpb, a.gm, a.out, a.batch))
