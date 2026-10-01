"""RoboProcessBench (GM-100 source) with a non-generative decision model.

Apples-to-apples with the released ProcessData-SFT baselines: train on the SFT rows, evaluate on the eval rows,
same multiple-choice items, accuracy per task family. Our model never generates: frames -> frozen DINOv2-S (one
vector per frame, ordered), question + options -> frozen sentence encoder (bge-small), a small list-wise scorer
picks the option. Report accuracy, parameters and ms/item next to the Qwen2.5-VL-7B LoRA numbers.

    uv run python -m jev.rpb build     # extract referenced frames -> features (needs data/gm100 videos)
    uv run python -m jev.rpb train     # train the scorer on SFT, evaluate on eval, write results json
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("data/RoboProcessBench"); GM = Path("data/gm100"); OUT = Path("outputs/rpb"); OUT.mkdir(parents=True, exist_ok=True)
LETTERS = "ABCDEF"


def gm_rows(split: str) -> list[dict]:
    return [r for r in (json.loads(l) for l in open(ROOT / f"splits/processdata_{split}.jsonl")) if r["source"] == "GM-100"]


def video_path(r: dict) -> Path:
    ep = r["source_unit_id"].split("__")[1]
    return GM / r["source_task_id"] / "videos/chunk-000/observation.images.camera_top" / f"{ep}.mp4"


def build() -> None:
    import imageio.v3 as iio
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    vit = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    txt = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)
    for split in ("eval", "sft"):
        rows = gm_rows(split)
        rows = [r for r in rows if video_path(r).exists()]
        # group frame requests by video
        want: dict[Path, set[int]] = defaultdict(set)
        for r in rows:
            for i in json.loads(r["frame_indices_json"]):
                want[video_path(r)].add(int(i))
        feats: dict[tuple[str, int], np.ndarray] = {}
        t0 = time.time(); n_frames = 0
        for vi, (vp, idxs) in enumerate(want.items()):
            frames = {}
            try:
                for t, f in enumerate(iio.imiter(vp)):
                    if t in idxs:
                        frames[t] = f
                    if t >= max(idxs):
                        break
            except Exception as e:
                print("video error", vp, e); continue
            if not frames:
                continue
            ts = sorted(frames)
            with torch.no_grad():
                x = torch.from_numpy(np.stack([frames[t] for t in ts])).to(dev).permute(0, 3, 1, 2).float() / 255.0
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                z = vit((x - mean) / std).float().cpu().numpy()
            for t, zi in zip(ts, z):
                feats[(str(vp), t)] = zi
            n_frames += len(ts)
            if vi % 100 == 0:
                print(f"  {split}: {vi}/{len(want)} videos, {n_frames} frames, {time.time()-t0:.0f}s", flush=True)
        # per item: ordered frame features (pad to 6), question/option text embeddings
        V, Q, O, M, meta = [], [], [], [], []
        qs = [r["question"] for r in rows]; opts = [[r[f"choice_{c}"] or "" for c in LETTERS] for r in rows]
        qe = txt.encode(qs, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        oe = txt.encode([o for os_ in opts for o in os_], batch_size=256, normalize_embeddings=True, show_progress_bar=False).reshape(len(rows), 6, -1)
        keep = []
        for k, r in enumerate(rows):
            idx = [int(i) for i in json.loads(r["frame_indices_json"])]
            vp = str(video_path(r))
            if any((vp, i) not in feats for i in idx):
                continue
            fr = np.stack([feats[(vp, i)] for i in idx])
            pad = np.zeros((6, fr.shape[1]), np.float32); pad[:len(fr)] = fr[:6]
            V.append(pad); Q.append(qe[k]); O.append(oe[k]); M.append([1 if (r[f"choice_{c}"] or "") else 0 for c in LETTERS]); keep.append(k)
        np.savez(OUT / f"{split}.npz", V=np.stack(V), Q=np.stack(Q), O=np.stack(O), M=np.array(M), y=np.array([LETTERS.index(rows[k]["answer"]) for k in keep]),
                 task=np.array([rows[k]["task_id"] for k in keep]), nframes=np.array([int(rows[k]["num_frames"]) for k in keep]), item=np.array([rows[k]["item_id"] for k in keep]),
                 unit=np.array([rows[k]["source_unit_id"] for k in keep]))
        print(f"{split}: {len(keep)}/{len(rows)} items with all frames -> {OUT/f'{split}.npz'}")


class Scorer:
    """List-wise option scorer: s(option) = MLP([visual_ctx, clip_change, frame_deltas, q, o, o*q]) with a tiny
    question-conditioned attention over the ordered frames. deltas=True adds the 5 consecutive frame differences
    (motion direction / ordering cues)."""

    def __init__(self, dv=384, dt=384, h=256, seed=0, deltas=False):
        import torch, torch.nn as nn
        torch.manual_seed(seed)
        self.deltas = deltas
        din = dv * 2 + (dv * 5 if deltas else 0) + dt * 2 + dt
        self.net = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.frame_pos = nn.Parameter(torch.zeros(6, dv)); self.q2v = nn.Linear(dt, dv)
        self.params = list(self.net.parameters()) + [self.frame_pos] + list(self.q2v.parameters())

    def forward(self, V, Q, O, M, nf):
        import torch
        B = V.shape[0]
        maskf = (torch.arange(6, device=V.device)[None, :] < nf[:, None]).float()  # valid frames
        Vp = V + self.frame_pos[None]
        att = torch.einsum("bfd,bd->bf", Vp, self.q2v(Q)) / 19.6
        att = att.masked_fill(maskf == 0, -1e9).softmax(-1)
        ctx = torch.einsum("bf,bfd->bd", att, Vp)                      # question-attended frame context
        first_last = (Vp[:, 0] - Vp[torch.arange(B), (nf - 1).clamp(min=0)])  # change over the clip (progress/order cues)
        q6 = Q[:, None, :].expand(-1, 6, -1); c6 = ctx[:, None, :].expand(-1, 6, -1); fl6 = first_last[:, None, :].expand(-1, 6, -1)
        parts = [c6, fl6]
        if self.deltas:
            d = (V[:, 1:] - V[:, :-1]) * (maskf[:, 1:, None])            # (B,5,dv), zero beyond the clip
            parts.append(d.reshape(B, -1)[:, None, :].expand(-1, 6, -1))
        x = torch.cat(parts + [q6, O, O * q6], -1)
        s = self.net(x).squeeze(-1)
        return s.masked_fill(M == 0, -1e9)


def train() -> None:
    import torch
    deltas = "--deltas" in sys.argv
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tr = np.load(OUT / "sft.npz"); te = np.load(OUT / "eval.npz")
    # validation = 10 % of SFT *episodes* (never eval items): epoch selection happens here, eval is scored once
    units = np.array(sorted(set(tr["unit"]))); rng = np.random.default_rng(0); val_units = set(rng.choice(units, size=len(units) // 10, replace=False))
    isval = np.array([u in val_units for u in tr["unit"]])
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    pack = lambda d, m: (T(d["V"][m]), T(d["Q"][m]), T(d["O"][m]), T(d["M"][m]), T(d["y"][m], torch.long), T(d["nframes"][m], torch.long))
    Vtr, Qtr, Otr, Mtr, ytr, ntr = pack(tr, ~isval); Vva, Qva, Ova, Mva, yva, nva = pack(tr, isval)
    Vte, Qte, Ote, Mte, yte, nte = pack(te, np.ones(len(te["y"]), bool))
    model = Scorer(deltas=deltas); [p.to(dev) for p in model.params]; model.net.to(dev); model.q2v.to(dev); model.frame_pos.data = model.frame_pos.data.to(dev)
    opt = torch.optim.AdamW(model.params, lr=3e-4, weight_decay=0.05)
    N = len(ytr); bs = 256; best = (0, None, -1)
    print(f"train {N} / val {len(yva)} / eval {len(yte)} items; deltas={deltas}")
    for ep in range(30):
        perm = torch.randperm(N, device=dev); model.net.train()
        for i in range(0, N, bs):
            b = perm[i:i + bs]
            loss = torch.nn.functional.cross_entropy(model.forward(Vtr[b], Qtr[b], Otr[b], Mtr[b], ntr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.net.eval()
        with torch.no_grad():
            vacc = (model.forward(Vva, Qva, Ova, Mva, nva).argmax(-1) == yva).float().mean().item()
            pred = model.forward(Vte, Qte, Ote, Mte, nte).argmax(-1).cpu().numpy()
        if vacc > best[0]:
            best = (vacc, pred.copy(), ep)
    print(f"selected epoch {best[2]} by validation acc {best[0]:.3f}")
    pred = best[1]; tasks = te["task"]
    qwen = json.load(open(ROOT / "qwen_lora_per_task.json"))
    dist = {l.split(",")[0]: l.strip().split(",") for l in open(ROOT / "metadata/task_distribution.csv") if l.startswith("T")}
    print("\nGM-100 eval, accuracy per task:  ours (DINOv2-S + bge-small + 0.6M scorer) | Qwen2.5-VL-7B LoRA | random | majority")
    res = {}
    for t in sorted(set(tasks), key=lambda s: int(s[1:])):
        m = tasks == t; a = float((pred[m] == te["y"][m]).mean()); q = qwen[t]["gm100"]
        res[t] = {"ours": a, "n": int(m.sum()), "qwen_lora": q[0] / q[1] if q else None, "random": float(dist[t][4]), "majority": float(dist[t][5])}
        print(f"  {t} (n={m.sum()}): {100*a:5.1f}% | {100*q[0]/q[1]:5.1f}% | {100*float(dist[t][4]):4.1f}% | {100*float(dist[t][5]):4.1f}%")
    # latency: one item through the scorer (features assumed computed by the shared encoder)
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(100): model.forward(Vte[:1], Qte[:1], Ote[:1], Mte[:1], nte[:1])
        if dev == "mps": torch.mps.synchronize()
    ms = (time.perf_counter() - t0) / 100 * 1000
    nparams = sum(p.numel() for p in model.params)
    print(f"scorer: {nparams/1e6:.2f}M params, {ms:.2f} ms/item on {dev} (+ ~7 ms per frame for DINOv2-S, ~3 ms for text)")
    print(f"overall GM-100 eval accuracy: {100*float((pred == te['y']).mean()):.1f}%")
    json.dump({"per_task": res, "scorer_params": nparams, "scorer_ms": ms, "overall": float((pred == te["y"]).mean()), "deltas": deltas, "epoch": best[2]},
              open(OUT / f"results{'_deltas' if deltas else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    {"build": build, "train": train}[sys.argv[1]]()
