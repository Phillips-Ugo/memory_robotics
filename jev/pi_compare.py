"""Does π0.5's own vision-language representation already answer the judge's questions?

(b) rollouts: heads on π features (task-1 checkpoint) vs the DINOv2-S heads, same seed-set A -> B protocol.
(a) RoboProcessBench GM-100: the same list-wise scorer on π (base pi05_libero) frame features vs DINOv2-S.

    uv run python -m jev.pi_compare rollouts outputs/pi_feats/rollouts_pi05_t1.npz
    uv run python -m jev.pi_compare rpb outputs/pi_feats/rpb_pi05_base.npz     # writes outputs/rpb_pi/{sft,eval}.npz, then run rpb.train --dir outputs/rpb_pi
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .train_head import episode_metrics


def rollouts(path: str, part: str = "both") -> None:
    d = np.load(path); F = d["F"].astype(np.float32); rows = [json.loads(r) for r in d["rows"]]
    D = F.shape[1] // 2
    F = {"img": F[:, :D], "lang": F[:, D:], "both": F}[part]
    # temporal window of 3 (same as the DINOv2 runs), by episode in step order
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["src"], rows[i]["ep"], rows[i]["step"]))
    rows = [rows[i] for i in order]; F = F[order]
    W = np.zeros((len(F), F.shape[1] * 3), np.float32); hist = {}
    for i, r in enumerate(rows):
        k = (r["src"], r["ep"]); h = hist.setdefault(k, []); h.append(F[i]); h[:] = h[-3:]
        pad = [h[0]] * (3 - len(h)) + h; W[i] = np.concatenate(pad)
    for r in rows:
        r["seedset"] = "A" if r["src"] == "rma_pi05_ft_task1" else "B"
    A = np.array([r["seedset"] == "A" for r in rows]); B = ~A
    sc = StandardScaler().fit(W[A]); Xtr, Xte = sc.transform(W[A]), sc.transform(W[B]); te_rows = [r for r, m in zip(rows, B) if m]
    print(f"π0.5 features ({part}, dim {F.shape[1]}, window 3): train {A.sum()} frames (seed set A) -> test {B.sum()} frames / {len({(r['src'], r['ep']) for r in te_rows})} episodes (seed set B)")
    out = {}
    for q, ops in (("stage_done_1", [(0.5, 3), (0.95, 5)]), ("stage_done_2", [(0.5, 3), (0.8, 3)])):
        ytr = np.array([r[q] for r, m in zip(rows, A) if m]); yte = np.array([r[q] for r in te_rows])
        p = LogisticRegression(max_iter=3000, C=0.5).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, p)); line = f"  {q}: auroc {auc:.3f} |"; out[q] = {"auroc": auc}
        for thr, k in ops:
            em = episode_metrics(te_rows, p, q, thr=thr, k=k); out[q][f"{thr}x{k}"] = em
            line += f" p>={thr}x{k}: delay {em['median_delay_steps']}/{em['p90_delay_steps']}, missed {em['missed']}/{em['with_stage']}, early {em['false_triggers']} |"
        print(line)
    Path("outputs/pi_feats").mkdir(exist_ok=True, parents=True)
    json.dump(out, open(f"outputs/pi_feats/rollouts_results_{part}.json", "w"), indent=1)


def rpb(path: str) -> None:
    from .rpb import ROOT, gm_rows, video_path, LETTERS
    d = np.load(path); F = d["F"].astype(np.float32); keys = {k: i for i, k in enumerate(d["keys"])}
    from sentence_transformers import SentenceTransformer
    import torch
    txt = SentenceTransformer("BAAI/bge-small-en-v1.5", device="mps" if torch.backends.mps.is_available() else "cpu")
    out = Path("outputs/rpb_pi"); out.mkdir(parents=True, exist_ok=True)
    for split in ("eval", "sft"):
        rows = gm_rows(split)
        V, Q, O, M, keep = [], [], [], [], []
        qe = txt.encode([r["question"] for r in rows], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        oe = txt.encode([r[f"choice_{c}"] or "" for r in rows for c in LETTERS], batch_size=256, normalize_embeddings=True, show_progress_bar=False).reshape(len(rows), 6, -1)
        for k, r in enumerate(rows):
            vp = f"/workspace/data/gm100/{r['source_task_id']}/videos/chunk-000/observation.images.camera_top/{r['source_unit_id'].split('__')[1]}.mp4"
            idx = [int(i) for i in json.loads(r["frame_indices_json"])]
            if any(f"{vp}|{i}" not in keys for i in idx):
                continue
            fr = np.stack([F[keys[f"{vp}|{i}"]] for i in idx]); pad = np.zeros((6, fr.shape[1]), np.float32); pad[:len(fr)] = fr[:6]
            V.append(pad); Q.append(qe[k]); O.append(oe[k]); M.append([1 if (r[f"choice_{c}"] or "") else 0 for c in LETTERS]); keep.append(k)
        np.savez(out / f"{split}.npz", V=np.stack(V), Q=np.stack(Q), O=np.stack(O), M=np.array(M), y=np.array([LETTERS.index(rows[k]["answer"]) for k in keep]),
                 task=np.array([rows[k]["task_id"] for k in keep]), nframes=np.array([int(rows[k]["num_frames"]) for k in keep]),
                 item=np.array([rows[k]["item_id"] for k in keep]), unit=np.array([rows[k]["source_unit_id"] for k in keep]))
        print(f"{split}: {len(keep)}/{len(rows)} items -> {out/f'{split}.npz'}")
    print("now: uv run python -m jev.rpb train --dir outputs/rpb_pi")


if __name__ == "__main__":
    {"rollouts": lambda p: [rollouts(p, part) for part in ("img", "both")], "rpb": rpb}[sys.argv[1]](sys.argv[2])
