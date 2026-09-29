"""Train stage heads on demo frames, test on held-out policy rollouts (seed set B): does supervision from the
benchmark's own demos transfer to the policy's visual distribution, and across tasks?

    uv run python -m jev.transfer_eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .train_head import episode_metrics, load

OPS = {"stage_done_1": [(0.5, 3), (0.8, 3), (0.95, 5)], "stage_done_2": [(0.5, 3), (0.8, 3), (0.95, 5)]}


def run(name, Xtr_raw, ytr_rows, Xte_raw, te_rows):
    sc = StandardScaler().fit(Xtr_raw); Xtr = sc.transform(Xtr_raw); Xte = sc.transform(Xte_raw)
    print(f"\n== {name}: train {len(Xtr)} frames -> test {len({(r['src'], r['ep']) for r in te_rows})} held-out rollout episodes")
    out = {}
    for q, ops in OPS.items():
        ytr = np.array([r[q] for r in ytr_rows]); yte = np.array([r[q] for r in te_rows])
        if ytr.min() == ytr.max():
            print(f"  {q}: no positives in training set"); continue
        p = LogisticRegression(max_iter=3000, C=0.5).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, p)); line = f"  {q}: auroc {auc:.3f} |"
        out[q] = {"auroc": auc}
        for thr, k in ops:
            em = episode_metrics(te_rows, p, q, thr=thr, k=k)
            line += f" p>={thr}x{k}: delay {em['median_delay_steps']}, missed {em['missed']}/{em['with_stage']}, early/false {em['false_triggers']} |"
            out[q][f"{thr}x{k}"] = em
        print(line)
    return out


def main():
    rows, F = load(Path("outputs/jev_frames"), "dinov2_vits14,dinov2_vits14_wrist", 1)
    seedset = np.array([r["seedset"] for r in rows]); teB = seedset == "B"
    te_rows = [r for r, m in zip(rows, teB) if m]; Xte = F[teB]
    drows = [json.loads(l) for l in open("outputs/jev_demos/index.jsonl")]
    D = np.concatenate([np.load("outputs/jev_demos/features_dinov2_vits14.npy"), np.load("outputs/jev_demos/features_dinov2_vits14_wrist.npy")], axis=1)
    dtask = np.array([r["task"] for r in drows])
    res = {}
    res["rollouts_A"] = run("rollouts seed set A (same distribution, reference)", F[seedset == "A"], [r for r, m in zip(rows, seedset == "A") if m], Xte, te_rows)
    res["demos_task1"] = run(f"task-1 demos ({len({r['seed'] for r in drows if r['task']==1})} seeds)", D[dtask == 1], [r for r, t in zip(drows, dtask) if t == 1], Xte, te_rows)
    res["demos_tasks123"] = run("tasks 1-3 demos (multi-task)", D, drows, Xte, te_rows)
    res["demos_tasks23"] = run("tasks 2-3 demos (zero-shot to task 1)", D[dtask != 1], [r for r, t in zip(drows, dtask) if t != 1], Xte, te_rows)
    Path("outputs/jev_demos/transfer_results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
