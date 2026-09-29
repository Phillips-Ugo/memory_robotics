"""Fit the deployment heads on ALL labelled episodes and export them as a small npz the adapter can load.

    uv run python -m jev.export_head --features dinov2_vits14,dinov2_vits14_wrist --window 3

The npz holds, per question, the StandardScaler (mean, scale) and logistic weights (W, b), plus the
operating points chosen from the held-out sweep (threshold, persistence in sampled frames).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .train_head import load

OPERATING = {"stage_done_1": (0.95, 5), "stage_done_2": (0.80, 3)}  # from the seed-set A->B sweep (Day 15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="dinov2_vits14,dinov2_vits14_wrist")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--out", default="jev/heads_task1.npz")
    args = ap.parse_args()
    rows, F = load(Path("outputs/jev_frames"), args.features, args.window)
    sc = StandardScaler().fit(F); X = sc.transform(F)
    blobs = {"features": np.array(args.features), "window": np.array(args.window), "mean": sc.mean_, "scale": sc.scale_}
    for q, (thr, k) in OPERATING.items():
        y = np.array([r[q] for r in rows])
        clf = LogisticRegression(max_iter=3000, C=0.5).fit(X, y)
        blobs[f"{q}_W"] = clf.coef_[0]; blobs[f"{q}_b"] = clf.intercept_
        blobs[f"{q}_thr"] = np.array(thr); blobs[f"{q}_k"] = np.array(k)
        print(f"{q}: train acc {clf.score(X, y):.3f}, operating point p>={thr} x{k}")
    np.savez(args.out, **blobs); print("wrote", args.out, Path(args.out).stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
