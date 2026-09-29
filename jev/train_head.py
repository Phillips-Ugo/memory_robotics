"""Train and evaluate fast decision heads on frozen frame features ("Jev for robots", v0).

    uv run python -m jev.train_head --features dinov2_vits14

Questions: stage_done_1, stage_done_2 (boolean) and current_stage (3-way choice).
Splits (all frames of an episode stay together):
  seedset   : train on seed set A (seeds 50-100, fixed prompt) -> test on seed set B (101-151, all strategies)
  strategy  : train on fixed-prompt episodes (A + B-fixed) -> test on primitive + memory episodes (different motion)
Metrics: frame accuracy, AUROC, and the episode-level numbers that matter for a controller:
  detection delay = steps from the true stage completion to the first sustained positive (k consecutive frames)
  false trigger   = fraction of episodes with a sustained positive before the stage was actually done
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def load(out: Path, feat: str, window: int = 1):
    """feat may be a comma list (concatenated); window>1 stacks the previous sampled frames of the same episode."""
    rows = [json.loads(l) for l in (out / "index.jsonl").read_text().splitlines()]
    F = np.concatenate([np.load(out / f"features_{f}.npy") for f in feat.split(",")], axis=1)
    assert len(rows) == len(F), (len(rows), F.shape)
    if window > 1:
        prev = {}
        stacked = np.zeros((len(F), F.shape[1] * window), dtype=F.dtype)
        for i, r in enumerate(rows):
            key = (r["src"], r["ep"]); hist = prev.setdefault(key, [])
            hist.append(F[i]); hist[:] = hist[-window:]
            pad = [hist[0]] * (window - len(hist)) + hist
            stacked[i] = np.concatenate(pad)
        F = stacked
    return rows, F


def episode_metrics(rows, p, label, thr=0.5, k=3, every=10):
    """Per-episode detection delay and false triggers for one boolean question."""
    by_ep = {}
    for r, pi in zip(rows, p):
        by_ep.setdefault((r["src"], r["ep"]), []).append((r["step"], r[label], pi))
    delays, false_trig, n_done, n_total = [], 0, 0, 0
    for ep, items in by_ep.items():
        items.sort(); steps = [s for s, _, _ in items]; y = [l for _, l, _ in items]; pr = [x for _, _, x in items]
        n_total += 1
        first_true = next((s for s, l in zip(steps, y) if l), None)
        # first sustained positive
        fired = None
        for i in range(len(pr) - k + 1):
            if all(x >= thr for x in pr[i:i + k]):
                fired = steps[i]; break
        if first_true is None:
            false_trig += int(fired is not None); continue
        n_done += 1
        if fired is None:
            delays.append(np.inf)
        elif fired < first_true - every:
            false_trig += 1; delays.append(fired - first_true)
        else:
            delays.append(fired - first_true)
    fin = [d for d in delays if np.isfinite(d) and d >= -every]
    return {"episodes": n_total, "with_stage": n_done, "median_delay_steps": float(np.median(fin)) if fin else None,
            "p90_delay_steps": float(np.percentile(fin, 90)) if fin else None,
            "missed": int(sum(1 for d in delays if not np.isfinite(d))), "false_triggers": false_trig}


def run_split(rows, F, name, train_mask, test_mask, model):
    print(f"\n== split: {name}  train {train_mask.sum()} frames / test {test_mask.sum()} frames")
    sc = StandardScaler().fit(F[train_mask])
    Xtr, Xte = sc.transform(F[train_mask]), sc.transform(F[test_mask])
    te_rows = [r for r, m in zip(rows, test_mask) if m]
    res = {}
    for q in ("stage_done_1", "stage_done_2", "current_stage"):
        ytr = np.array([r[q] for r, m in zip(rows, train_mask) if m]); yte = np.array([r[q] for r in te_rows])
        clf = (LogisticRegression(max_iter=2000, C=0.5) if model == "linear" else
               MLPClassifier(hidden_layer_sizes=(256,), max_iter=200, early_stopping=True, random_state=0)).fit(Xtr, ytr)
        pred = clf.predict(Xte); acc = float((pred == yte).mean())
        line = f"  {q:14s} acc {acc:.3f}"
        out = {"acc": acc}
        if q != "current_stage":
            p1 = clf.predict_proba(Xte)[:, 1]
            auc = float(roc_auc_score(yte, p1)) if 0 < yte.mean() < 1 else float("nan")
            em = episode_metrics(te_rows, p1, q)
            line += f" auroc {auc:.3f} | delay median {em['median_delay_steps']} p90 {em['p90_delay_steps']} steps, missed {em['missed']}/{em['with_stage']}, false triggers {em['false_triggers']}/{em['episodes']}"
            out.update({"auroc": auc, **em})
        print(line); res[q] = out
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="dinov2_vits14")
    ap.add_argument("--out", default="outputs/jev_frames")
    ap.add_argument("--model", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--window", type=int, default=1, help="stack the last N sampled frames (temporal context)")
    args = ap.parse_args()
    rows, F = load(Path(args.out), args.features, args.window)
    seedset = np.array([r["seedset"] for r in rows]); strat = np.array([r["strategy"] for r in rows])
    results = {}
    results["seedset_A_to_B"] = run_split(rows, F, "seed set A -> B (new seeds, all strategies)", seedset == "A", seedset == "B", args.model)
    results["fixed_to_primitive_memory"] = run_split(rows, F, "fixed-prompt episodes -> primitive + memory episodes", strat == "fixed", strat != "fixed", args.model)
    tag = args.features.replace(",", "+") + f"_{args.model}_w{args.window}"
    Path(args.out, f"results_{tag}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
