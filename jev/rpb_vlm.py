"""Zero-shot VLM baselines on RoboProcessBench GM-100 (eval split) with the benchmark's own prompt template,
via the Message Batches API (50 % cost, async).

    uv run python -m jev.rpb_vlm submit --model claude-haiku-4-5
    uv run python -m jev.rpb_vlm submit --model claude-sonnet-5
    uv run python -m jev.rpb_vlm collect --model claude-haiku-4-5     # when batches have ended -> per-task accuracy

Inputs follow metadata/prompt_templates.md: single frame; ordered frames ("Frame k of n"); T8 shuffled frames
labelled X/Y/Z per display_labels_json; T9 two labelled panels. Answer parsed from the last <ANSWER>..</ANSWER>.
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic
import imageio.v3 as iio
import numpy as np
from PIL import Image

from .rpb import ROOT, gm_rows, video_path

OUT = Path("outputs/rpb_vlm"); OUT.mkdir(parents=True, exist_ok=True)
LETTERS = "ABCDEF"
STD = ("{question}\n\n{options}\n\nChoose exactly one option label.\n\nOutput protocol:\n- You may include brief reasoning before the final answer.\n"
       "- The final line must be exactly: <ANSWER>A</ANSWER>\n- Do not output anything after </ANSWER>.")
T8 = ("You are shown 3 frames from a robot manipulation task.\nThe frames are labeled X, Y, Z (these labels are arbitrary identifiers, not positional, and not time-ordered).\n"
      "Determine the correct chronological order of these frames (from earliest to latest).\nChoose exactly one 3-letter permutation, for example: YXZ means Y happened first, then X, then Z.\n\n"
      "Output protocol:\n- You may include brief reasoning before the final answer.\n- The final line must be exactly: <ANSWER>XYZ</ANSWER>\n- Do not output anything after </ANSWER>.")
T9 = ("A single comparison image shows two labeled robot-manipulation panels from the same episode.\nThe left-right placement of the panels and the labels are arbitrary identifiers and do not indicate temporal order.\n"
      "Which labeled panel happened earlier in the real manipulation sequence, X or Y?\nChoose exactly one label: X or Y.\n\n"
      "Output protocol:\n- You may include brief reasoning before the final answer.\n- The final line must be exactly: <ANSWER>X</ANSWER>\n- Do not output anything after </ANSWER>.")


def jpeg_b64(frame: np.ndarray, q: int = 85) -> str:
    buf = io.BytesIO(); Image.fromarray(frame).save(buf, format="JPEG", quality=q); return base64.standard_b64encode(buf.getvalue()).decode()


def img_block(b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


def build_content(r: dict, frames: list[np.ndarray]) -> list[dict]:
    t = r["task_id"]; labels = json.loads(r["display_labels_json"]) if r.get("display_labels_json") else None
    content: list[dict] = []
    if t == "T8":
        for lab, f in zip(labels, frames):
            content += [{"type": "text", "text": f"Frame {lab}:"}, img_block(jpeg_b64(f))]
        content.append({"type": "text", "text": T8}); return content
    if t == "T9":
        a, b = frames[0], frames[1]
        panel = np.concatenate([a, np.full((a.shape[0], 16, 3), 255, np.uint8), b], axis=1)
        content += [{"type": "text", "text": f"Left panel is labeled {labels[0]}, right panel is labeled {labels[1]}."}, img_block(jpeg_b64(panel))]
        content.append({"type": "text", "text": T9}); return content
    if len(frames) == 1:
        content.append(img_block(jpeg_b64(frames[0])))
    else:
        for i, f in enumerate(frames, 1):
            content += [{"type": "text", "text": f"Frame {i} of {len(frames)} (time-ordered):"}, img_block(jpeg_b64(f))]
    opts = "\n".join(f"{c}: {r[f'choice_{c}']}" for c in LETTERS if r.get(f"choice_{c}"))
    q = r["question"] or "Which option best describes what is shown?"
    content.append({"type": "text", "text": STD.format(question=q, options=opts)}); return content


def load_frames(rows: list[dict]) -> dict[tuple[str, int], np.ndarray]:
    want: dict[Path, set[int]] = defaultdict(set)
    for r in rows:
        for i in json.loads(r["frame_indices_json"]):
            want[video_path(r)].add(int(i))
    out = {}
    for k, (vp, idxs) in enumerate(want.items()):
        for t, f in enumerate(iio.imiter(vp)):
            if t in idxs:
                out[(str(vp), t)] = f
            if t >= max(idxs):
                break
        if k % 100 == 0:
            print(f"  frames: {k}/{len(want)} videos", flush=True)
    return out


def submit(model: str) -> None:
    client = anthropic.Anthropic()
    rows = [r for r in gm_rows("eval") if video_path(r).exists()]
    frames = load_frames(rows)
    reqs = []
    for r in rows:
        idx = [int(i) for i in json.loads(r["frame_indices_json"])]; vp = str(video_path(r))
        if any((vp, i) not in frames for i in idx):
            continue
        reqs.append({"custom_id": r["item_id"], "params": {"model": model, "max_tokens": 400,
                     "messages": [{"role": "user", "content": build_content(r, [frames[(vp, i)] for i in idx])}]}})
    ids = []
    chunk = 700  # keep each batch well under the 256 MB request limit
    for i in range(0, len(reqs), chunk):
        b = client.messages.batches.create(requests=reqs[i:i + chunk])
        ids.append(b.id); print(f"submitted batch {b.id} with {len(reqs[i:i+chunk])} requests", flush=True)
    json.dump({"model": model, "batches": ids, "n": len(reqs)}, open(OUT / f"batches_{model}.json", "w"))


def collect(model: str) -> None:
    client = anthropic.Anthropic()
    meta = json.load(open(OUT / f"batches_{model}.json"))
    rows = {r["item_id"]: r for r in gm_rows("eval")}
    preds, usage = {}, defaultdict(int)
    for bid in meta["batches"]:
        while True:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                break
            print(f"  {bid}: {b.processing_status} {b.request_counts}", flush=True); time.sleep(60)
        for res in client.messages.batches.results(bid):
            if res.result.type != "succeeded":
                preds[res.custom_id] = None; continue
            msg = res.result.message
            text = "".join(bl.text for bl in msg.content if bl.type == "text")
            m = re.findall(r"<ANSWER>\s*([A-Za-z]{1,3})\s*</ANSWER>", text)
            preds[res.custom_id] = m[-1].strip().upper() if m else None
            usage["in"] += msg.usage.input_tokens; usage["out"] += msg.usage.output_tokens
    acc = defaultdict(lambda: [0, 0]); unparsed = 0
    for iid, p in preds.items():
        r = rows[iid]; gold = r["answer"] if r["task_id"] not in ("T8",) else r["answer_text"]
        if r["task_id"] == "T9":
            gold = "X" if r["answer_text"].startswith("Image X") else "Y"
        if p is None:
            unparsed += 1
        a = acc[r["task_id"]]; a[0] += int(p == gold); a[1] += 1
    res = {t: {"acc": a[0] / a[1], "n": a[1]} for t, a in acc.items()}
    print(f"{model} on GM-100 eval ({len(preds)} items, {unparsed} unparsed): " + ", ".join(f"{t} {100*res[t]['acc']:.1f}%" for t in sorted(res, key=lambda s: int(s[1:]))))
    print(f"tokens: {usage['in']} in / {usage['out']} out")
    json.dump({"model": model, "per_task": res, "unparsed": unparsed, "usage": dict(usage), "preds": preds}, open(OUT / f"results_{model}.json", "w"), indent=1)


if __name__ == "__main__":
    model = sys.argv[sys.argv.index("--model") + 1]
    {"submit": submit, "collect": collect}[sys.argv[1]](model)
