"""Poll the pod's training log, keep a metrics history, flag divergence.

    uv run python scripts/train_monitor.py            # one poll -> outputs/train_metrics.json (+ prints health)
    uv run python scripts/train_monitor.py --loop 600 # poll every 10 min, print one line per poll

Parses openpi's stdout ("Step N: grad_norm=..., loss=..., param_norm=...") and tqdm
progress lines, plus nvidia-smi. Health checks: NaN/inf loss; loss trend up (last
window mean > previous window mean * 1.15 after warmup); grad-norm spike (> 5x
running median); process dead; no progress in 15 min.
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runpod_orchestrate import load, ssh  # noqa: E402

OUT = Path("outputs/train_metrics.json")
STEP_RE = re.compile(r"Step (\d+): grad_norm=([\d.eE+-]+|nan|inf), loss=([\d.eE+-]+|nan|inf), param_norm=([\d.eE+-]+)")
PROG_RE = re.compile(r"Progress on: ([\d.]+)(k?)it/([\d.]+)kit rate:([\d.]+)s/it remaining:([\d:]+) elapsed:([\d:]+)")


def poll() -> dict:
    st = load()
    cmd = ("grep -a 'Step [0-9]*: grad_norm' /workspace/train_t1.log; echo '---PROG'; grep -a 'Progress on' /workspace/train_t1.log | tail -1; "
           "echo '---GPU'; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader; "
           "echo '---PROC'; pgrep -fc '^[^ ]*python[^ ]* scripts/train.py'; echo '---CKPT'; ls /workspace/memory_robotics/vendor/openpi/checkpoints/pi05_rma_lora/t1/ 2>/dev/null | grep -v tmp | tr '\\n' ' '; "
           "echo; echo '---MTIME'; stat -c %Y /workspace/train_t1.log; date +%s")
    rc, out = ssh(st["ip"], st["port"], cmd, timeout=90)
    steps, prog, gpu, proc, ckpt, mtime = [], None, None, None, [], (None, None)
    section = None
    for line in out.splitlines():
        if line.startswith("---"):
            section = line[3:]; continue
        if section is None:
            m = STEP_RE.search(line)
            if m:
                s_, g, l, p = m.groups()
                steps.append({"step": int(s_), "grad_norm": float(g), "loss": float(l), "param_norm": float(p)})
        elif section == "PROG":
            m = PROG_RE.search(line)
            if m:
                cur, k, tot, rate, rem, el = m.groups()
                prog = {"step": float(cur) * (1000 if k == "k" else 1), "total": float(tot) * 1000, "sec_per_step": float(rate),
                        "remaining": rem, "elapsed": el}
        elif section == "GPU" and "," in line:
            mu, mt, ut, tp = [x.strip().split()[0] for x in line.split(",")]
            gpu = {"mem_used_mib": int(mu), "mem_total_mib": int(mt), "util_pct": int(ut), "temp_c": int(tp)}
        elif section == "PROC" and line.strip().isdigit():
            proc = int(line.strip())
        elif section == "CKPT":
            ckpt = [c for c in line.split() if c.isdigit()]
        elif section == "MTIME" and line.strip().isdigit():
            mtime = (mtime[1], int(line.strip())) if mtime[1] is not None else (int(line.strip()), None)
    log_age_s = (mtime[1] - mtime[0]) if all(mtime) else None
    return {"polled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "steps": steps, "progress": prog,
            "gpu": gpu, "train_procs": proc, "checkpoints": sorted(int(c) for c in ckpt), "log_age_s": log_age_s}


def health(m: dict) -> list[str]:
    flags = []
    losses = [s["loss"] for s in m["steps"]]
    grads = [s["grad_norm"] for s in m["steps"]]
    if any(x != x or x in (float("inf"), float("-inf")) for x in losses + grads):
        flags.append("NaN/inf in loss or grad_norm")
    if len(losses) >= 12:
        prev, last = losses[-12:-6], losses[-6:]
        if statistics.mean(last) > 1.15 * statistics.mean(prev):
            flags.append(f"loss trending up: last-6 mean {statistics.mean(last):.4f} vs prior {statistics.mean(prev):.4f}")
    if len(grads) >= 6:
        med = statistics.median(grads[:-1]) or 1e-9
        if grads[-1] > 5 * med:
            flags.append(f"grad_norm spike {grads[-1]:.2f} (median {med:.2f})")
    if m["train_procs"] == 0:
        flags.append("training process is not running")
    if m["log_age_s"] is not None and m["log_age_s"] > 900:
        flags.append(f"no log output for {m['log_age_s']//60} min")
    return flags


def summary(m: dict) -> str:
    p = m["progress"] or {}
    last = m["steps"][-1] if m["steps"] else None
    g = m["gpu"] or {}
    fl = health(m)
    return (f"step {int(p.get('step', 0))}/{int(p.get('total', 0))} {p.get('sec_per_step', '?')}s/it rem {p.get('remaining', '?')} | "
            f"loss {last['loss']:.4f} grad {last['grad_norm']:.3f} (@{last['step']})" if last else "no step metrics yet") + \
           f" | gpu {g.get('util_pct', '?')}% {g.get('mem_used_mib', '?')}MiB {g.get('temp_c', '?')}C | ckpts {m['checkpoints']} | " + \
           ("OK" if not fl else "FLAGS: " + "; ".join(fl))


RUN_META = {"run": "pi05_rma_lora / t1", "gpu": "1x NVIDIA L40 (46 GB)", "recipe": "LoRA (gemma_2b_lora + gemma_300m_lora), init pi05_libero, batch 16, 8000 steps, cosine lr peak 5e-5 warmup 500",
            "data": "RoboMemArena task 1 subtask demos (belu/rma_task1)", "log_every": 100, "save_every": 1000,
            "targets": {"paper_tsr": 20.0, "paper_csr": 42.8, "paper_recipe": "full fine-tune, batch 128, 40k steps, 4x H100", "stock": "0/51"}}
EVAL = Path("outputs/rma_pi05_ft_task1/results.json")
TEMPLATE = Path(__file__).with_name("train_dashboard.html")
DASH = Path("outputs/train_dashboard.html")


def render(m: dict) -> None:
    if not TEMPLATE.exists():
        return
    html = TEMPLATE.read_text().replace("__METRICS__", json.dumps(m).replace("</", "<\\/"))
    DASH.write_text(html)


def main() -> None:
    loop = int(sys.argv[sys.argv.index("--loop") + 1]) if "--loop" in sys.argv else 0
    while True:
        prev = json.loads(OUT.read_text()) if OUT.exists() else {}
        m = poll()
        m["health"] = health(m)
        m["meta"] = RUN_META
        hist = prev.get("history", [])
        p, g = m["progress"] or {}, m["gpu"] or {}
        hist.append({"t": m["polled_at"], "step": p.get("step"), "sec_per_step": p.get("sec_per_step"),
                     "util": g.get("util_pct"), "mem": g.get("mem_used_mib"), "temp": g.get("temp_c"), "flags": m["health"]})
        m["history"] = hist[-400:]
        m["eval"] = json.loads(EVAL.read_text()) if EVAL.exists() else None
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(m, indent=1))
        render(m)
        print(f"[{m['polled_at']}] {summary(m)}", flush=True)
        if not loop:
            break
        time.sleep(loop)


if __name__ == "__main__":
    main()
