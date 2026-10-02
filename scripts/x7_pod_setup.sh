#!/usr/bin/env bash
# X7 setup: openpi env + base pi05_libero checkpoint + GM-100 videos referenced by RoboProcessBench + RPB tables.
# The task-1 LoRA checkpoint and our rollout videos are uploaded from the Mac separately (UPLOAD_DONE marker).
set -u
export DEBIAN_FRONTEND=noninteractive
cd /workspace
[ -d memory_robotics/.git ] || git clone -q https://github.com/Phillips-Ugo/memory_robotics.git
cd memory_robotics && git pull -q origin main
bash scripts/setup_gpu_box.sh > /workspace/setup.log 2>&1 || { echo "[SETUP FAILED]"; tail -20 /workspace/setup.log; exit 1; }
. /workspace/env.sh; . /workspace/token.sh 2>/dev/null
python3 scripts/patch_openpi_config.py --repo-id belu/rma_task1
echo "[SETUP OK $(date -u +%H:%M)]"
cd vendor/openpi && uv pip install -q h5py imageio imageio-ffmpeg >/dev/null 2>&1
uv run python /workspace/memory_robotics/scripts/download_pi05.py > /workspace/pi05_dl.log 2>&1 && echo "[BASE CKPT OK $(date -u +%H:%M)]" || echo "[BASE CKPT FAILED]"
mkdir -p /workspace/data && cd /workspace/data
uv run --project /workspace/memory_robotics/vendor/openpi huggingface-cli download ProcessBench-2026/RoboProcessBench --repo-type dataset --include "splits/*" "metadata/*" --local-dir RoboProcessBench > /dev/null 2>&1 && echo "[RPB TABLES OK]"
uv run --project /workspace/memory_robotics/vendor/openpi python - <<'PY'
import os, json, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
rows=[json.loads(l) for f in ("eval","sft") for l in open(f"RoboProcessBench/splits/processdata_{f}.jsonl")]
need=sorted(set((r["source_task_id"], r["source_unit_id"].split("__")[1]) for r in rows if r["source"]=="GM-100"))
paths=[f"{t}/videos/chunk-000/observation.images.camera_top/{e}.mp4" for t,e in need]
tok=os.environ.get("HF_TOKEN"); t0=time.time()
def get(p):
    for a in range(3):
        try: return hf_hub_download("rhos-ai/gm100-cobotmagic-lerobot", p, repo_type="dataset", token=tok, local_dir="gm100")
        except Exception as e:
            if a==2: return "ERR"
            time.sleep(5)
with ThreadPoolExecutor(16) as ex:
    res=list(ex.map(get, paths))
print(f"[GM100 VIDEOS OK {len(paths)-res.count('ERR')}/{len(paths)} in {time.time()-t0:.0f}s]", flush=True)
PY
echo "[X7 SETUP DONE $(date -u +%H:%M)]"
