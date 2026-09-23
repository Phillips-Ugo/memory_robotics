#!/usr/bin/env bash
# Resume the stopped M2b pod, pull the training log, eval videos and final checkpoint, then TERMINATE it.
# Needs RUNPOD_API_KEY in the env and a positive RunPod balance. Usage: bash scripts/runpod_fetch_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python - <<'PY'
import runpod, os, json, time
runpod.api_key=os.environ["RUNPOD_API_KEY"]
st=json.load(open("outputs/runpod_pod.json"))
p=runpod.get_pod(st["id"])
if p.get("desiredStatus")!="RUNNING":
    runpod.resume_pod(st["id"], gpu_count=1)
for _ in range(60):
    p=runpod.get_pod(st["id"]); ports=[x for x in ((p.get("runtime") or {}).get("ports") or []) if x.get("privatePort")==22]
    if ports:
        st["ip"], st["port"]=ports[0]["ip"], ports[0]["publicPort"]; json.dump(st, open("outputs/runpod_pod.json","w"), indent=1); break
    time.sleep(10)
else: raise SystemExit("pod has no ssh port after 10 min")
print("ssh", st["ip"], st["port"])
PY
IP=$(python3 -c "import json;print(json.load(open('outputs/runpod_pod.json'))['ip'])")
PORT=$(python3 -c "import json;print(json.load(open('outputs/runpod_pod.json'))['port'])")
SSH="ssh -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=no -p $PORT"
for i in $(seq 1 30); do $SSH root@$IP true 2>/dev/null && break; sleep 10; done
mkdir -p outputs/m2b_run/checkpoint_7999
$SSH root@$IP "du -sh /workspace/memory_robotics/vendor/openpi/checkpoints/pi05_rma_lora/t1/7999/params /workspace/memory_robotics/outputs/rma_pi05_ft_task1"
rsync -az --info=progress2 -e "$SSH" root@$IP:/workspace/train_t1.log root@$IP:/workspace/eval_t1.log root@$IP:/workspace/m2b.log outputs/m2b_run/
rsync -az --info=progress2 -e "$SSH" root@$IP:/workspace/memory_robotics/outputs/rma_pi05_ft_task1/ outputs/rma_pi05_ft_task1/
rsync -az --info=progress2 -e "$SSH" root@$IP:/workspace/memory_robotics/vendor/openpi/checkpoints/pi05_rma_lora/t1/7999/params root@$IP:/workspace/memory_robotics/vendor/openpi/checkpoints/pi05_rma_lora/t1/7999/assets root@$IP:/workspace/memory_robotics/vendor/openpi/checkpoints/pi05_rma_lora/t1/7999/_CHECKPOINT_METADATA outputs/m2b_run/checkpoint_7999/
ls -la outputs/m2b_run outputs/m2b_run/checkpoint_7999; ls outputs/rma_pi05_ft_task1 | wc -l
uv run python scripts/runpod_orchestrate.py terminate
echo "[FETCH DONE, POD TERMINATED]"
