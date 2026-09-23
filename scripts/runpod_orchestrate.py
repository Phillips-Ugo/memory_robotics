"""Run M2b on RunPod from this machine: create pod -> wait -> ssh -> GPU check
(terminate + next GPU type on failure) -> pipeline -> stream log -> stop.

    uv run python scripts/runpod_orchestrate.py check          # provision, GPU check, terminate (cents)
    uv run python scripts/runpod_orchestrate.py pipeline       # full M2b through the smoke train, keep pod
    uv run python scripts/runpod_orchestrate.py ssh            # print the ssh command for the current pod
    uv run python scripts/runpod_orchestrate.py run "<cmd>"    # run a command on the current pod
    uv run python scripts/runpod_orchestrate.py train         # launch the real 8k-step LoRA run (background on the pod)
    uv run python scripts/runpod_orchestrate.py train-status  # last train log lines + checkpoints on disk
    uv run python scripts/runpod_orchestrate.py eval [STEP] [TRIALS]   # serve checkpoint, run task-1 eval, fetch results.json
    uv run python scripts/runpod_orchestrate.py stop|terminate

Needs RUNPOD_API_KEY and (for pipeline) HF_TOKEN in the environment; ~/.ssh/id_ed25519
registered in RunPod settings. Pod id is remembered in outputs/runpod_pod.json.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import runpod

runpod.api_key = os.environ["RUNPOD_API_KEY"]
STATE = Path("outputs/runpod_pod.json")
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU_ORDER = ["NVIDIA RTX A6000", "NVIDIA L40", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40S"]
CLOUDS = ["COMMUNITY", "SECURE"]
SSH_OPTS = ["-i", os.path.expanduser("~/.ssh/id_ed25519"), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20"]


def save(pod: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(pod, indent=1))


def load() -> dict | None:
    return json.loads(STATE.read_text()) if STATE.exists() else None


def create(gpu: str, cloud: str) -> dict | None:
    try:
        pod = runpod.create_pod(
            name="memory-robotics-m2b", image_name=IMAGE, gpu_type_id=gpu, cloud_type=cloud,
            gpu_count=1, volume_in_gb=150, container_disk_in_gb=20, volume_mount_path="/workspace",
            ports="22/tcp", support_public_ip=True, start_ssh=True,
        )
        print(f"created pod {pod['id']} on {gpu} ({cloud})", flush=True)
        return pod
    except Exception as e:  # noqa: BLE001
        print(f"  {gpu} ({cloud}) unavailable: {str(e)[:120]}", flush=True)
        return None


def wait_ssh(pod_id: str, timeout: int = 600) -> tuple[str, int]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = runpod.get_pod(pod_id)
        rt = p.get("runtime") or {}
        for port in rt.get("ports") or []:
            if port.get("privatePort") == 22 and port.get("isIpPublic"):
                return port["ip"], int(port["publicPort"])
        time.sleep(10)
    raise TimeoutError("pod never exposed ssh")


def ssh(ip: str, port: int, cmd: str, stream: bool = False, timeout: int | None = None) -> tuple[int, str]:
    full = ["ssh", *SSH_OPTS, "-p", str(port), f"root@{ip}", cmd]
    if stream:
        proc = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = []
        for line in proc.stdout:  # type: ignore[union-attr]
            print(line, end="", flush=True)
            out.append(line)
        proc.wait(timeout=timeout)
        return proc.returncode, "".join(out)
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


GPU_CHECK = ("python3 -c \"import torch; ok=torch.cuda.is_available(); print('GPU OK' if ok else 'GPU BROKEN')\" 2>/dev/null; "
             "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null")


def provision() -> tuple[dict, str, int]:
    """Create a pod that passes the GPU check; terminate failures and try the next option."""
    for cloud in CLOUDS:
        for gpu in GPU_ORDER:
            pod = create(gpu, cloud)
            if not pod:
                continue
            try:
                ip, port = wait_ssh(pod["id"])
                # sshd can take a moment after the port appears
                for _ in range(12):
                    rc, out = ssh(ip, port, GPU_CHECK, timeout=60)
                    if rc == 0 and out.strip():
                        break
                    time.sleep(10)
                print(out.strip(), flush=True)
                if "GPU OK" in out:
                    save({"id": pod["id"], "ip": ip, "port": port, "gpu": gpu, "cloud": cloud})
                    return pod, ip, port
                print("  GPU check failed on this host -> terminating and trying the next", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  provisioning error: {e}", flush=True)
            runpod.terminate_pod(pod["id"])
    raise SystemExit("no host passed the GPU check")


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "check"
    if action == "check":
        pod, ip, port = provision()
        print(f"pod {pod['id']} healthy at {ip}:{port}; terminating (check only)")
        runpod.terminate_pod(pod["id"]); STATE.unlink(missing_ok=True)
        return
    if action == "pipeline":
        tok = os.environ.get("HF_TOKEN", "")
        if not tok.startswith("hf_"):
            raise SystemExit("HF_TOKEN not set")
        st = load()
        if st:
            ip, port = st["ip"], st["port"]
            print(f"reusing pod {st['id']} at {ip}:{port}")
        else:
            pod, ip, port = provision()
        auto_stop = "(nohup sh -c 'sleep 21600; runpodctl stop pod $RUNPOD_POD_ID' > /dev/null 2>&1 < /dev/null &)"
        ssh(ip, port, auto_stop)
        ssh(ip, port, f"mkdir -p /workspace && echo 'export HF_TOKEN={tok}' > /workspace/token.sh")
        # fully detach (stdin/stdout/stderr) or ssh waits for the background job to finish
        launch = ("curl -sSL https://raw.githubusercontent.com/Phillips-Ugo/memory_robotics/main/scripts/m2b_pipeline.sh "
                  "-o /workspace/m2b.sh && (nohup bash /workspace/m2b.sh > /workspace/m2b.log 2>&1 < /dev/null &) ; sleep 2; echo launched")
        print(ssh(ip, port, launch)[1].strip())
        # stream until the sentinel or the process ends
        ssh(ip, port, "tail -n +1 -f /workspace/m2b.log | grep --line-buffered -v Xet & "
                      "while pgrep -f m2b.sh > /dev/null; do sleep 15; done; sleep 2; kill %1 2>/dev/null; "
                      "grep -c 'M2B PIPELINE OK' /workspace/m2b.log", stream=True)
        return
    st = load()
    if not st:
        raise SystemExit("no current pod (outputs/runpod_pod.json)")
    ip, port = st["ip"], st["port"]
    ENV = ". /workspace/env.sh; cd /workspace/memory_robotics/vendor/openpi"
    if action == "train":
        # keep the pod alive long enough: reset the auto-stop to 8 h
        ssh(ip, port, "pkill -f 'sleep 21600' ; (nohup sh -c 'sleep 28800; runpodctl stop pod $RUNPOD_POD_ID' > /dev/null 2>&1 < /dev/null &)")
        cmd = (f"{ENV} && (PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 nohup uv run scripts/train.py pi05_rma_lora --exp-name t1 "
               f"--overwrite --no-wandb-enabled > /workspace/train_t1.log 2>&1 < /dev/null &) ; sleep 5; pgrep -fc 'train.py pi05_rma_lora'")
        print("train procs:", ssh(ip, port, cmd)[1].strip())
        return
    if action == "train-status":
        cmd = ("grep -v Xet /workspace/train_t1.log | grep -i 'step\|loss\|error\|Traceback' | tail -8; "
               "echo '-- checkpoints:'; ls /workspace/memory_robotics/vendor/openpi/checkpoints/pi05_rma_lora/t1/ 2>/dev/null | tail -5; "
               "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")
        print(ssh(ip, port, cmd)[1])
        return
    if action == "eval":
        step = sys.argv[2] if len(sys.argv) > 2 else "8000"
        trials = sys.argv[3] if len(sys.argv) > 3 else "51"
        ckpt = f"checkpoints/pi05_rma_lora/t1/{step}"
        serve = (f"{ENV} && pkill -f serve_policy; (nohup uv run scripts/serve_policy.py policy:checkpoint "
                 f"--policy.config=pi05_rma_lora --policy.dir={ckpt} > /workspace/server.log 2>&1 < /dev/null &) ; "
                 "for i in $(seq 1 60); do grep -q 'server listening' /workspace/server.log && break; sleep 5; done; "
                 "tail -1 /workspace/server.log")
        print(ssh(ip, port, serve, timeout=600)[1].strip())
        ev = (". /workspace/env.sh; cd /workspace/memory_robotics/vendor/RoboMemArena/evaluation_benchmark && "
              f"MUJOCO_GL=egl PYTHONUNBUFFERED=1 ../../rma-venv/bin/python scripts/eval_task1_only.py "
              f"--adapter-spec /workspace/memory_robotics/scripts/02_rma_pi05_adapter.py:build_adapter "
              f"--num-trials-per-task {trials} --seed 50 --video-out-path /workspace/memory_robotics/outputs/rma_pi05_ft_task1 "
              "2>&1 | grep --line-buffered 'Episode\|Final result\|Results JSON\|Traceback\|Error'")
        ssh(ip, port, ev, stream=True)
        Path("outputs/rma_pi05_ft_task1").mkdir(parents=True, exist_ok=True)
        subprocess.run(["scp", *SSH_OPTS, "-P", str(port),
                        f"root@{ip}:/workspace/memory_robotics/outputs/rma_pi05_ft_task1/results.json",
                        "outputs/rma_pi05_ft_task1/results.json"])
        print("fetched outputs/rma_pi05_ft_task1/results.json")
        return
    if action == "watch":
        log = sys.argv[2] if len(sys.argv) > 2 else "/workspace/m2b.log"
        seen = 0
        while True:
            rc, out = ssh(ip, port, f"grep -v Xet {log} 2>/dev/null | grep -v 'Map:\\|Creating parquet'", timeout=60)
            lines = out.splitlines()
            for l in lines[seen:]:
                print(l, flush=True)
            seen = len(lines)
            if any(k in out for k in ("M2B PIPELINE OK", "Traceback", "Error:")):
                break
            rc2, alive = ssh(ip, port, "pgrep -fc 'm2b.sh|train.py'", timeout=30)
            if alive.strip() == "0":
                print("[no pipeline/train process on the pod]"); break
            time.sleep(60)
        return
    if action == "ssh":
        print(f"ssh {' '.join(SSH_OPTS)} -p {st['port']} root@{st['ip']}")
    elif action == "run":
        rc, out = ssh(st["ip"], st["port"], " ".join(sys.argv[2:]), stream=True)
    elif action == "stop":
        runpod.stop_pod(st["id"]); print("stopped", st["id"])
    elif action == "terminate":
        runpod.terminate_pod(st["id"]); STATE.unlink(missing_ok=True); print("terminated", st["id"])


if __name__ == "__main__":
    main()
