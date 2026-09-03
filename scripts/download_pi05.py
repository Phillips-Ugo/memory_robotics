"""Resumable, verified download of the pi05_libero checkpoint (~11.6 GiB).

openpi's built-in downloader (gcsfs recursive get) stalls on slow links and, if
restarted, appends to half-written files -> corrupted OCDBT checkpoint. gcsfs also
hangs silently near the end of large objects on some boxes. So: list the files via
gcsfs (cheap, reliable), but move the bytes with curl over plain HTTPS
(storage.googleapis.com serves public objects) with resume + timeouts, then verify
every file's size against the bucket. Safe to re-run until it prints OK.

Run from the openpi env (gcsfs is installed there):
    cd vendor/openpi && OPENPI_DATA_HOME=/workspace/openpi_cache \
        uv run python /workspace/memory_robotics/scripts/download_pi05.py
Then serve_policy.py finds the checkpoint in its cache and skips its own download.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import gcsfs

BUCKET_PATH = "openpi-assets/checkpoints/pi05_libero"
cache = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser().resolve()
dest = cache / "openpi-assets" / "checkpoints" / "pi05_libero"

fs = gcsfs.GCSFileSystem(token="anon")
files = [f for f in fs.find(BUCKET_PATH) if not f.endswith("/")]
infos = {f: fs.info(f)["size"] for f in files}
total = sum(infos.values())
print(f"{len(files)} files, {total/2**30:.2f} GiB -> {dest}", flush=True)


def fetch(remote: str, tmp: Path) -> None:
    url = "https://storage.googleapis.com/" + urllib.parse.quote(remote)
    cmd = [
        "curl", "-sS", "-L", "--fail",
        "-C", "-",                      # resume from whatever tmp already has
        "--retry", "10", "--retry-all-errors", "--retry-delay", "3",
        "--speed-limit", "100000", "--speed-time", "30",  # abort if <100 KB/s for 30 s
        "-o", str(tmp), url,
    ]
    subprocess.run(cmd, check=True)


done_bytes = 0
for i, (remote, size) in enumerate(sorted(infos.items(), key=lambda kv: kv[1]), 1):
    local = dest / remote[len(BUCKET_PATH) + 1 :]
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size == size:
        done_bytes += size
        continue
    tmp = local.with_suffix(local.suffix + ".tmp")
    for attempt in range(1, 8):
        try:
            t0 = time.time()
            if tmp.exists() and tmp.stat().st_size > size:
                tmp.unlink()  # over-long tmp can't be resumed; start clean
            fetch(remote, tmp)
            got = tmp.stat().st_size
            if got != size:
                raise IOError(f"size mismatch {got} != {size}")
            tmp.replace(local)
            done_bytes += size
            print(f"[{i}/{len(files)}] {local.name}  {size/2**20:.0f} MiB in {time.time()-t0:.0f}s  "
                  f"({done_bytes/total*100:.1f}% total)", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt} failed for {remote}: {e}", flush=True)
            time.sleep(3 * attempt)
    else:
        sys.exit(f"giving up on {remote}")

bad = [f for f, s in infos.items()
       if not (dest / f[len(BUCKET_PATH) + 1 :]).exists()
       or (dest / f[len(BUCKET_PATH) + 1 :]).stat().st_size != s]
if bad:
    sys.exit(f"{len(bad)} files still wrong — re-run")
print(f"OK: all {len(files)} files verified at {dest}", flush=True)
