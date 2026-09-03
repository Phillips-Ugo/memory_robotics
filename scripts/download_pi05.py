"""Resumable, verified download of the pi05_libero checkpoint (~11.6 GiB).

openpi's built-in downloader (gcsfs recursive get) stalls on slow links and, if
restarted, appends to half-written files -> corrupted OCDBT checkpoint. This one
downloads file by file, verifies each against the bucket's size, retries, and skips
files that are already complete, so it can be re-run safely until everything matches.

Run from the openpi env (gcsfs is installed there):
    cd vendor/openpi && OPENPI_DATA_HOME=/workspace/openpi_cache \
        uv run python /workspace/memory_robotics/scripts/download_pi05.py
Then serve_policy.py finds the checkpoint in its cache and skips its own download.
"""

from __future__ import annotations

import os
import sys
import time
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

done_bytes = 0
for i, (remote, size) in enumerate(sorted(infos.items(), key=lambda kv: kv[1]), 1):
    local = dest / remote[len(BUCKET_PATH) + 1 :]
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size == size:
        done_bytes += size
        continue
    for attempt in range(1, 6):
        try:
            t0 = time.time()
            tmp = local.with_suffix(local.suffix + ".tmp")
            fs.get_file(remote, str(tmp))
            got = tmp.stat().st_size
            if got != size:
                raise IOError(f"size mismatch {got} != {size}")
            tmp.replace(local)
            done_bytes += size
            mb = size / 2**20
            print(f"[{i}/{len(files)}] {local.name}  {mb:.0f} MiB in {time.time()-t0:.0f}s  "
                  f"({done_bytes/total*100:.1f}% total)", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt} failed for {remote}: {e}", flush=True)
            time.sleep(3 * attempt)
    else:
        sys.exit(f"giving up on {remote}")

bad = [f for f, s in infos.items() if not (dest / f[len(BUCKET_PATH) + 1 :]).exists()
       or (dest / f[len(BUCKET_PATH) + 1 :]).stat().st_size != s]
if bad:
    sys.exit(f"{len(bad)} files still wrong — re-run")
print(f"OK: all {len(files)} files verified at {dest}", flush=True)
