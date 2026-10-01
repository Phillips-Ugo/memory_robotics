# Release status of candidate benchmarks for the decision model (checked 2026-10-01)

| Benchmark | Released | Data | Gating / notes |
|---|---|---|---|
| **RoboProcessBench / ProcessData** (2606.13040) | partial | HF `ProcessBench-2026/RoboProcessBench` (QA tables 57,892 rows: SFT 48,841 / Eval 9,051; prompts; scorer; Qwen2.5-VL-7B LoRA adapter + its predictions) and GH `ProcessBench-2026/RoboProcessBench` (MIT) | **No pixels redistributed**: `visual_ref` + `frame_indices_json` point into 4 upstream datasets (GM-100 MIT, RH20T CC BY-SA/NC, REASSEMBLE CC BY, AIST-Bimanual CC BY); `scripts/extract_frames_*.py` reconstruct frames. Eval per source: GM-100 2,643 items, RH20T 2,422, REASSEMBLE 2,562, AIST 1,424. All MCQ, accuracy micro / task-macro / source-macro. |
| **FailBench** (2609.03611) | no | GH `Metric-AI-Lab/failbench` is a landing page; HF nothing | "soon". 2,197 attempts from 14 sources when it drops. |
| **RoboMonitor** (2609.30715) | not found | — | 25 h pretraining data + 66-episode phase/failure/completion benchmark described, unreleased. |
| **SAFE** (2506.09937) | partial | GH `vla-safe/SAFE` (code), Drive zips: Franka+π0-FAST rollouts 25 GB, WidowX+OpenVLA 1.3 GB | LIBERO-Long rollouts must be regenerated with their openvla/openpi forks (per-episode csv+pkl+mp4, success in filename). |

Decision: RoboProcessBench on the GM-100 source (public, MIT, LeRobot format) is the apples-to-apples target: train on the SFT rows of that source, evaluate on its 2,643 eval rows, compare per-source against the released Qwen-LoRA predictions and the random/majority baselines in `metadata/task_distribution.csv`. Report latency/params alongside accuracy.
