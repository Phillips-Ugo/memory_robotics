# Survey: fast discriminative "decision models" beside a VLA (2026-09-29)

Compiled by a research agent from arXiv / project pages / GitHub; items marked *(uncertain)* were not verified at source.

## 1. Success / progress / stage-completion detectors
- **SuccessVQA** (Du et al. 2023, arxiv 2303.07280): success detection as VQA on Flamingo; the reference baseline; slow.
- **GVL** (Ma et al. 2024, 2411.04549): zero-shot progress by asking Gemini to order shuffled frames; VOC metric; not real-time.
- **SAFE** (Gu et al., NeurIPS 2025, 2506.09937): VLA internal features separate success/failure generically; small probe + conformal threshold; free at inference. **SAFECAST** (2608.04246) adds contrast sets + functional conformal calibration.
- **VLA-FAIL** (2606.21386): training-free Mahalanobis on VLA tokens + action-chunk consistency; **AUCPDT** metric (precision, recall, detection time).
- **FIPER** (NeurIPS 2025, 2510.09459): OOD score + chunk entropy, conformal on successful rollouts only.
- **Hide-and-Seek** (NeurIPS 2026, 2605.30834): failure detector from trajectory-level labels via contrastive localization; SOTA on LIBERO/VLABench across OpenVLA/π0/π0.5.
- **Foresight** (2606.23085) and **CheckVLA** (2607.26789): action-conditioned world-model latents; CheckVLA at 5 % false alarm: 77.9 % timely recall action-conditioned vs 48.6 % observation-only.
- **RoboMonitor** (Siemens/WPI, Sep 2026, 2609.30715): closest "Jev for robots" — one Qwen3-VL-4B predicts phase, failure probability, completion from multi-camera video; predictive pre-training; 52 labeled episodes → 93.1 % phase accuracy, spurious switches 15.2 → 5.0 %; **180–200 ms** per window (too slow for <50 ms).
- **FailureSpot** (2609.04277): timestamp-level detection from VLA internals with weak labels mined from action chunks.
- Subtask-completion heads: **SeqVLA** (2509.14138, "is this subtask done?" head on π0), **SparkVLA** (2608.16172: Stop-vs-prefix ordinal preferences from demo subtask boundaries; 98 % LIBERO-Long, 4.14 Hz), **τ0-VLA** (2608.16885).
- Progress: **ProcVLM** (2605.08774: Qwen3-VL-2B regression head, VOC 0.81/0.73), **Multiview Progress Prediction** (2603.00151: ViT-B/16 or MobileNetV2 + tiny LSTM, 4.86 % MAE — small backbones suffice), **ProgVLA** (2605.28231).
- Reality check: **FailBench** (2609.03611): best of 13 VLM detectors 0.77 balanced accuracy, near chance on contact-rich tasks, bias toward "success", and fine-tuned detectors underperform their bases. Also ARMOR, I-FailSense, AHA, Guardian.

## 2. Value / verifier-guided VLA inference
- **V-GPS** (CoRL 2024, 2410.13816): offline-RL Q re-ranks samples from 5 policies; plug-in critic pattern.
- **RoboMonkey** (CoRL 2025, 2506.17811): sample N, perturb, LLaVA-7B verifier with reward head trained on 20M synthetic pairs (RMSE to demo); power-law scaling in N; +8 SIMPLER; ~650 ms for 16 candidates on H100 — verifier is the bottleneck.
- **MG-Select** (ICLR 2026, 2510.05681): verifier-free (KL from masked-input reference); baseline to beat.
- **VGAS** (2602.07399), **DEAS** (2510.07730: simple MLP critics on VLA features), **UF-OPS** (2603.10282: +49 % real), **VERITAS** (2606.18247).
- **π*0.6 / RECAP** (2511.14759): value = smaller-backbone VLA (670M), distributional over 201 bins, negative time-to-success from episode success labels.

## 3. Small fast models and backbones
- TinyVLA (2409.12514), SmolVLA (2506.01844, 450M, 87.3 % LIBERO), **TurboVLA** (2607.27205: 0.2B, 31 ms / 32 Hz on RTX 4090, 97.6 % LIBERO), VLA-Perf (2602.18397: π0 3.2 ms on B100; vision compute-bound).
- timm 4090 throughput: ViT-S/16 7,693 img/s, ViT-B/16 2,787, SigLIP-B/16 2,439, MobileNetV4-small 32,546. DINOv3 ViT-S/16 ≈ 8 ms fp16 on an A10 (2604.27128). ViT-S/B encoder + small head fits <10 ms with 2–3 views.

## 4. Reward / preference models for robot data
RoboReward (2601.00675: 4B/8B Qwen3 RMs, counterfactual negatives), RoboArena (2506.18123), Robometer/RBM-1M (2603.02115), Robo-Dopamine GRM (2512.23703; 2.0 2608.15680), Large Reward Models (2603.16065), ROBORMBENCH (2609.05401: RMs flip under paraphrase), TimeRewarder (2509.26627), SimpleVLA-RL, πRL (2510.25889: π0.5 77 → 98 % LIBERO), VLA-RL (2505.18719), VLA-RFT (2510.00406), HELP (2607.09776).

## 5. Benchmarks
RoboMemArena (2605.10921: 26 tasks, 151 subtasks, 15.1k keyframe-aligned subtask segments; π0.5 21.5 % TSR vs PrediMem 38.5 %), LIBERO, SimplerEnv, RoboCasa365, RLBench; stage-labelled: RoboCerebra (2506.06677), MEMOBench (2609.07047), RoboMME (2603.04639), RoboProcessBench (2606.13040: 58k QA over phase/contact/progress/outcome), ProcVQA, FailBench, RoboFail, Robo2VLM-1 (2505.15517). Gemini Robotics-ER 2: 57.4 % on 5-bin progress — progress bins are hard even for frontier models.

## 6. Training recipes and label sources
Demo subtask boundaries (SparkVLA, ProcVLM, RoboMemArena, multiview-progress), boundary jitter and perceptual-change weighting; gripper/pose/force heuristics (Robo2VLM, See-Plan-Rewind 2603.09292); episode-level labels only (SAFE, Foresight, Hide-and-Seek, π*0.6); synthetic negatives (RoboMonkey RMSE pairs, RoboReward relabeling, SAFECAST contrast sets); temporal contrast (VIP/LIV, TimeRewarder; survey 2607.21655). Calibration: conformal prediction is the standard, calibrated on successful rollouts, evaluated at fixed false-alarm rates.

## Design implications
**(a) Backbone for <50 ms:** frozen/lightly-tuned ViT-S/B (DINOv2/v3 or SigLIP, 224 px) + small temporal head; read the VLA's own internal features when available (SAFE: zero extra cost); condition on the proposed action chunk (CheckVLA); tiny temporal module to suppress flips; keep the LLM out of the loop.
**(b) Questions first:** 1) subtask done / current stage (drives prompt switching; best labelled); 2) interrupt / re-prompt (failure risk; probes + conformal); 3) which chunk is best (real gains but N verifier passes; MG-Select is the free baseline). Grasp success = special case of stage completion from gripper state.
**(c) Free labels:** every frame of `<primitive>_k_*.hdf5` is "stage k in progress, <k done"; last frames of file k = "k done"; gripper_states → grasp/release; concatenating files by (task, seed) gives per-subtask progress in [0,1]; harness rollouts (results.json stage steps) add policy data with hindsight labels; failures supply negatives; RoboMonkey's RMSE-pair recipe for action selection.
**(d) Evaluation:** frame accuracy/AUROC, VOC, MAE; detection delay at fixed false-alarm rate, AUCPDT, spurious-switch rate, timely recall; splits seen/unseen tasks, sim-calibrate/real-test, cross-policy; closed-loop TSR/CSR uplift when the detector gates prompt switching or chunk selection.
