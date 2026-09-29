# Jev (TypeSafe AI) — research brief, 2026-09-29

Compiled from public sources by a research agent; every claim carries a source tag; "not public" is marked.

## 1. What Jev is
- TypeSafe's first "System One model": "a new class of frontier models built to make fast, structured decisions that software can use directly" — "unstructured state in, typed probabilistic decisions out" [S1]. Launched 15 Sep 2026 [S2][S6].
- Interface: `state` (string / JSON / message array) + typed `questions` (instructions + criteria), all answered "in parallel and in isolation against the same state"; "adding questions barely changes the response time" [S9–S11].
- Question types: **Choice** (≤255 options; choice + per-option probabilities + confidence), **Score** (2–10 rubric levels; interpolated score + rung probabilities), **Noul** (yes/no; one probability, no separate confidence) [S9–S13]. `confidence` is a concentration statistic of the distribution, with recommended routing thresholds (<0.5 human, 0.5–0.9 caution, >0.9 act) [S13].
- Non-generative: "not trained to generate text", max output tokens 0 [S15][S16]. Text only ("No image, audio, or video input") [S9]. 64k context, state ≤32k [S9].
- Latency: 70–500 ms end to end per TypeSafe [S1]; independent ~76–150 ms typical, ~620 ms complex [S17][S18]. Price $0.042/MTok input, output free [S1][S16].

## 2. Architecture (public)
- Exact phrases: "parallel sampler for maximum efficiency"; post-training "Reinforcement Learning for Calibrated Decisions (RLCD)"; "Calibrated: higher confidence means higher accuracy" [S1]. "Jev ingests the state once and evaluates every question against it in parallel" [S9].
- CEO (HN/X): "Architecture is close to the chest"; the "inference algorithm is fundamentally different" from constrained decoding; "Zero-shot over instruction-tuned"; "instead of sacrificing size ... we instead sacrifice text generation for composability" [S14][S24].
- Press: transformer-based, trained exclusively on synthetic data, "cannot hallucinate" = no out-of-schema outputs [S2][S8][S25].
- **Not public:** size, encoder vs decoder, base model, distillation, weights, technical report.
- Third-party probing (Hume, ~10k calls) [S27]: no autoregressive decoding (latency independent of option count, scales with input only); shared state encoding with isolated per-question branches; list-wise option scoring (order- and distractor-sensitive); ECE 0.031 on MMLU. Speculation: causal decoder, sparse MoE ~10B active.

## 3. Training / calibration / benchmarks
- Disclosed: RLCD (reward "honest" probabilities), synthetic-data-only, English primary, not trained on customer data [S1][S9][S28]. Reward function, base model, human labels: not public.
- TypeSafe's eval on 4 internal workflows: Jev 67.8 % agreement with a frontier reference at $0.0004/case, 0.4 s; GPT-5.6 Sol 74.1 % at $0.084, 23 s; Claude Opus 5 73.1 % at $0.18, 38 s [S30].
- Independent: 91.5 % agreement with Fable 5.1 on 6k rubric checks at ~200× lower cost [S21][S31]; social-science labeling −11.6 macro-F1 vs best LLM at 44× lower cost, confidence better calibrated than verbalized LLM confidence [S32]; OOD tickets 75 % with ECE 0.107, type-specific temperature (Choice/Score overconfident, Boolean underconfident) [S33]; phishing 62.6 % single question → 95 % after decomposing into 5 Nouls + logistic regression on 1k labels [S34]; spam 98.3 % zero-shot ≈ TF-IDF LR [S35]; aggregate: "level with mid-price LLMs, 6.5–11.5 points behind frontier" [S17].
- Known weaknesses: literal reading, arithmetic, dates, indirection, distractor-heavy state, no structural invariants across questions [S15].

## 4. Why it blew up
- HN 1,989 points; Vercel: "fastest-adopted model in AI Gateway history" (13 % of paid teams in 24 h) [S14][S38]. Uses: agent tool/subagent selection, continue/retry/stop decisions, risk scoring, guardrails, routing to humans, LLM-judge in Langfuse/LangSmith [S19][S21][S22][S38].
- Karpathy: Jev "revealed latent demand ... under-invested into because of a race to higher intelligence" [S18]. Pushback: calibrated ≠ correct; "confidently wrong" is possible [S29][S14].

## 5. Company
TypeSafe AI, SF, founded 2024. Diogo Almeida (CEO; OpenAI RLHF/InstructGPT/ChatGPT), Sasha Sheng (COO; FAIR), Erik Gafni (CTO). $40M seed led by DCVC [S6][S39–S42].

## 6. Closest public analogues
ModernBERT encoder classifiers (Laya clone, ~35 ms) [S43][S44]; reward-model heads on decoder LMs (ArmoRM, Skywork-Reward); generative verifiers / PRMs (token-probability yes/no readouts) — closest to Noul; open clones: Nimble (Qwen3.5-9B logit readout, cached shared prefix, 90.1 % vs Jev 93.2 %), Kev-0.5B (decision head + LoRA), mini-jev (first-token softmax) [S44][S45].

## Known vs inferred
**Facts:** transformer; non-generative; parallel sampler; state once, questions in parallel & isolated; RLCD; synthetic data; text-only; typed outputs with calibrated probabilities; size not sacrificed. **Inferred:** no autoregressive decoding; shared prefix + isolated question branches; list-wise option scoring; per-type readouts. **Speculation:** causal decoder, MoE ~10B active, distillation.

## Sources
S1 typesafe.ai/blog/introducing-system-one-models-and-jev · S2 techcrunch.com/2026/09/18/… · S3 oreilly.com/radar/will-typesafes-jev-change-how-we-build-ai-applications · S6 finance.yahoo.com (…typesafe-ai-emerges-stealth-40m) · S8 theregister.com/…/typesafe-ai-debuts-model-for-machines-that-plays-doom · S9 docs.typesafe.ai/models.md · S10 vercel.com/docs/ai-gateway/modalities/evaluation · S11 docs.typesafe.ai · S12 developers.cloudflare.com/ai/models/typesafe/jev · S13 docs.typesafe.ai/confidence.md · S14 news.ycombinator.com/item?id=49717558 · S15 docs.typesafe.ai/model-jaggedness/jev-1.13.md · S16 vercel.com/ai-gateway/models/jev · S17 dev.to/aws-builders/jev-after-eight-days-of-independent-tests… · S18 theregister.com/devops/2026/09/23/shut-up-and-calculate… · S19 vercel.com/changelog/typesafe-ai-jev-now-available-on-ai-gateway · S21 langfuse.com/blog/2026-09-18-using-typesafes-jev-for-evals · S22 langchain.com/blog/jev-agent-evals-langsmith · S24 x.com/CompleteSkeptic/status/2100282816143802860 · S25 en.wikipedia.org/wiki/Jev_(AI_model) · S27 archerhume.com/posts/jevs-architecture-unmasked · S28 systemonemodels.org/guides/rlcd-explained · S29 prefactor.tech/blog/jev-calibrated-confidence-is-not-correctness · S30 datacamp.com/blog/system-one-models-jev · S31 reticle.sh/blog/typesafe-jev-playbook · S32 arxiv.org/abs/2609.24574 · S33 github.com/scienthoon/jev-ood-calibration · S34 github.com/anisselbd/jev-phishing-bench · S35 github.com/bitnovus/jev-spam-eval · S38 vercel.com/blog/ai-gateway-jev-model-launch · S39 typesafe.ai/team · S41 pitchbook.com/profiles/company/658924-30 · S43 arxiv.org/abs/2412.13663 · S44 lilting.ch/en/articles/jev-clones-architecture-comparison · S45 kevnu.com/en/posts/typesafe-jev-technical-deconstruction…
