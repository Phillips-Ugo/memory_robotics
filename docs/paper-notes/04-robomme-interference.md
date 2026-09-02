# RoboMME-Interference: memory under cross-session interference (arXiv, June 2026)

**Link:** https://arxiv.org/html/2606.22338v2
**Read:** 2026-09-02 (found by Belu; verified against the HTML) · **Phase:** 0/2
**Why it matters:** the closest existing thing to a Half B benchmark. Defines what
we are NOT building, precisely.

## 1. What it claims

Robot memory systems that work with a clean history decay as unrelated sessions
accumulate — the robotics analogue of the LLM long-context finding (LongMemEval:
~30-point recall drop). Their related work states the gap in our words: "Robot memory
has no comparable benchmark"; prior work "primarily measures memory using a single
episode; none measures whether that memory keeps up as the session history grows."

## 2. How it measures it

- Built on RoboMME. A *session* = a separate episode with environment reset (so this
  IS cross-episode).
- History buffer = one relevant demonstration + k ∈ {0,1,3,7} unrelated sessions
  (each = 32 stored frames) + the query episode.
- 9 task families (imitation, video-conditioned reference, permanence), 50 test
  episodes each; 9 memory variants (FrameSamp/TokenDrop × Context/Modulator/Expert,
  recurrent TTT, π₀.₅ baseline).
- Key numbers: FrameSamp-Modulator **45.3% at k=0 → 19.3% at k=7**. Retrieval fix
  (split buffer at visual scene changes, SigLIP-embed sessions + current obs, cosine
  threshold 0.923, pass only the best match) → **44.7–44.9% at every k**.

## 3. What it can't do / limitations (the ones that define our lane)

- **Memory is given, not earned.** The remembered content is a demonstration video
  the query explicitly references ("re-pick what was picked in the demo"). The robot
  never discovers a fact through its own success/failure. Demonstration memory, not
  experiential memory.
- **Nothing changes.** Distractors are *irrelevant*, never *conflicting*. No stale
  facts, no revision, no recovery metric. A world that never changes only tests
  retrieval — and that's what they found: a retrieval-only fix fully solves it.
- Their own caveat: the fix "depends on the query resembling its demonstration."
  Visual similarity can't tell a fixed drawer from a sticky one.
- Sim only; single checkpoint per system; separable-demonstration tasks only.
- Stated future work = our roadmap: "one continuous stream with no clean session
  boundaries"; "experience spread across several sessions" (= consolidation).

## 4. What I'd test

- Their retrieval fix on OUR benchmark: it should hold while the world is static and
  fail at the first change event. That's X4's headline chart, with a published
  method as the foil.
- Reuse their interference axis (k unrelated episodes) as one dimension of ours —
  interference × experience × change is the full space; they cover one axis.

## One-liner for the log

The first cross-episode robot memory benchmark — but memory is handed to the robot
as a demo and the world never changes, so retrieval alone solves it; the
earned-memory and revision axes are still open, and that is exactly our benchmark.
