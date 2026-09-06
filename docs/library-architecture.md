# Memory layer — architecture (Phase 4 design, v0.1)

*2026-09-05. Everything here is what the benchmark prototype already does, written
down as a system; the library is the prototype with persistence, a schema, and better
rules. Storage is commodity; the rules and the schema are the product.*

## API (unchanged since day one)

```python
mem = MemoryLayer(path="site_4B.db")
mem.observe(episode)                # after each task: raw log in, facts updated
ctx = mem.recall(task, obs)         # before each task: structured beliefs
txt = mem.recall_text(task, obs)    # same, rendered for an LLM/VLA prompt
mem.snapshot() / mem.explain(fact)  # audit: why do we believe this?
```

## Three tiers

| tier | holds | store | queried when |
|---|---|---|---|
| 1 raw episodes | actions, observations/keyframes/video, outcome, timestamp | files (JSONL/Parquet + mp4), append-only | audit, training, re-consolidation |
| 2 episode summaries | task, entities touched, events (`open(middle)->jam`), success, steps | SQLite table + small embedding index (`sqlite-vec` / FAISS) | last-k and retrieval baselines; fallback fuzzy recall |
| 3 facts | `(entity, attribute, value, evidence, last_confirmed, last_contradicted, confidence)` | SQLite table | every `recall()` |

Fleet scale later: tiers 2–3 → Postgres + pgvector, tier 1 → object storage. A
deployment detail, not a design decision.

## Fact schema

```
entity        "drawer:middle" | "object:butter" | "place:kitchen_4B" | "task:put_away"
attribute     sticky | heavy | location | fast | preferred_drawer | ...
value         bool | str
evidence      count of direct confirmations
last_confirmed   episode index / timestamp of the last confirming observation
last_contradicted   ... of the last contradicting observation
confidence    f(evidence, age, contradictions)   [0, 1]
source        list of tier-2 episode ids (explainability)
```

## Consolidation (episode -> facts)

Rule-based over typed events, exactly as `ConsolidatedKB.observe` does today:
- a failure with a known cause writes/strengthens a fact (`jam` -> sticky=True)
- a success where a fact predicted failure *contradicts* it (`open->ok` on a
  "sticky" drawer): confidence drops, `last_contradicted` set; after k
  contradictions the fact flips, not deletes (history kept)
- success-only evidence (`place->praised`, a cheap `open`) writes positive facts
- robust actions (`pull_hard`) are *not* evidence (Day 4b: playing safe produces
  no evidence) — this is why revision needs probing

**Branching-factor rule (Day 9c):** keep every failure; keep every success that
either contradicts a stored fact or confirms a *choice* (praise). Discard successes
that merely repeat expectations — that is where the 4% storage comes from.

## Revision policy (the moat)

- **Probe schedule:** a fact unconfirmed for `probe_after` episodes is offered to the
  planner as "re-test cheaply once" (today: 8; the central knob — X4 sweeps it).
- **Age decay:** confidence decays with episodes since `last_confirmed`; below a
  threshold the fact is advisory, not binding.
- **Contradiction handling:** one contradiction downgrades; two flip; the old value is
  kept as history with its evidence window (the world may change back).
- **Change detection:** a burst of contradictions across facts of the same entity
  flags "this entity changed" and lowers confidence on its sibling facts.

## Retrieval

1. **Entity-keyed** first: facts about the task's objects/places/drawers.
2. **Task-kind rules**: fetch -> location facts + not-in exclusions; put_any -> preference,
   sticky, fast; put -> sticky(drawer), heavy(object).
3. **Fallback**: embedding search over tier-2 summaries for entities the task
   mentions by other names; return at most k episodes.
Returned context is small on purpose: measured retrieval precision of raw
episode search is <20% (Day 7d); facts are the compressed, relevant form.

## The bridge to real robots: episode understanding

The benchmark emits typed events. A real robot emits video + actions. The missing
component is an **episode-understanding model**: a VLM that watches an episode (or its
keyframes — MemER/RoboMemArena style) and emits the typed events + entities the
consolidator expects (`{skill, target, outcome}`), plus task success. This is the
hardest engineering piece and the one most likely to be a paper on its own. Until it
exists, the library ingests structured logs (which many deployments already have:
task planners, skill libraries, teleop annotations).

## Milestones

- **L0** ✓: `bench/memory.ConsolidatedKB` — dict of facts, probe rule, ~60 lines.
- **L1** ✓ (5 Sep): same logic on SQLite, `MemoryLayer` class, explain(), text renderer;
  passes the benchmark identically to L0 (AUC 0.865 both, `bench/evaluate.py`).
- **L2** ✓ (5 Sep): `RevisionPolicy` — per fact type: schedule / contradiction / never;
  confidence = f(evidence, age, contradictions). X4: never-probe is optimal for
  sticky/heavy in the current world (see log Day 9e); change-detection TODO.
- **L3** ◐ (5 Sep): entity extraction from free-text tasks (`memlayer/entities.py`,
  vocabulary = what the memory has seen); embedding fallback TODO.
- **L4**: episode-understanding adapter for one VLA log format (RoboMemArena's
  keyframe annotations as the first target).
- **v0.1 release**: L1–L3 + benchmark adapter + docs.
