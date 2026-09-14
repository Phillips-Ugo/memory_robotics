# memlayer adapters

Each adapter turns one system's episode logs into `memlayer.ingest.StageEpisode`s.

| adapter | input | status |
|---|---|---|
| `robomemarena.py` | `results.json` from the RoboMemArena harness (needs `scripts/patch_rma_stage_timing.py`) | tested on synthetic logs; real π₀.₅ logs pending M2b |
| `bench` (built in) | `bench.env.EpisodeLog` typed skill events | the benchmark; exact |

Usage:
```python
from memlayer import MemoryLayer
from memlayer.ingest import StageIngester
from memlayer.adapters.robomemarena import load_results

mem = MemoryLayer("site.db"); ing = StageIngester(mem)
for ep in load_results("outputs/rma_pi05_task1/results.json"):
    ing.observe(ep)
print(ing.recall_text())          # "Trouble with 'place' on the tomato sauce: failed 5+ times ..."
print(mem.explain("object:tomato_sauce", "trouble:place"))
```
