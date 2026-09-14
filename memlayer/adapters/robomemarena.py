"""RoboMemArena harness logs -> memlayer StageEpisodes.

Input: the `episodes` list from `eval_common.run_eval` (one dict per episode with
"ep", "seed", "TSR", "CSR", "stage_done": {name: bool}), optionally with
"stage_steps": {name: step_index_when_completed} if the harness records it
(see scripts/patch_rma_stage_timing.py), plus the task prompt.

Stage names are "<NN>_<Verb>_<Object>_<Place>" (e.g. "02_Place_Tomato_Basket",
"03_Open_Middle_Drawer"). Sequential stage checks mean a stage after the first
failure was never attempted; we mark those attempted=False so they carry no evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..ingest import StageAttempt, StageEpisode, parse_stage_name

# words in stage names -> canonical entity keys (BDDL object names)
ENTITY_HINTS = {
    "tomato": "object:tomato_sauce", "tomato sauce": "object:tomato_sauce",
    "cookies": "object:cookies", "butter": "object:butter", "popcorn": "object:popcorn",
    "cream": "object:cream_cheese", "pudding": "object:chocolate_pudding", "chocolate": "object:chocolate_pudding",
    "milk": "object:milk", "orange juice": "object:orange_juice", "juice": "object:orange_juice",
    "basket": "place:basket", "top drawer": "place:top_drawer", "middle drawer": "place:middle_drawer",
    "bottom drawer": "place:bottom_drawer", "cabinet2": "place:cabinet2", "microwave": "place:microwave",
    "frypan": "place:frypan", "drainer": "place:bowl_drainer", "bowl": "place:bowl", "mug": "place:mug",
}


def episode_from_harness(ep: dict, task_prompt: str, sequential: bool = True) -> StageEpisode:
    stage_done: dict[str, bool] = ep["stage_done"]
    stage_steps: dict[str, int] = ep.get("stage_steps", {})
    stages, prev_t, blocked = [], 0, False
    for name, passed in stage_done.items():
        verb, ents = parse_stage_name(name, ENTITY_HINTS)
        steps = None
        if passed and name in stage_steps:
            steps = int(stage_steps[name]) - prev_t
            prev_t = int(stage_steps[name])
        stages.append(StageAttempt(name=name, verb=verb, entities=ents, passed=bool(passed),
                                   steps=steps, attempted=not blocked))
        if sequential and not passed:
            blocked = True  # later stages were never reachable
    return StageEpisode(episode_idx=int(ep["ep"]), task=task_prompt, stages=stages,
                        success=bool(ep.get("TSR", 0) >= 100.0), total_steps=ep.get("total_steps"),
                        meta={"seed": ep.get("seed"), "CSR": ep.get("CSR")})


def load_results(path: str | Path) -> list[StageEpisode]:
    """Load a run_eval result JSON ({"prompt":..., "episodes":[...]}) into StageEpisodes."""
    d = json.loads(Path(path).read_text())
    return [episode_from_harness(e, d["prompt"]) for e in d["episodes"]]
