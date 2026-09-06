"""L3-lite: entity extraction from free-text task strings.

Real deployments hand the memory a sentence, not a Task object. Match known entity
names (drawers, objects, places) in the text — tolerant to underscores/spaces and
plurals — and infer the task kind from verbs. Vocabulary comes from what the memory
has seen (its fact entities) plus anything the caller registers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextTask:
    obj: str
    drawer: str
    kind: str
    text: str


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("_", " ")).strip()


def extract(text: str, objects: list[str], drawers: list[str]) -> TextTask:
    t = " " + _norm(text) + " "
    def find(cands):
        hits = [(t.find(" " + _norm(c) + " "), c) for c in cands if " " + _norm(c) + " " in t]
        hits += [(t.find(" " + _norm(c) + "s "), c) for c in cands if " " + _norm(c) + "s " in t]
        return min(hits)[1] if hits else None
    obj = find(objects)
    drawer = find(drawers)
    if re.search(r"\b(bring|fetch|get|find|take out|hand me)\b", t):
        kind, drawer = "fetch", "table"
    elif drawer is None and re.search(r"\b(away|anywhere|any drawer|somewhere)\b", t):
        kind, drawer = "put_any", "any"
    elif drawer is not None:
        kind = "put"
    else:
        kind, drawer = "put_any", "any"
    return TextTask(obj or "", drawer, kind, text)
