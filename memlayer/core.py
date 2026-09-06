"""L1: the consolidated fact store on SQLite.

Same rules as bench.memory.ConsolidatedKB (the benchmark prototype), persisted:
  - typed events -> facts (entity, attribute, value, evidence, last_confirmed,
    last_contradicted, source episodes)
  - robust actions are not evidence; a success where a fact predicted failure is a
    contradiction (recorded, and the fact flips on direct evidence)
  - probe rule: a fact unconfirmed for `probe_after` episodes is offered for re-test
Tier-2 episode summaries are stored too (for audit / explain / raw-log baselines).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bench.memory import Beliefs


@dataclass
class RevisionPolicy:
    """Per fact type (Day 9e): facts whose staleness merely costs efficiency are
    re-tested on a schedule (or never, when P(change)*cost(stale) < cost(probe));
    facts whose staleness fails the task (location, house rule) are revised by the
    contradiction the failure produces, no schedule needed."""

    probe_after: dict[str, int | None] = None  # attribute -> episodes between re-tests (None = never)
    decay_after: int = 40  # confidence halves every this many episodes without confirmation

    def __post_init__(self):
        if self.probe_after is None:
            self.probe_after = {"sticky": 8, "heavy": 8, "location": None, "preferred_drawer": None, "fast": None}

    def should_probe(self, attribute: str, age: int) -> bool:
        k = self.probe_after.get(attribute)
        return k is not None and age >= k

    def confidence(self, evidence: int, age: int, contradictions: int = 0) -> float:
        base = 1 - 0.5 ** evidence
        return base * 0.5 ** (age / self.decay_after) * 0.5 ** contradictions

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    idx INTEGER PRIMARY KEY, task TEXT, kind TEXT, obj TEXT, drawer TEXT,
    success INTEGER, steps INTEGER, events TEXT, text TEXT
);
CREATE TABLE IF NOT EXISTS facts (
    entity TEXT, attribute TEXT, value TEXT,
    evidence INTEGER DEFAULT 0, last_confirmed INTEGER DEFAULT -1, last_contradicted INTEGER DEFAULT -1,
    sources TEXT DEFAULT '[]',
    PRIMARY KEY (entity, attribute)
);
"""


@dataclass
class Fact:
    entity: str
    attribute: str
    value: str
    evidence: int
    last_confirmed: int
    last_contradicted: int
    sources: list[int]


class MemoryLayer:
    name = "memlayer-L1"

    def __init__(self, path: str | Path = ":memory:", probe_after: int | None = 8,
                 policy: RevisionPolicy | None = None) -> None:
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)
        self.policy = policy or RevisionPolicy(probe_after={"sticky": probe_after, "heavy": probe_after,
                                                            "location": None, "preferred_drawer": None, "fast": None})
        self.t = int(self.db.execute("SELECT COALESCE(MAX(idx), 0) FROM episodes").fetchone()[0])

    # ---- tier 2/3 primitives ------------------------------------------------
    def _fact(self, entity: str, attribute: str) -> Fact | None:
        r = self.db.execute("SELECT entity, attribute, value, evidence, last_confirmed, last_contradicted, sources "
                            "FROM facts WHERE entity=? AND attribute=?", (entity, attribute)).fetchone()
        return Fact(*r[:6], json.loads(r[6])) if r else None

    def _write(self, entity: str, attribute: str, value: str, confirmed: bool, ep: int) -> None:
        f = self._fact(entity, attribute)
        if f is None or f.value != value:
            # new fact, or direct evidence of a different value: the fact flips, keeping the
            # previous value's evidence in the contradiction timestamp
            self.db.execute(
                "INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?)",
                (entity, attribute, value, 1, ep, f.last_confirmed if f else -1, json.dumps([ep])))
        else:
            self.db.execute(
                "UPDATE facts SET evidence=evidence+1, last_confirmed=?, sources=? WHERE entity=? AND attribute=?",
                (ep, json.dumps((f.sources + [ep])[-20:]), entity, attribute))

    def facts(self) -> list[Fact]:
        return [Fact(*r[:6], json.loads(r[6])) for r in self.db.execute(
            "SELECT entity, attribute, value, evidence, last_confirmed, last_contradicted, sources FROM facts")]

    # ---- API ----------------------------------------------------------------
    def observe(self, log) -> None:
        ep = log.episode_idx
        self.t = ep
        obj = log.task.obj
        for e in log.events:
            if e.skill == "test_drawer":
                self._write(f"drawer:{e.target}", "sticky", str(e.outcome == "sticky"), True, ep)
            elif e.skill == "test_object":
                self._write(f"object:{e.target}", "heavy", str(e.outcome == "heavy"), True, ep)
            elif e.skill == "open" and e.outcome in ("ok", "jam"):
                self._write(f"drawer:{e.target}", "sticky", str(e.outcome == "jam"), True, ep)
                if e.outcome == "ok":
                    self._write(f"drawer:{e.target}", "fast", str(e.steps <= 2), True, ep)
            elif e.skill == "pick" and e.outcome in ("ok", "drop"):
                self._write(f"object:{e.target}", "heavy", str(e.outcome == "drop"), True, ep)
            elif e.skill == "look_in" and e.outcome == "found":
                self._write(f"object:{obj}", "location", e.target, True, ep)
                self.db.execute("DELETE FROM facts WHERE entity=? AND attribute=?", (f"object:{obj}", f"not_in:{e.target}"))
            elif e.skill == "look_in" and e.outcome == "empty":
                self._write(f"object:{obj}", f"not_in:{e.target}", "True", True, ep)
                f = self._fact(f"object:{obj}", "location")
                if f and f.value == e.target:
                    self.db.execute("DELETE FROM facts WHERE entity=? AND attribute='location'", (f"object:{obj}",))
            elif e.skill == "place" and e.outcome == "praised":
                self._write("rule:put_any", "preferred_drawer", e.target, True, ep)
                self.db.execute("DELETE FROM facts WHERE entity='rule:put_any' AND attribute LIKE 'rejected:%'")
            elif e.skill == "place" and e.outcome == "rejected":
                self._write("rule:put_any", f"rejected:{e.target}", "True", True, ep)
                f = self._fact("rule:put_any", "preferred_drawer")
                if f and f.value == e.target:
                    self.db.execute("DELETE FROM facts WHERE entity='rule:put_any' AND attribute='preferred_drawer'")
            # pull_hard / pick_two_hand: no evidence
        self.db.execute("INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?)",
                        (ep, log.task.text, getattr(log.task, "kind", "put"), obj, log.task.drawer, int(log.success),
                         log.steps, json.dumps([e.__dict__ for e in log.events]), log.text))
        self.db.commit()

    def _fresh(self, f: Fact) -> bool:
        return not self.policy.should_probe(f.attribute, self.t - f.last_confirmed)

    def confidence(self, f: Fact) -> float:
        return self.policy.confidence(f.evidence, self.t - f.last_confirmed, int(f.last_contradicted > f.last_confirmed))

    def recall(self, task, initial_obs: dict | None = None) -> Beliefs:
        b = Beliefs()
        for f in self.facts():
            kind, _, name = f.entity.partition(":")
            if f.attribute == "sticky" and f.value == "True":
                if self._fresh(f):
                    b.sticky_drawers.add(name)
                elif name == task.drawer:
                    b.probe_drawers.add(name)
            elif f.attribute == "heavy" and f.value == "True" and name == task.obj:
                (b.heavy_objects if self._fresh(f) else b.probe_objects).add(name)
            elif f.attribute == "fast" and f.value == "True":
                b.fast_drawers.add(name)
            elif f.attribute == "location" and name == task.obj:
                b.object_in[name] = f.value
            elif f.attribute.startswith("not_in:") and name == task.obj:
                loc = self._fact(f.entity, "location")
                if loc is None or f.last_confirmed > loc.last_confirmed:
                    b.object_not_in.setdefault(name, set()).add(f.attribute.split(":", 1)[1])
            elif f.attribute == "preferred_drawer":
                b.preferred_drawer = f.value
            elif f.attribute.startswith("rejected:"):
                pref = self._fact("rule:put_any", "preferred_drawer")
                if pref is None or f.last_confirmed > pref.last_confirmed:
                    b.rejected_drawers.add(f.attribute.split(":", 1)[1])
        return b

    def recall_text(self, task, initial_obs: dict | None = None) -> str:
        lines = []
        for f in self.facts():
            kind, _, name = f.entity.partition(":")
            age = self.t - f.last_confirmed
            tail = f"(evidence: {f.evidence}, last confirmed {age} episode(s) ago)"
            if f.attribute == "sticky":
                lines.append(f"The {name} drawer {'sticks' if f.value == 'True' else 'opens normally'} {tail}.")
            elif f.attribute == "heavy":
                lines.append(f"The {name} {'is heavy' if f.value == 'True' else 'is light'} {tail}.")
            elif f.attribute == "fast" and f.value == "True":
                lines.append(f"The {name} drawer opens quickly {tail}.")
            elif f.attribute == "location":
                lines.append(f"The {name} was last found in the {f.value} drawer {tail}.")
            elif f.attribute == "preferred_drawer":
                lines.append(f"Things put away belong in the {f.value} drawer {tail}.")
        return "\n".join(lines)

    def explain(self, entity: str, attribute: str) -> str:
        f = self._fact(entity, attribute)
        if f is None:
            return f"no fact for {entity}.{attribute}"
        eps = ", ".join(str(i) for i in f.sources[-5:])
        return (f"{entity}.{attribute} = {f.value} (confidence {self.confidence(f):.2f}): {f.evidence} observation(s), "
                f"last confirmed at episode {f.last_confirmed}, last contradicted at {f.last_contradicted}; evidence episodes: {eps}")

    def bytes_stored(self) -> int:
        return 40 * self.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def surfaced(self, task) -> list[int]:
        return []
