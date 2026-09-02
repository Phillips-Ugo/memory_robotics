"""Cross-episode memory benchmark, v0 (Phase 2 / experiment X2).

Worlds with secrets: hidden persistent properties that no single episode reveals for
free, but that a robot with working cross-episode memory exploits on the next visit.
The score is the shape of the curve across episodes, not success on any one episode.

v0 uses an abstract skill-level simulator (bench.env) so the memory module is the only
moving part; robosuite skills plug in behind the same interface later.
See docs/benchmark-design.md.
"""
