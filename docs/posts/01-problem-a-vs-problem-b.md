# Post #1 — Robots are about to remember everything, except last week

*Draft 2026-09-02. Status: DRAFT — rewrite first + last paragraphs in my own voice
before posting. Target: LinkedIn + X.*

---

**Robots are about to remember everything — except last week**

Robotics is having its data moment. Figure committed $1B to data collection; every
lab is hoarding teleop hours. The bet is that more experience makes better robots.
But almost nobody is asking the question that decides whether that bet pays off:
*what does the robot do with its experience after the episode ends?*

Spending the last week reading the field's memory benchmarks (RoboMemArena, RoboMME,
MemER) and running one of them on my laptop, I realized the work splits into two very
different problems that everyone lumps together as "robot memory":

**Problem A — memory within an episode.** Did the robot remember which cup it already
filled, 800 steps ago in the same task? This field is maturing fast: real benchmarks
with leaderboards, and honest, humbling numbers — π₀.₅, a frontier VLA, completes
only ~21% of RoboMemArena's memory-dependent tasks; the best memory-augmented model
manages ~38%.

**Problem B — memory across episodes.** Does the robot stop repeating a mistake it
made *last Tuesday*? Does it know this apartment's drawer sticks, that this
customer's mug is heavier than it looks? Here's what shocked me: **there is no
standard benchmark for this. At all.** The closest work is on the LLM-agent side
(Voyager, MemGPT, Mem0) — not robotics.

And Problem B is the one deployment actually runs on. The facts a deployed robot
needs change faster than any fleet retraining cycle — they're local, mutable,
sometimes private to one customer's site. They can't live in the weights. They need
a memory layer: store, consolidate, retrieve, and — the hard part nobody talks
about — *revise*, because the drawer eventually gets fixed and the robot has to stop
avoiding it.

So that's what I'm building over the next six months: a benchmark that measures
cross-episode memory, and a library that passes it. I'm a beginner learning this
stack end to end, and I'm posting everything — receipts so far: my first VLA eval
(65% on PushT, n=20, and a lesson on why you always report a confidence interval),
and a memory benchmark's full evaluation harness running on a MacBook.

Next post: the silent bug that made a pretrained policy score 0% — and what it
taught me about never trusting a network's inputs.

---

## Queue

- #2 — the normalization bug (research log, Day 0 part 2)
- #3 — "How to build a benchmark for something nobody measures" (docs/benchmark-design.md)
- #4 — M2: reproducing π₀.₅ on RoboMemArena for $X on a rented 4090
