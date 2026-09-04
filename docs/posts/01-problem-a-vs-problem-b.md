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
customer's mug is heavier than it looks? Here's what shocked me: exactly **one**
benchmark touches this (RoboMME-Interference, June 2026) — and it hands the robot a
demonstration video and checks whether it survives distractors. **Nobody measures
whether a robot learns a fact from its own experience, and nobody measures what
happens when that fact stops being true.** The rest of the closest work is on the
LLM-agent side (Voyager, MemGPT, Mem0) — not robotics.

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

## X version (chosen hook: contrarian; each paragraph = one tweet if threaded)

Figure is spending $1B on robot data. Almost none of it will be remembered.

Not a knock on Figure — it's a gap in the whole field. Spent the week reading every
robot-memory paper I could find (RoboMemArena, RoboMME, MemER), and they all solve
the same half of the problem.

Half A: memory within a task. Did the robot remember which cup it filled 800 steps
ago? Real benchmarks, real leaderboards. Frontier VLAs like π₀.₅ score ~21% on them.
Hard, but crowded.

Half B: memory across tasks. Does the robot stop jamming the same sticky drawer it
jammed yesterday? Does it know *this* apartment's mug is heavy?

Exactly one benchmark touches Half B (RoboMME-Interference, June 2026) — and it
hands the robot a demo video and checks if it survives distractors. Nobody tests
whether a robot learns a fact from its *own* experience. And nobody tests what
happens when the fact stops being true.

And Half B is what deployment actually runs on. "This drawer sticks" is local,
changes weekly, and is private to one customer — it can't live in the model weights.
It needs a memory layer that stores, retrieves, and — the hard part — revises when
the drawer gets fixed.

So that's what I'm building: the benchmark that measures cross-episode memory, then
the library that passes it.

I'm a beginner learning this stack end to end and posting everything. Receipts so
far: first VLA eval (65% on PushT, n=20 — plus a lesson in why you always report a
confidence interval) and a memory benchmark's full eval harness running on my
MacBook.

Next: the silent bug that made a pretrained policy score 0%.

## Queue

- #2 — the normalization bug (research log, Day 0 part 2)
- #3 — "How to build a benchmark for something nobody measures" (docs/benchmark-design.md)
- #4 — M2: reproducing π₀.₅ on RoboMemArena for $X on a rented 4090


# Post #2 — Week 2: the memory test (drafted 2026-09-04, refine in your voice)

I gave a robot a memory. Its success rate went from 49% to 80%.

Here's the test I built.

A kitchen with secrets: one drawer sticks, one box is heavier than it looks. The
robot isn't told any of this. It does 30 tasks in a row in the same kitchen, and the
only way to learn the secrets is to fail and remember.

Left video: no memory. It tries the gentle pull first (the cheap move), the drawer
jams, then it pulls hard. It grabs the butter lightly, drops it, then grips firmly.
Every task is day one.

Right video: same robot, same task, but it remembers this kitchen. Straight to the
hard pull. Straight to the firm grip.

Then the twist. Halfway through the 30 tasks I quietly fix the sticky drawer and
make a different one stick.

The memory that remembered everything perfectly was the only one that got *worse*
when the world changed, and took 40% more wasted actions on the drawer that now
works. With a language model reading the memory instead of a fixed script, it
collapsed from 99% to 66%. The memory that periodically double-checks its own
beliefs didn't drop.

Real physics (MuJoCo, Panda arm), 600 episodes per memory. Code, chart, videos:
github.com/Phillips-Ugo/memory_robotics

Video: docs/figures/side_by_side_2026-09-04.mp4
