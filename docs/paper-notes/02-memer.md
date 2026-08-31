# MemER: Scaling Up Memory for Robot Control via Experience Retrieval (arXiv 2025)

**Link:** https://arxiv.org/abs/2510.20328
**Code / project page:** https://jen-pan.github.io/memer/
**Read:** <date> · **Phase:** 0 (read) / 1 (reproduce)

*Verified 2026-08-28: hierarchical policy — high-level VLM selects and tracks
relevant keyframes from experience, passes keyframes + recent frames + text
instruction to a low-level policy. Compatible with existing VLAs. Evaluated on
three real-world long-horizon manipulation tasks requiring minutes of memory.*

## 1. What it claims
Using a heurically aporach is better for memory

## 2. How it measures it
conditions on keyfrmae + recent frrame + text instructions and then compare vs human

## 3. What it can't do / limitations

<!-- Hint: the keyframe-selection interface is a candidate template for MY
     memory-layer <-> policy boundary. What breaks if the memory spans days
     instead of minutes? -->
     Computationally expensive and also you feed bad frames so you need to also remove frames!!
     I'm also curiosu as to why the keyfrma eslection was simply the median?

## 4. What I'd test

## One-liner for the log
huericical approach to meory