# A/A floor for mistral-7b: the split-KV win is REAL, at 17–26x the noise floor

Ran 2026-09-12, nobara-pc, after `splitkv-8000-reanchor-2026-09-12.md` whose §Limitations flagged
the missing floor. Pre-registration: `splitkv-aa-floor-PREREGISTERED.md`, written before any cell.
Binary `serve-cuda-1f224682`, driver `595.91.07`, preflight green on every stage.

## The floor

Both arms identical, run through the same path as a real cell — fresh `serve` per arm, arms adjacent,
order alternating — so the floor contains the fresh-process and first-arm-in-cell effects the real
comparison is exposed to.

| mode | depth | arm A | arm B | ratio | \|1−r\| |
|---|--:|--:|--:|--:|--:|
| aa-off | 3900 | 47.02 | 47.01 | 1.00020 | 0.020% |
| aa-off | 8000 | 33.56 | 33.54 | 1.00045 | 0.045% |
| aa-on | 3900 | 48.09 | 48.11 | 0.99959 | 0.041% |
| aa-on | 8000 | 34.73 | 34.68 | 1.00142 | **0.142%** |

**FLOOR = 0.142%** — the max, as pre-registered, not the mean.

**Running both polarities was load-bearing, not belt-and-braces.** The ON floor at 8000 (0.142%) is
**3.2x** the OFF floor at the same depth (0.045%), exactly the asymmetry the pre-registration
predicted from the ON arm's different kernel and launch geometry. An OFF-only floor would have
understated the bar by a factor of three.

## The effect, measured twice

| cell | 2026-09-12 | repeat | run-to-run | vs floor |
|---|--:|--:|--:|--:|
| mistral-7b:3900 | 1.02402 | 1.02405 | 0.003% | **16.9x** |
| mistral-7b:8000 | 1.03538 | 1.03711 | 0.173% | **26.1x** |

## Verdict: REAL — with one sub-check tripped, recorded rather than waved through

**The pre-registered NOISE branch is decisively ruled out.** Both cells clear 1 + floor in two
independent runs, in the same direction, by 17x and 26x. The 2.4–3.7% is not drift.

**The stability sub-check trips at 8000, marginally.** The pre-registration made reproducibility a
SEPARATE test and set its bar at the A/A floor: run-to-run at 8000 is **0.173%** against a 0.142%
floor. By the letter of the ambiguity clause, that cell's *magnitude* is not established. It does not
touch the REAL finding — the wobble is 1/20th of the effect — but it means the honest quotation at
8000 is **~3.6% ± 0.2pp**, not 3.711%. The 3900 cell has no such caveat: 0.003% run-to-run, which is
1/47th of its own floor.

Recorded because the pre-registration said "both are reported; neither is allowed to stand in for
the other", and a clean primary result is exactly when it is tempting to let a tripped secondary go
unmentioned.

## What is now confirmed

`splitkvNever` is **mis-keyed**. `cuda/resident.go:223-223` sets the class on query-head count alone,
anchored on phi3-mini's nH=32. mistral-7b is **the same nH=32** and measures the opposite sign:

| model | nH | nKV | hd | KV floats/key | @3900 |
|---|--:|--:|--:|--:|--:|
| phi3-mini (MHA) | 32 | 32 | 96 | 3072 | **0.746** |
| mistral-7b (GQA 4:1) | 32 | 8 | 128 | 1024 | **1.024** |

A 25% loss and a 2.4% win at the anchor's own head count. nH cannot be the discriminator. KV traffic
per key remains the plausible mechanism — split-KV buys occupancy, which only helps a latency-bound
kernel, and MHA moves 3x the KV bytes per key.

## Still not established

**One geometry.** "GQA vs MHA" is still confounded with "mistral-7b specifically"; a third nH >= 24
GQA model would separate them, and is not run here. D7 (Qwen2.5-7B, nH=28, GQA) is on mistral's side
by published config, but its weights are not on this box, so that pairing remains inference.

**The mechanism is unmeasured.** KV-traffic-per-key explains the data and was not tested against it;
an ncu occupancy/bandwidth read on both geometries would settle whether phi3-mini is in fact
bandwidth-saturated where mistral is not.
