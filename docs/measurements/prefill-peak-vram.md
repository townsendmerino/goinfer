# Prefill's peak VRAM against the A13 refusal floor

**Date** 2026-08-13 · **Box** RTX 2070 SUPER, 8 GB · **Question** can the shipped prefill path reach
the A13 trigger?

## The trigger, restated

A13's poisoning stimulus is **exhaustion**: allocate until even a 1 MiB request is refused (~144 MiB
free), and a later launch on that context returns success and writes nothing. Deterministic, 5/5.

The earlier percentage bands (`≤12%` clean, `≥25%` poisons) are **withdrawn** — a re-run of that
synthetic hold-and-release at a *fixed* percentage returned `C C C C P C`, so the probe is
intermittent and the non-monotonicity was its own noise. Prefill is therefore closed against a direct
measurement of its own peak rather than against any band.

## Method

`TestPrefillCrossover` — the real dense **qwen2.5-coder-1.5b-instruct-q4_k_m**, resident, prompts of
128 / 512 / 1024 / 2048 tokens, 3 repetitions each, prefill + 64 decode steps per point. Sampled
under `GOINFER_VRAM_TRACE=1` (`cuda/vramtrace_test.go`): `cuMemGetInfo` from an independent goroutine
on the same primary context, in-process, throughout the run. **445 samples.**

The minimum across all samples is the peak-consumption point — the moment prefill's `scratch` (the
allocation `cuda/prefill.go` describes as "hundreds of MB at M=3000") is live on top of the resident
weights and KV.

## Result

| | bytes | MiB |
|---|---|---|
| free, MAX during the run | 7 665 090 560 | 7310.2 |
| **free, MIN during the run** | **6 031 634 432** | **5752.2** |
| refusal floor (`TestAllocFloor` drains to) | 151 191 552 | 144.2 |
| **margin** | | **5608.0** |

**Prefill's worst moment leaves 39.9× the refusal floor free.** It does not approach the trigger; it
is not in the same order of magnitude as the trigger.

## CORRECTION (2026-08-13, same day): the 1.5B was not the worst case

The run above uses a **1.5B**. The gate then failed `TestB2DenseFlagship`, which runs prefill against
a **7B q4_k_m** — ~4.9 GB of an 8 GB card — and the failure prompted the obvious question: prefill's
scratch competes with the resident weights, so a bigger model leaves less headroom, and 39.9× was
measured on a model chosen for convenience rather than for pressure.

Re-measured under the same tracer, `TestB2DenseFlagship`, 7B resident, prompts to M=2048, **1347
samples**:

| | MiB |
|---|---|
| free MAX during the run | 7310.2 |
| **free MIN during the run** | **1820.2** |
| refusal floor | 144.2 |
| **margin** | **1676.0 — 12.6× the floor** |

**The conclusion is unchanged; the number was three times too generous.** Prefill against the largest
model this box runs still peaks at 12.6× the refusal floor. Quote **12.6×**, not 39.9× — the latter
is a property of a small model, not of the prefill path.

(The gate failure that prompted this was NOT prefill's fault: `GOINFER_HEAVY_TESTS=1` had been
exported into group 2b, so heavy tests ran there and left VRAM held. B2 passes cleanly on a correct
invocation. The re-measurement stands on its own regardless.)

## What this closes

Prefill was the one enumerated shipped path that both allocates largely inside a live context *and*
keeps using that context afterwards. It is now closed **by measurement of the quantity that actually
matters** (free VRAM at peak) rather than by an argument about allocation size against a band that
has since been withdrawn.

It does not close A13 — the trigger is real and reproducible, and the thread factor (why a pinned
test goroutine poisons where the resident's executor does not) remains uncharacterised. It closes the
question of whether any shipped path *reaches* it. See `docs/QUEUE.md` A13 for the five-path
enumeration this measurement is item 1 of.

## Caveat, stated rather than buried

The harness's largest prompt is **M = 2048**, not the M ≈ 3000 the `prefill.go` comment names. Scratch
is linear in M, so M = 3000 would consume roughly 1.46× this run's scratch. Even attributing the
*entire* 1558 MiB peak-to-trough swing to scratch and scaling it by 1.46 leaves ≈ 5030 MiB free —
still 35× the floor. The conclusion is not sensitive to the gap.
