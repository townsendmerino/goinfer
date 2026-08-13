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
