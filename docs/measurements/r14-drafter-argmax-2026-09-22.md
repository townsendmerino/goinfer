# R14 step 0: what the spec loop's full-logits download + host argmax costs, measured in its round

`docs/tasks/red-october.md` R14: the drafter's `DraftTokens` syncs, downloads the whole `M×vocab` logits block and
runs a serial host argmax per row, where the main decode path has an on-device `argmax_reduce` and a 4-byte readback.
RTX 2070 SUPER, driver 595.91.07, Qwen3-4B int4 target + Qwen3-4B-DFlash (f32) drafter — the gate-3 pairing, its
`code` (w=7) and `math` (w=8) suites, 96 new tokens per prompt, `dflashLoop` (the resident round loop).

## Scope correction, read from the code before measuring

The brief names one site. There are two with the identical shape, and every round pays both:

1. `cuda/drafter.go` `DraftTokens` — the drafter head over the block's `M = w−1` rows.
2. `cuda/prefill.go` `batchedHeadArgmax` — the VERIFY head over `M = w` rows (`PrefillLastNArgmax`), whose own comment
   already says "a batched argmax kernel is a later, separate ~1 ms".

Prompt prefill is NOT exposed: the batched tail heads only the last row (`tailLast`) unless every row's argmax was
asked for, so the `M×vocab` download is confined to the spec paths. `FuseContext` and the block forward download
`n×hidden` and `M×hidden` rows — that is the drafter's data flow, not a logits download, and out of R14's scope.

The reduction is the same on both sides: strict `>` from element 0 (lowest index on ties) on the host, the same
tie-break in `argmax_reduce` on the device. A batched device kernel with that reduction picks the same token by
construction; the lossless contract is a bit-identity check, not a fidelity lane.

## Instrument

`headArgProf` (`SetHeadArgProfForTest`), nil by default: wall time of the three phases at both sites — the stream
drain (which exists either way: something must wait for the head GEMV before any readback), the D2H, the host
argmax loop — plus call and row counts. `TestR14DrafterArgmaxProfile` runs the gate-3 loop with it armed after a
discarded warm-up and reports the phases per round and as a share of the spec wall-clock.

## Pre-registered decision rule (written before the run)

The removable cost is **D2H + host argmax** (the sync stays; a device argmax still needs a drain before its 4-byte
readback). Its share of the spec wall is the ceiling for the build.

- **Kill without building if D2H + host < 1.5% of the spec wall** on both suites.
- **Build if ≥ 1.5%:** one `argmax_rows` kernel (grid = M blocks of `argmax_reduce`'s reduction, one row each,
  `M` ints out), used at both sites; `argmax.ptx` regenerated. Gates: per-row ids equal to the host loop's on every
  call of a full gate-3 run (a test asserts it), and the emitted token sequence byte-identical to plain greedy
  (`TestDFlashLoop_lossless`, unchanged).
- **Ship** at a paired-median spec-wall speedup ≥ 2% (ABBA, same prompts, same loaded model, 6 pairs) — the tail is
  a fixed per-round cost, so the speedup should land near the measured share; park 1–2%; kill < 1%. Within 5% of a
  threshold → parked.

## Result 1 — step 0 (`TestR14DrafterArgmaxProfile`, log `r14-drafter-argmax-2026-09-22-profile.log`)

| suite | w | rounds | ms/round | tail calls (rows) | sync /round | **D2H /round** | **host argmax /round** | **D2H+host share** |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| code | 7 | 41 | 30.4 | 84 (582, 6.9/call) | 4.05 ms (13.3%) | 1.86 ms | 2.67 ms | **14.9%** |
| math | 8 | 33 | 35.3 | 68 (557, 8.2/call) | 5.19 ms (14.7%) | 2.18 ms | 3.46 ms | **16.0%** |

**Build funded, by an order of magnitude over the 1.5% line.** The removable tail is ~15% of every spec round.
Two things the phases say individually:

- The D2H moves at **4.65–4.70 GB/s** — a fresh pageable Go slice per call (`make([]float32, M*vocab)`), staged
  through the driver's bounce buffer, not a pinned transfer. 4.2 MB per call at that rate is the 1.9–2.2 ms.
- The host argmax runs at **1.2–1.4 ns/element**, serial, over ~1.0 M elements per call: 2.7–3.5 ms/round. Both
  sites together read ~8.5 MB of logits per round to produce 15 ints.
- The sync (13–15%) is NOT the tail's cost: it is the drain the whole batched forward needs before any readback, and
  it stays under a device argmax (which then reads back `4·M` bytes instead of `4·M·vocab`).

## Result 2 — the build and the A/B (`TestR14DrafterArgmaxAB`, log `r14-drafter-argmax-2026-09-22-ab.log`)

**Built and shipped, default.** `argmax_rows` (`cuda/argmax.cu`): `argmax_reduce`'s per-thread strided scan and
lowest-index tie-break, one block per row, `M` ints out. `argmax.ptx` regenerated at the same NVRTC (12.9.86) it
was first built with — the existing `argmax_reduce` entry is byte-identical before and after, checked. Both tails
now go through one `cudaResident.argmaxRows` (`DraftTokens` and `batchedHeadArgmax`): launch, drain, `4·M`-byte
readback. The old tail survives only as the A/B's do-nothing arm (`SetHostArgmaxForTest`), plus a check mode that
runs both on the same logits and compares row-for-row.

**Correctness.** 582 device calls compared row-for-row against the host loop on the same logits: all equal. The
two arms' emitted token sequences are identical on both prompts. Existing gates re-run on the device path:
`TestPrefillLastNArgmax_matchesPerRow` (every verify row bit-identical to a sequential `Forward`),
`TestDFlashLoop_lossless`, `TestArgmaxTieBreak`, `TestGenerateBlockSpec_production`, `TestBlockSpecStream`,
`TestResidentDrafter_{block,fuse}Parity`, `TestBatchedCapture_matchesPerToken` — all PASS.

**Speed** (code suite, w=7, 2 prompts × 96 tokens per arm, ABBA, 6 pairs, one loaded model):

| pair | host tail | device tail | speedup |
|---:|---:|---:|---:|
| 0–5 | 1288–1309 ms | 1050–1055 ms | 1.221–1.247× |

**Paired median 1.234×; pooled 7791 → 6311 ms. Verdict by the pre-registered rule: SHIP (≥ 2%).**

The realised gain (19% of the host-arm wall) EXCEEDS the 14.9% the profiler attributed to D2H + host argmax. That
is not a measurement contradiction but an instrument boundary: each host-arm call allocated a fresh 4.2 MB
`[]float32` (~350 MB of garbage per suite), and the GC work that buys lands outside the three timed windows. The
profiler measured the tail's own phases correctly and under-counted the tail's total cost; the A/B, which measures
the round, is the number that stands.

## What this does not cover

- The drain (13–15% of the round) is untouched — it is the batched forward finishing, not the tail.
- `FuseContext`/`DraftBlock` still download `n×hidden` / `M×hidden` rows to the host and re-upload them to the
  target: the drafter's data flow between the two runners. That round trip is a separate question, not R14's.
- One model/pairing (Qwen3-4B + DFlash). The mechanism is per-row-of-vocab, so the share scales with `vocab·M` and
  falls with target size; a 26B/262k-vocab pairing would put a different share on the same fixed cost.
