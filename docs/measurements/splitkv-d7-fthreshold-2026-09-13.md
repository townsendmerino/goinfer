# D7 confirms the f axis: split-KV wins +9.9% on the geometry the gate says "never"

Ran 2026-09-13, nobara-pc. Pre-registration: `splitkv-d7-fthreshold-PREREGISTERED.md`, written
before any cell, with **f (DRAM-roof fraction) named as the sole discriminator** after the
max-of-three operationalisation was PARKED on 2026-09-12. Binary `serve-cuda-1f224682`, driver
`595.91.07`, preflight green (loadavg 0.04, 36 °C).

## Primary test — the three-point ordering: CONFIRMED

    f  =  t_roof / t_actual        t_roof = 2 · nKeys · nKV · hd · 4 B / BW      BW = 448 GB/s

| model | nH | floats/key | **f (measured)** | force ratio @3900 | ratio @8000 |
|---|--:|--:|--:|--:|--:|
| D7 Qwen2.5-7B | 28 | **512** | **13.5 %** | **1.0496** | **1.0993** |
| mistral-7b | 32 | 1024 | 26.9 % | 1.0240 | 1.0354 |
| phi3-mini | 32 | 3072 | 67.8 % | 0.7460 | — (4k model) |

**Monotone in both directions across a 6× span of bytes/key.** D7 exceeds mistral by 2.55 pp
against an A/A floor of 0.268% — **9.5× the floor** — and the ordering was the pre-registered
primary because an ordering is far harder to satisfy by accident than a threshold.

## Secondary test — the f < 0.40 bar, and a prediction made in advance

The pre-registration predicted D7's f would land "near **13%**" from geometry alone, before the
kernel was profiled. Measured:

    predicted roof 35.7 µs / measured duration 263.9 µs  →  f = 13.5%
    ncu DRAM throughput                                  =  14.04%      (0.5 pp apart)

The model now predicts the measurement on all three geometries — 13.5 vs 14.0, 25.9 vs 26.9,
65.3 vs 67.8 — and the 0.40 bar classifies all three correctly. That is the check my 2026-09-12
max-of-three failed, and it failed because L1TEX and L2 measure cache re-read pressure (GQA reads
each KV head nH/nKV times through L1), a different bottleneck the max operator folded in. D7's
L1/TEX is 50.92% against a DRAM of 14.04% — the same shape as mistral, and the reason max-of-three
could not separate them.

## What this costs today

D7 is **nH=28 ≥ splitkvMaxHeads=24**, so the shipped gate classes it `splitkvNever` and runs the
single-block kernel. At depth 8000 that forfeits **+9.94%** (35.84 → 39.40 tok/s), on the exact
cell queue-performance P24 was opened over.

Cross-validation worth recording: the OFF arm at 8000 measured **35.84 tok/s** against P24's
independently-taken **35.7** — 0.4% apart, different session, different harness path. The two agree.

Honest scope: the deficit P24 names is goinfer retaining 0.49 of its shallow rate to 8000 where
llama.cpp retains 0.73. Split-KV moves that to **0.54**. Real, and about a fifth of the gap — not
the whole answer to P24.

## Occupancy, a third time

12.61% achieved, 28 blocks on 40 SMs — indistinguishable from mistral's 12.68% and phi3's 11.34%.
Three geometries, three near-identical occupancies, three different signs. `cuda/resident.go:222`'s
"already fills the device" remains refuted, now on three points instead of two.

## Limitations

1. **Two of three geometries are GQA.** phi3-mini is the only MHA point, so "f" and "MHA vs GQA"
   are still partly confounded. Llama-2-7B (MHA, 4096 floats/key, predicted a LARGER loss than
   phi3's 0.746) would anchor the far end and is not on this box.
2. **f is measured, not config-computable** — it needs t_actual. It is the right scientific
   discriminator; the shipped gate needs the config-side proxy (`kvFloatsPerKey`), and the two
   should not be conflated in code.
3. Depths 3900 and 8000 only. D7's shallow behaviour is unmeasured, which is exactly why the
   re-key below must keep a depth threshold.

## Implication for the re-key, stated as a correction to the obvious version

A binary "allow ≤1024 / never ≥3072" would drop the depth threshold, and that is P6a's hard-won
protection — a constant that fired 3–12× too early cost 18–25%, and 1.5B still loses at 128 (0.933).
D7's shallow depths are unmeasured here. The narrower change: **`kvFloatsPerKey` selects the class,
and the allowed class inherits `splitkvConservative` (3072) as its depth threshold.** All three
measured points then classify correctly and no unmeasured geometry changes behaviour.

Trap for whoever writes it: `splitkvConservative = 3072` is a **depth in keys**; phi3-mini's
`kvFloatsPerKey = 3072` is a **byte count**. Same number, different units, adjacent in the same file.
