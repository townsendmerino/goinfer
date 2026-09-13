# PRE-REGISTERED — is f depth-invariant? (decides whether the gate can be one formula)

Written 2026-09-13 before profiling. Follows `splitkv-d7-fthreshold-2026-09-13.md`.

## Why it matters

The three-point result establishes f = t_roof/t_actual as the discriminator at ONE depth (3900).
The measured benefit, though, GROWS with depth on every geometry (D7 1.0496→1.0993, mistral
1.0240→1.0354, 1.5B 1.189→1.282→1.392). Two readings, and they imply different gates:

- **f depth-invariant** — f is a pure geometry property; the sign is fixed per model and only the
  MAGNITUDE grows with depth (attention's share of the step rises). The gate is then ONE formula
  (roofline from config) plus the existing depth threshold. Clean.
- **f climbs with depth** — every model drifts toward bandwidth-bound as context grows, so the sign
  itself is depth-dependent and the formula needs its own depth term. Less clean, and it would
  predict split-KV's advantage eventually reversing at very deep context.

## Test

ncu `attn_batched` at M=1, split-KV OFF, **depth 8000**, on D7 and mistral-7b — the two geometries
already measured at 3900. Grid asserted `(28,1,1)` / `(32,1,1)` before any number is read.

    f@3900:  D7 13.5%   mistral 26.9%
    roof@8000: D7 73.1 µs   mistral 146.3 µs

## Decision rule

Ratio r = f@8000 / f@3900, per model.

- **INVARIANT** — both models land in r ∈ [0.85, 1.15]. Gate can be one formula + depth threshold.
- **CLIMBS** — either model r > 1.15. The formula needs a depth term; report the size of it.
- **FALLS** — either model r < 0.85. Unexpected; would mean the advantage grows on both axes and
  the conservative depth thresholds are costing more than assumed.
- **SPLIT** — the two models disagree in direction. f is not a single property and the whole
  formula idea is weaker than it looks; park it.

Bar of ±15% chosen because predicted-vs-measured agreement so far has run 0.5–2.5 pp on values of
13–68%, i.e. a few percent relative; 15% is comfortably outside that.

## Prediction

Invariant. If bytes and time both scale with nKeys, f is constant. Concretely this predicts D7's
kernel at 8000 lands near 73.1/0.135 ≈ 540 µs, roughly 2x its 263.9 µs at 3900.
