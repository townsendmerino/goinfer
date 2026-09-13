# PRE-REGISTERED — the A/A floor for mistral-7b, and does the 2026-09-12 effect reproduce?

Written 2026-09-12 **before any cell ran**, after the run it exists to check.

## Why

`splitkv-8000-reanchor-2026-09-12.md` measured mistral-7b force-ratios of **1.0240** (3900) and
**1.0354** (8000) — split-KV winning for a geometry the shipped gate classes `splitkvNever`. That
run's own §Limitations names two defects: **no A/A floor was characterised**, so a 2.4% effect
cannot be told from drift, and the effect was measured **once**. §B6.3 characterises a floor per
cell in advance; that pre-registration omitted it. This run supplies both.

## Arms

All mistral-7b, depths 3900 and 8000, same binary `serve-cuda-1f224682`, same harness path (fresh
`serve` per arm, arms adjacent in time, order alternating).

| mode | on-arm env | off-arm env | what it measures |
|---|---|---|---|
| `aa-off` | `SPLITKV_ATTN=0` | `SPLITKV_ATTN=0` | noise floor, split path OFF both sides |
| `aa-on` | `SPLITKV_MIN_KEYS=0` | `SPLITKV_MIN_KEYS=0` | noise floor, split path ON both sides |
| `force` | `SPLITKV_MIN_KEYS=0` | `SPLITKV_ATTN=0` | the effect, repeated — reproducibility |

Both A/A polarities are run because the floors need not be equal: the ON arm exercises a different
kernel with different launch geometry, and the older in-process harness recorded ON's spread as
roughly 10x OFF's. Using an OFF-only floor to clear an ON-vs-OFF effect would understate it.

## Decision rule

Let **floor** = max over all four A/A cells of |ratio − 1|. This is deliberately the MAX, not the
mean: the floor is what any single cell could have produced by chance, and there are only four.

- **REAL** — both `force` repeats exceed 1 + floor, AND both agree in direction with 2026-09-12
  (within its own spread). The mistral win stands, `splitkvNever` is mis-keyed for GQA, and the
  finding in the previous record is confirmed rather than suggested.
- **NOISE** — either `force` repeat falls inside 1 ± floor. The 2.4–3.5% was drift; phi3-mini's
  `never` verdict is not contradicted by mistral, and the previous record's central finding is
  **withdrawn**. This is the outcome that costs the most to admit, which is why it is written first.
- **AMBIGUOUS → PARKED** — the repeats straddle 1 + floor, or disagree with 2026-09-12 by more than
  the A/A floor. Report and stop; do not run a third pass hunting for a tiebreak.

Pre-registered second thing that can disagree: **reproducibility is separate from clearing the
floor.** A repeat that clears the floor but lands far from 1.0240/1.0354 means the cell is not
stable, and a stable-but-small effect and an unstable-but-large one call for different responses.
Both are reported; neither is allowed to stand in for the other.

## What this cannot do

It cannot generalise from one geometry. Even a clean REAL leaves "GQA vs MHA" confounded with
"mistral-7b specifically" — a third nH >= 24 GQA model is still needed, and is not run here.
