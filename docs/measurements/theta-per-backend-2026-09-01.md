# Theta, measured per backend — and the domain that excluded Metal (2026-09-01)

**`AdaptiveDepth.Theta` was enforced in `[0,1)`. Metal measures 1.006–1.048. Every value Metal
actually has was rejected and silently replaced by 0.5 — the most over-drafting setting on the
dial — at the one moment the measurement said "do not draft at all."**

## Provenance

| | |
|---|---|
| box | MacBook, Apple M1 Pro, 8 cores, 16 GB, macOS 26.6.2 |
| toolchain | go1.27.0 darwin/arm64 |
| models | qwen2.5-coder **0.5B** and **1.5B** instruct q4_k_m, from `~/models` on the internal SSD |
| definition | `T(n)` = wall time of one verify pass over `n` tokens at fixed depth; `Theta = (least-squares slope of T(n)) / T(1)` |
| probes | `decoder/theta_probe_test.go` (CPU control), `metal/theta_probe_test.go`, `cuda/theta_probe_test.go` — one definition, three backends, directly comparable |

## Measured

**CPU control, re-measured** (the instrument check — if it cannot reproduce the one value already
believed, its answer elsewhere is not trustworthy either):

| depth | T(1) | slope | Theta |
|---|---|---|---|
| 128 | 12 285 µs | 6 215 µs/node | **0.506** |
| 512 | 14 513 µs | 7 724 µs/node | **0.532** |

Recorded previously as 0.456. So the shipped default of 0.5 was right for the path it was named
after — the batched-CPU `ForwardN` — and wrong everywhere else, which is the whole finding.

**Metal**, four cells:

| model | depth | T(1) | Theta | T(16)/T(1) |
|---|---|---|---|---|
| 0.5B | 128 | 5 933 µs | **1.020** | 16.12 |
| 0.5B | 512 | 6 728 µs | **1.048** | 16.81 |
| 1.5B | 128 | 13 401 µs | **1.006** | 16.11 |
| 1.5B | 512 | 14 225 µs | **1.019** | 16.07 |

`T(n)/T(1)` is linear to n=16 in every cell. That is not "roughly linear" — it is what a loop
predicts exactly, and [`metal/backend.go:280`](../../metal/backend.go) is exactly that: a plain
`for` over single-token `Forward` calls. `ResidentForward`'s own interface doc says `ForwardN` runs
"K tokens ... in ONE command buffer (one Submit/Poll)". Metal satisfies the bit-identity half of
that contract trivially and the batching half not at all.

**CUDA**: 0.155–0.251, previously measured, not re-run here (needs the Linux box).

## The defect

```go
if a.Theta <= 0 || a.Theta >= 1 { a.Theta = 0.5 }   // spec_adaptive.go, before
```

Three separate problems, and they compound:

1. **The domain excluded the measurement.** Not a bad default — an *unreachable* correct value. No
   caller could have fixed this by configuration.
2. **The substitute was the worst available choice.** `Depth()` uses
   `floor(ln Theta / ln alpha)`, so a *smaller* Theta drafts *deeper*. Replacing 1.02 with 0.5 does
   not fail safe toward "draft less"; it fails toward "draft as if nodes were half price."
3. **Nothing set it per backend anyway.** Both production entry points
   (`session.go`, `spec_adaptive.go`) construct `&AdaptiveDepth{}` with Theta unset, so *every*
   backend ran the CPU constant that the file's own doc comment had been asking someone to measure
   since it was written. The errors run in **opposite** directions: CUDA at ~1/3 of the constant
   drafts far too shallow; Metal drafts when it should not draft at all.

A fourth, found while fixing: `Depth()` forces `D>=1` every `ProbeEvery` idle rounds to refresh a
stale acceptance estimate. Under `Theta >= 1` the decision does not depend on alpha at all, so that
probe **cannot change any future answer** — it is a token drafted and discarded every 16 rounds,
forever. It needed its own assertion, because the probe branch sits *before* the `alpha <= Theta`
test: a fix that corrected only the comparison leaves it drafting, and the domain test still passes.

## Does wiring it actually make Metal faster? — measured

`TestMetalThetaAB`, one 256-token prompt, 48 generated tokens, medians of 3, arms interleaved,
**including the do-nothing arm** (a "beats every configuration" claim is worthless if *off* wins —
and here off is expected to win against the old setting):

| arm | median | vs off |
|---|---|---|
| off — plain `Generate`, no speculation | **1869.4 ms** (repeat 1900.1) | — |
| spec, `Theta=0.5` (shipped until now) | 1999.7 ms | **1.07× slower** |
| spec, Theta wired (1.02) | 1881.7 ms | **1.01×** |

So Metal was paying **~7%** to speculate in a way its own cost model says can never pay, and the
wired value recovers it by declining — landing within the off arm's own run-to-run spread (1869 vs
1900, 1.6%).

**Read the size honestly: ~6% on this workload, not a headline.** The magnitude depends on the
corpus's acceptance rate — a copy-heavy corpus drafts successfully more often and wastes less, a
novel one wastes more. What is not workload-dependent is the direction: on a backend where a verify
node costs a full step, drafting is a strict loss, and the controller could not be told so.

## What this does not do

- **It does not make Metal's `ForwardN` a batch.** Theta ≈ 1.0 *because* it is a loop; this change
  makes the controller tell the truth about that, it does not change the truth. Batching `ForwardN`
  is the follow-on, and it is the item with the real upside — it would move Theta itself, at which
  point this table is re-measured and Metal's constant is replaced.
- **The CUDA end-to-end effect is NOT validated.** The 0.155–0.251 constant is measured, but its
  effect on acceptance and throughput has not been run on the box. The conservative end (0.251) is
  wired deliberately: Theta sits inside `floor(ln Theta / ln alpha)`, so understating it drafts
  deeper, and a too-deep draft on a low-acceptance stream is the exact failure the adaptive
  controller exists to prevent. The win is under-claimed rather than the regression risked. **That
  box run is owed.**
- **WebGPU is unmeasured** and falls through to the 0.5 default, explicitly rather than by accident.
