# CUDA Theta — the end-to-end validation that was owed (2026-09-01)

**The shipped `Theta = 0.5` made speculative decode SLOWER than not speculating at all on two of
four cells (0.91× and 0.92× of `off`). Wiring the measured value removes that. It buys back a
regression; it does not add a speedup.**

## Provenance

| | |
|---|---|
| box | `nobara-pc`, RTX 2070 SUPER, **driver 595.91.07**, Nobara 44 — the §B8 anchor stack |
| goinfer | `89911e2` |
| toolchain | go1.27.0 linux/amd64, `-tags "cuda goinfer_testhooks"` |
| models | qwen2.5-coder **0.5B** and **1.5B** instruct q4_k_m, int4, from `~/models` on local NVMe |
| harness | `cuda/theta_ab_test.go` (`TestThetaAB`), arms interleaved per repetition |
| run | 162.2 s, **zero lossless violations**, 4 of 8 cells surviving the pre-registered loop rule |

No peer is involved. Both arms are goinfer, so this is goinfer-vs-goinfer work and no
`bench_peer.py` number appears here.

## Theta re-measured first

The constant is only worth validating if it reproduces. `TestThetaProbe_CUDA`, same definition as
the CPU control and the Metal probe (`Theta = slope of T(n) / T(1)`):

| model | depth | T(1) | Theta |
|---|---|---|---|
| 0.5B | 128 | 4 752 µs | **0.153** |
| 0.5B | 512 | 4 493 µs | **0.178** |
| 1.5B | 128 | 5 562 µs | **0.228** |
| 1.5B | 512 | 5 834 µs | **0.243** |

Range 0.153–0.243 against 0.155–0.251 recorded. Stable. The shipped constant `thetaFor("cuda") =
0.251` sits just above the top of the measured range — deliberately conservative, because Theta
appears inside `floor(ln Theta / ln alpha)` so understating it drafts *deeper*, and a too-deep
draft on a low-acceptance stream is the failure the adaptive controller exists to prevent.

## The A/B

Median ms per cell; `off` is plain generation with no drafter.

| cell | off | fixed-8 | ada@0.5 (shipped) | **ada@wired (0.251)** | ada@measured | ada@0.30 |
|---|---|---|---|---|---|---|
| 0.5b code-continue-2 | 751 | 509 | 531 | **521** | 511 | 535 |
| 1.5b code-continue | 1390 | 1685 | 1525 | **1396** | 1398 | 1402 |
| 1.5b code-continue-2 | 1348 | 909 | 911 | **916** | 922 | 929 |
| 1.5b agent-loop-turn2 | 1592 | 1819 | 1722 | **1624** | 1623 | 1711 |

As a ratio to `off` (>1 is faster than not speculating):

| cell | fixed-8 | ada@0.5 | **ada@wired** |
|---|---|---|---|
| 0.5b code-continue-2 | 1.47× | 1.41× | **1.44×** |
| 1.5b code-continue | **0.82×** | **0.91×** | **1.00×** |
| 1.5b code-continue-2 | 1.48× | 1.48× | **1.47×** |
| 1.5b agent-loop-turn2 | **0.88×** | **0.92×** | **0.98×** |

## What it shows

**1. The do-nothing arm earned its place, again.** On two of four cells the SHIPPED configuration
was *slower than not speculating* — 0.91× and 0.92× of `off`. Without `off` in the arm set this
reads as "adaptive beats fixed-8, ship it." This repo has already found one speculation suite
where no configuration beat running no drafter at all; that is why the arm is mandatory here.

**2. The wired value fixes exactly that.** `ada@wired` vs `ada@0.5`: **+2.0%, +9.3%, −0.5%,
+6.0%**. Both losing cells return to parity with `off` (1.00×, 0.98×); the two cells that were
already winning are unchanged within noise.

**3. The conservative choice cost nothing.** `ada@wired` (0.251) lands within ~1% of
`ada@measured` (0.155 / 0.235) on every cell. Taking the top of the measured range to under-claim
the win rather than risk a regression turned out to be free. **That was not guaranteed** — it is a
result, not a vindication of the reasoning, and a different curve could have made it expensive.

**4. `fixed-8` is the worst arm on the hard cells** (0.82×, 0.88×), which is the adaptive
controller justifying its existence independently of Theta.

## Read the size honestly

**This buys back a regression; it does not create a speedup.** On the two bad cells the wired
value recovers to *parity with not speculating*, and on the two good cells it is a wash. The
CHANGELOG entry for the Theta fix says CUDA "drafts far too shallow" — true, and worth up to
+9.3% — but the mechanism of the gain is removing wasted verify work, not extracting new
throughput. Same shape as the Metal result (there: speculation was 1.07× slower than `off`, and
the wired value recovered ~6%).

Four cells is a small sample, and half the corpus was excluded by the loop rule before any timing
was read (distinct-trigram < 0.70 on `prose-doc` both models, `code-continue` and
`agent-loop-turn2` at 0.5B). The exclusion is pre-registered and applied blind, but it does mean
the surviving cells skew code-like.

## Still not claimed

- **WebGPU is unmeasured** and falls through to the 0.5 default.
- Two models, one card, one quant (int4), greedy only. Sampled decode is not covered.
- The constant is left at **0.251** rather than moved to the per-model measured values: the A/B
  shows no resolvable difference between them, so there is nothing to buy by splitting it per
  model, and one number per backend is the simpler thing to keep true.
