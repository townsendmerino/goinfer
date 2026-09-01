# P19 — the fused (FlashAttention-style) schedule loses on CPU. Closing the portable form.

**Measured, at fixed precision, at production shapes: fused is 1.031× serial (a wash) and 0.690×
with the matmul's column parallelism on (a loss). The pre-registered bar was ≥1.30× clears /
<1.10× closes. It closes.**

The correctness half is fine — cosine **1.000000000**, max|diff| ~1e-8 — so this is a schedule that
computes the right thing and is not worth running.

## Provenance

| | |
|---|---|
| box | `nobara-pc`, Ryzen 7 3700X, Nobara 44 (CPU measurement; no GPU involved) |
| goinfer | `89e4d51` + this harness |
| harness | `decoder/p19_fused_attn_test.go` (`GOINFER_P19=1`) |
| shapes | kt=256, hd=128, nKeys=8192 — production's `attnRowTile(K=8192, nKeys=8192)` |
| precision | **f32 both arms.** Fixed by the item's own rule: "a win that only appears with f32 enabled is A3's win, not this item's" |
| load | 0.13 / 0.04 / 0.01 at start |

Gathers sit outside the timed region: both arms pay identical ones, so including them would only
dilute the effect. Neither arm masks; identical in both, so the ratio is unaffected.

## Pre-registered bar (written before the kernel existed)

	>= 1.30x  -> clears its own bar, prototype in production
	<  1.10x  -> close the item
	1.10-1.30 -> ambiguous, parks

Correctness stated in advance as a tolerance, not bit-identity, because the running-max rescale
re-associates by construction: **cosine ≥ 0.9999**.

## Result

| | materialized | best fused | ratio |
|---|---|---|---|
| **serial** (stable) | 52.9 ms | 51.3 ms (kb=512) | **1.031×** |
| **parallel** (as shipped) | 37.4 ms | 54.1 ms (kb=256) | **0.690×** |

Per block size, parallel arm:

| kb | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| ratio | 0.633× | 0.690× | 0.515× | 0.519× |
| cosine | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 |

## Why it loses, which the control establishes rather than guesses

**The serial control was run because a kernel A/B in this repo has already published a ratio that
was mostly core count** — G24's first pass read 17.6× against a documented ~3.7× because it raced
parallel f32 against serial acc64. The same trap was available here and in the opposite direction:
`MatmulBT` fans out over its **N output columns**, and the two arms present very different N. The
materialized QKᵀ has N=nKeys=**8192**; the fused one has N=kb, **128–1024**.

Serially the schedule is a **wash** — 1.031×. So on a CPU it is not removing meaningful traffic:
the `kt × nKeys` score block at 8 MiB is streamed well enough by the blocked matmul that avoiding
it buys nothing measurable.

In parallel the materialized arm gains **1.41×** (52.9 → 37.4 ms) while the fused arm gains
**nothing** (51.3 → 54.1 ms, i.e. slightly worse). Fusion forfeits the column parallelism by
construction, because it exists precisely to make the score block small.

**Instability, recorded rather than smoothed:** the parallel materialized arm read 45.3 ms then
37.4 ms across two runs — 21% apart — moving the headline ratio 0.852× → 0.690×. The serial arm was
stable (52.9 vs 53.2). So the serial 1.031× is the citable reading and the parallel figure is
directional. Both are below the close bar, which is why the verdict does not turn on which one is
quoted.

## What is closed, and what is NOT

**CLOSED: the portable / pure-Go / CPU form of the lever** — which is the form the item was filed
on. Its own words: *"FA-2/FA-3 themselves are hand-written CUDA leaning on Hopper
warp-specialization and TMA; none of that ports to pure Go. **The portable claim is the I/O
schedule, not the kernels.**"* The portable claim is the thing measured here, and it does not hold
on CPU.

**NOT tested: the CUDA form.** The 55%-of-prefill share that made this item interesting was
measured on CUDA (`cuda-prefill-attention-share-2026-09-01.md`), where the score matrix goes to
HBM and the arithmetic of the trade is entirely different. Nothing here refutes a fused CUDA
kernel; it just is not the cheap portable win the item hoped for, and building one is the expensive
path the item already identified as not porting from FA-2/FA-3.

**Also not tested: an implementation that does not sit on top of `MatmulBT`.** This prototype
composes the fused schedule out of the general blocked matmul, which is why it inherits that
matmul's parallelism model. A hand-written fused inner kernel — one loop owning QKᵀ, the running
softmax and the AV fold, parallelised over QUERY ROWS instead of key columns — would not forfeit
parallelism the same way. That is a real caveat on the strength of this negative, and it is also a
much larger piece of work than the item was scoped for. **The measured claim is: the schedule
expressed portably over the existing matmul loses. It is not: fusion cannot win on CPU.**

## The correction this run also produced

The CPU attention share that made this item look near-dead earlier in the day (17.4%) was a **MoE**
figure. That was corrected before this ran — dense models are 55% (CUDA, K=3900) and ~70% (Mac CPU
acc64, K=8192) — so this item was measured on its merits rather than dismissed on a
model-class error. The negative result stands on this measurement, not on that mistake.
