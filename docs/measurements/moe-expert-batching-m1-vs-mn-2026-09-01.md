# MoE expert matmul, M=1 vs M=N — Lever 4's ceiling did not bound this (2026-09-01)

**Batching the expert matmul over rows is worth 1.55× at N=8 rising to 2.13× at N=256, measured
with weight locality already perfect. That is the axis the parked "not a compute lever" verdict
holds fixed, so it does not bound this and Lever 4 should be reopened as a scoped campaign.**

## Provenance

| | |
|---|---|
| box | MacBook, Apple M1 Pro, 8 cores, 16 GB, macOS 26.6.2 |
| shapes | Mellum2 real expert geometry — hidden **2304**, **moe_intermediate 896**, group 32 |
| weights | real **int4 / W4A8** `linalg.QuantizeInt4`, through the same `matmul(be, *WeightMat, …)` entry point production uses |
| harness | `decoder/moe_expert_batch_test.go` (`GOINFER_MOE_BATCH_PROBE=1`) |
| method | identical total rows in both arms, best of 3, arms interleaved, M=1 re-measured after M=N |

## Why this was asked again

The 2026-09-01 full-model profile (`mellum2-fullmodel-profile-RESULT.md`) moved the target:

| bucket | before A3 (acc64) | today's default |
|---|---|---|
| attention | 70.4% | 17.4% |
| **moeMLP** | 16.3% | **42.1%** |

and inside `moeMLP`, **`swiGLUExpert` is 93.1%** — so the expert weight matmuls are ~39% of
prefill and the largest single bucket. `routeExperts` is 1.7% of `moeMLP`; routing is not the cost.

At K=8192 the batched-prefill loop calls `moeMLP` **once per row**
([`decoder/forwardn.go:545`](../../decoder/forwardn.go)) and `swiGLUExpert` issues its three
matmuls at **M=1** ([`decoder/mlp.go:292`](../../decoder/mlp.go)), so an expert's weights are
re-read for every token that routes to it.

## What the parked verdict actually measured

`docs/task-moe-streaming.md` Lever 4 is parked on *"Expert-major MoE prefill batching is NOT a
compute lever"* (`mellum2-moe-prefill-split-RESULT.md`, 2026-08-28), whose ceiling was defined as:

> `uniform` (one repeated id) is both the control and the **ceiling**: a chunk whose rows all
> select the same experts is exactly what expert-major batching manufactures, so the lever cannot
> beat it.

**Both arms of that experiment call `moeMLP` per row at M=1.** What `uniform` varies is *which*
weights get touched, so it captures the **bandwidth/locality** half of batching — an expert's
weights stay cache-resident across rows — and not the **M=1 → M=N** half. Those are different
axes: a GEMV is latency- and ILP-bound in a way a GEMM over hundreds of rows is not, and they
coincide only if the workload is purely bandwidth-bound.

So that result soundly answers *"does routing diversity cost much?"* — no, ≤5%, and today's
profile independently agrees. It does not answer *"would batching rows into GEMMs help?"*

## The measurement

| rows (N) | M=1 per row | M=N per row | speedup |
|---|---|---|---|
| 8 | 182.2 µs | 117.5 µs | **1.55×** |
| 32 | 203.4 µs | 99.8 µs | **2.04×** |
| 64 | 206.0 µs | 99.1 µs | **2.08×** |
| 128 | 207.2 µs | 98.1 µs | **2.11×** |
| 256 | 203.5 µs | 95.4 µs | **2.13×** |

> **CORRECTED 2026-09-01, same day.** The first run of this table used
> `inter = 7168`, which is Mellum2's **dense** `intermediate_size`. The experts are
> `moe_intermediate_size = **896**`, and `swiGLUExpert` is called with `moe.IntermediateDim` — so
> the original figures (1.32× → 1.67×) measured expert matmuls **8× wider than the ones production
> issues**. The correction moved the result the way the mechanism predicts: narrower matmuls
> amortise per-call overhead worse at M=1, so batching helps **more**, not less. The error
> understated the win rather than inventing one, which is the direction that would have quietly
> killed the item.

**The M=1 arm here reuses one expert's weights across every row, so it IS the parked experiment's
`uniform` condition** — perfect locality, nothing left for batching to win on the bandwidth axis.
The 2.13× is therefore exactly the component that experiment's design could not observe. M=1 cost
per row is flat at ~182–207 µs across the whole ladder (it has no batch to amortise over); M=N
falls monotonically and **flattens near 2.1×** by N=64–256.

Real N is larger than the table's right edge: at K=8192 with top-k routing over 64 experts, an
expert sees on the order of 10³ rows per layer.

## What this does NOT establish — read before quoting a speedup

**No end-to-end number is claimed here, deliberately.** Multiplying 39% of prefill by 2.13× to
get an end-to-end figure is exactly the projection this repo retracted twice today: a ratio measured
under one set of controls applied to a system running under another. What is missing from this
microbenchmark is real:

- **The gather/scatter cost.** Rows routing to one expert are scattered through the sequence.
  Batching them requires permuting rows in and out, which this measures not at all — and it is the
  single most likely reason the real win lands below 2.13×.
- **Real routing distribution.** Rows per expert are uneven; small groups sit at the left of this
  table where the win is 1.55×, not the right.
- **Memory pressure at scale.** The full-model profile ran with swap growing 5.8 → 18.3 GB. A
  cache-warm microbenchmark on three weight matrices does not reproduce that, and a benchmark that
  omits the real pressure can promote a mechanism as easily as it can exonerate one.

So the finding is: **the door the parked verdict closed was a different door.** The next step is a
scoped campaign that measures the permutation cost against the batching win on the real forward,
not a re-derivation of the microbenchmark.
