# P18 — expert-major MoE prefill is 4.36× end-to-end, and the reason is not what I predicted

**FUND, by more than twenty times the pre-registered bar. And the mechanism I proposed —
eliminating per-row allocation — is refuted: it is worth 0.99×.**

## Provenance

| | |
|---|---|
| box | MacBook, Apple M1 Pro, 8 cores, 16 GB, macOS 26.6.2 |
| model | Mellum2, **full 28 layers**, `~/models/mellum2-unq`, **int4**, MoE (64 experts, top-k 8, `moe_intermediate_size` 896) |
| K | 4096, real routing |
| harness | `decoder/moe_expert_batch_test.go` (`TestMoEExpertMajor_endToEnd`) |
| method | paired, interleaved, alternating lead, medians of 2 pairs; warm-up discarded |
| gate | non-vacuity asserted in BOTH directions — the batched path must run with the flag on and must NOT run with it off |

## Result

| pair | per-row (today) | scratch-only | expert-major |
|---|---|---|---|
| 1 | 1206.9 s | 1218.5 s (0.99×) | **276.6 s (4.36×)** |
| 2 | 1192.3 s | 1165.2 s (1.02×) | **265.0 s (4.50×)** |
| **median** | **1206.9 s** | 0.99× | **4.364×** |

Pre-registered rule: fund ≥15%, park <8%, 8–15% ambiguous. **+336.4% → FUND.**

## The hypothesis this refutes was mine

The kernel microbenchmark said batching the expert matmul is worth **2.13×**, and `swiGLUExpert` is
~39% of prefill. Amdahl on that predicts **1.26×**. Measuring **4.36×** — 3× more than the stated
mechanism allows — is not a result to bank; it says something else is doing the work.

I proposed per-row allocation: `moeMLP` is called with a `nil` scratch on this path and allocates
~5 slices per row per layer (114,688 calls at K=4096), and the K=8192 profile recorded **339,293
GCs and 20.9 GB allocated**, with `make([]float32, nE)` alone at 46.2 s.

**Measured directly, it is worth nothing: 0.99× and 1.02×.** The `scratch-only` arm reuses one
scratch across the row loop and changes no arithmetic whatsoever, so whatever it buys is pure
allocation cost — and it buys nothing, twice, paired. The cheap five-line alternative I floated
("just give the prefill path a reusable scratch") is dead.

## So what does explain 4.36×?

Stated as hypothesis, because it is not measured: the microbenchmark reused **one** expert's
weights across every row, cache-warm. Production touches 8 of 64 experts per row with consecutive
rows selecting different ones, so per-row processing re-streams expert weights continuously — on a
machine where this model's working set drove swap to 18 GB. Expert-major loads each expert once and
runs ~512 rows through it.

If that is right, **my own microbenchmark understated the win because it omitted the pressure** —
the "synthetic reproduces shape, not pressure" trap, in the direction that would have killed the
item rather than oversold it. 2.13× measured cache-warm; 4.36× in the real memory regime.

The honest status: **the end-to-end number is measured and reproducible; the decomposition of
*why* is not.** The two things ruled out are routing diversity (1.7% of `moeMLP`, and the parked
Lever 4 measured its ceiling at ≤5%) and allocation (0.99×).

## Correctness

Bit-identical, which is what makes this cheap to adopt: no golden changes, no documented
divergence, no flag for users to reason about.

- `TestMoEExpertMajor_bitIdentical` asserts `!=` on every logit through the real forward, at K=600
  (crossing the chunk boundary) **and at K=4096**, the depth the claim is made at.
- **Mutation-proven**: folding in reverse rank order reddens it at 1,381,871 of 1,382,400 logits,
  first divergence `1.0782002` vs `1.0782003` — precisely what a tolerance bar waves through, which
  is why the gate demands equality.
- **Non-vacuity asserted**: a chunk counter confirms the batched path actually ran (224 chunks at
  K=4096 = 8 × 28 layers, matching the geometry exactly). `moeMLPBatch` refuses for several
  legitimate reasons, and a refusal would make both arms the identical per-row path — the test
  would pass while proving nothing.

Bit-identity is achievable because the matmuls are M-invariant by documented contract, and because
the per-row fold is kept in **routing-rank order**: every (row, rank) expert output is computed
first, then folded in the order `moeMLP` uses. Float addition is not associative, so an
expert-major visit order would otherwise change the result — which is exactly what the mutation
demonstrates.

## Not claimed

- **One model, one depth, one machine.** MoE only by construction; dense models have no expert FFN.
- The mechanism behind the 4.36× (above).
- `moeMLPBatch` declines for a live pager, a shared expert, and the routing test seams. Those paths
  are unmeasured and unchanged.
