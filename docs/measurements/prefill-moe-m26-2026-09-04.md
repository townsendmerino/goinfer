# M26 is on the batched prefill path, bit-identical — and it is worth 8%, not the 2.3× I projected

**The batching was never M26's bottleneck.** All three guards that kept Gemma-4-26B-A4B off CUDA's
batched prefill are removed, none needed a new kernel, the result is bit-identical on the real
model (0 of 262,144 logits differ, C′ expert cache live) — and it buys **1.085×**. The
active-parameter arithmetic in queue-performance P20 said the attention half was ~45% of the
per-token weight traffic and the dense FFN branch another ~25%, predicting ~2.3×. Measured, the
whole batchable half is worth 8%.

**That is the finding.** A number this far under its projection is not a disappointing result, it
is a relocated bottleneck: ~92% of M26's prefill is somewhere batching does not reach.

## Provenance

| | |
|---|---|
| box | `nobara-pc`, AMD64, **NVIDIA GeForce RTX 2070 SUPER (8 GB)**, driver **595.91.07**, Linux 7.2.0-202.nobara.fc44 |
| model | **Gemma-4-26B-A4B**, goinfer kind-4 `.giw` bundle, **local disk** `~/models/gemma4-26b-int4.giw` — the peer matrix's M26 |
| config | `MoECacheExperts: true, ResidentContext: 8192` — what `scripts/bench_peer.py` launches as `-moe-cache-experts -ctx 8192`. C′ self-caps to **12 slots/layer** (64 would need 6.5 GB against 1.8 GB free) |
| geometry | 30 layers, all MoE; 5 non-uniform vs layer 0 and the same 5 K=V; hidden 2816, inter 2112, 128 experts, moe_inter 704 |
| free VRAM after load | 0.44 GB |
| harness | `cuda/prefill_longprompt_test.go`, `GOINFER_HEAVY_TESTS=1` |
| versions | go1.27.0, `aikit/gpu v0.32.0`, goinfer at `4ee59e15` + this change |
| method | **both arms in ONE process**, cache `Reset()` between them, batched arm first then sequential; single timed run per cell |
| logs | `docs/measurements/prefill-chunking-2026-09-04/` — `m26-paired.log`, `m26-parity.log`, `m26-attribution.log` |

## Result

| M | batched | sequential | ratio |
|---|---|---|---|
| 512 | 43.681 ms/token | 47.382 ms/token | **1.085×** |
| 2048 | 43.416 ms/token | 47.009 ms/token | **1.083×** |

Reproducible across a 4× change in M, and **both arms are flat with depth** — 43.681→43.416 and
47.382→47.009. Whatever dominates this prefill does not grow with the attended span.

## Bit-identity

`TestPrefillMoE_real26B`: the batched pass against the sequential per-token path, same resident,
same positions, M=48 — **0 of 262,144 logits differ, max |diff| 0**, with `cacheExperts=true` so the
C′ host→VRAM expert DMA runs inside the per-row loop.

A tolerance-based check would have been worthless here and the gate says so: routing is a discrete
argmax over router logits, so a small numerical difference does not perturb a row slightly, it runs
**a different expert** and the row is unrelated. Equality is the only assertion that means anything
on a MoE model.

Also gated on fixtures: `gemma4-moe-scaled` (real 26B FFN shapes, 4/4 layers MoE) 0/4096, and
`gemma4-dense-scaled` (per-layer geometry + K=V, no MoE) 0/256. **Mutation-proven** — binding row 0
for every row reddens 4095/4096 (max |diff| 13.47); dropping the K=V `v_norm` reddens 255/256;
re-hoisting layer 0's geometry reddens 255/256.

## Where the other 92% goes — measured, not inferred

`GOINFER_MOE_CACHE_PROF=1`, same model, same M=512, both arms in one process
(`m26-attribution.log`):

| arm | wall | C′ DMA | share of arm | `loadRoutedExperts` calls | syncs |
|---|---|---|---|---|---|
| batched | 22.217 s | **12.839 s** | **59.5%** | 15,360 | 15,360 |
| sequential | 24.231 s | **12.819 s** | 54.5% | 15,360 | 15,360 |

**The DMA is the same in both arms to within 0.16%** — 12.839 s against 12.819 s, over an identical
15,360 calls (512 rows × 30 layers). Batching does not touch it and was never going to: the row loop
performs exactly the transfers the per-token loop did.

Subtract it and the batchable remainder is 11.412 s sequential → **9.378 s batched, −17.8%**. That
is the real size of what this change does, and it is diluted to 8% by a DMA floor it cannot move.

**Amdahl on the measured share, not on a parameter count: with 59.5% untouchable, everything else
going to zero caps the win at 1.68×.** Anything further has to attack the transfer itself.

Two properties of that transfer are worth carrying forward. It costs the same per token however
many tokens are in flight — so it is per-row work, and the one restructuring that changes per-row
expert traffic is **expert-major** (group a chunk's rows by expert, so a layer fetches each distinct
expert once per chunk instead of once per row). And it synchronizes once per (row, layer): 15,360
synchronizes for a 512-token prompt.

**No number is projected for expert-major here.** The bytes moved were not captured in this run,
only the call count and the time, so the fetch-count reduction cannot be turned into a time
estimate without assuming the driver is fetch count rather than per-call overhead. That assumption
is exactly the shape of the one this document exists to record as refuted. Measure it when it is
built.

## What the 8% refutes

The projection came from counting active parameters per token on M26's config: attention
projections ~45%, dense FFN branch ~25%, routed experts ~30%. If batching removes the first two the
way it removed D7's weight term (12.480 → 2.777 ms/token, ~4.5×), per-token cost should have fallen
to ~20 ms. It fell to 43.4.

**Parameter counts price weight READS. M26 does not pay for weight reads — it pays for a PCIe
transfer.** The expert stack exceeds the card, so C′ streams routed experts host→VRAM per token
through 12 slots per layer, and the per-row FFN loop still performs every one of those transfers.
Batching the attention half removes VRAM reads that were never the constraint.

This is the same shape as P18's refuted mechanism and P19's Amdahl estimate, for the third time
this fortnight: **an arithmetic share of a quantity that is not the bottleneck predicts nothing.**
The tell was available beforehand and was not read — the sequential prefill was already measured
FLAT with depth (46.701 → 46.434 ms/token), and a cost that ignores the attended span is not a
compute cost.

## Decision

Pre-registered rule (queue-performance P20): fund ≥15%, park <8%, **8–15% ambiguous → parked
pending a second mechanism**. Measured 8.3–8.5% ⇒ **ambiguous band**.

**The rule was not evaluated on its own terms and this is recorded rather than glossed.** It
specified "end-to-end on M26 at depth 8000, paired and interleaved"; what ran was in-process at
M=512/2048, paired within one process but not interleaved and not at depth 8000. The proxy is close
— both arms flat with depth means 8000 should read the same — but a rule is not satisfied by a
measurement that resembles it.

**And the rule's inference was backwards, which is worth more than the rule.** It read "if the
attention-half slice is small, do not fund expert-major". But a small attention-half win means the
experts dominate, which is an argument *for* expert-major, not against. The pre-registration
encoded an assumption that the two halves compete for the same bottleneck; they do not.
CLAUDE.md's own corollary applies — after a rule fails, the reliable fix is a second independent
pre-registration, not a better-worded first one.

**Ship it anyway, and the reason is not the 8%.** It is bit-identical, it costs nothing to keep,
and it is the prerequisite for expert-major: the row loop it introduces is exactly the loop an
expert-major gather replaces.

**The second mechanism the ambiguous band asked for is now named and measured**: the C′ host→VRAM
expert DMA, 59.5% of the batched arm, unchanged by batching. That is what the next increment has to
move, and it is a different lever from anything in this change.

## What this does not touch

**M35 (Qwen3.6-35B-A3B).** 30 of its 40 layers are `linear_attention` with in-place recurrent
state; batched prefill has no form of them at all. That is a chunked delta-rule scan and real new
math, not a restructuring — the only part of P20 that is.
