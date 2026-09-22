# R11/P20: CUDA expert-major MoE prefill, built and shipped opt-in — real, small win on Mellum2; the actual target (M26) needs a further, un-built extension

Follows `p20-expert-locality-2026-09-21.md` (the measurement that reversed a too-hasty "infeasible" read and earned this build). Implementation: `cuda/moe_expert_major.go` (generic MoE path only), wired into `cuda/prefill.go`'s per-row MoE FFN loop behind `GOINFER_CUDA_MOE_EXPERT_MAJOR` (default off). No new `.cu` kernel: `gemv_w4a8_moe`/`gemv_w4a8_moe_wacc` (the audited, frozen `moe.ptx`) and `residual_batched` (`prefill_batched.ptx`) are reused unchanged — the whole win is a REORDERING of which (row, routing-rank) pair each existing kernel call processes, plus a bit-identity-preserving rank-ordered fold, mirroring the CPU's own P18 precedent (`decoder/mlp.go`'s `moeMLPBatch`) exactly: route every row first, bucket (row, rank) pairs by expert, admit/DMA each DISTINCT expert once and run every row assigned to it, write each (row, rank) output into a rank-indexed scratch buffer (never accumulated-into-live, so ordering is decided later), then fold every row in the ORIGINAL rank order at the end via `topK` sequential batched-add launches.

## What is NOT covered (named up front, not a footnote)

`Ly.g4moe` layers (Gemma-4's parallel dense‖MoE FFN, `gemma4MoeMLPPre/Post`) are explicitly declined, exactly as CPU's `moeMLPBatch` declines a shared expert. **This means M26 — the model the locality measurement was run against, and the one the ~2.3x projection was for — does NOT benefit from this build.** Gemma-4's combine (two branches, two post-norms, a join, a per-layer scalar) needs a structurally similar but distinct extension (its own per-row `g4x1`/routing storage, its own rank-ordered fold into a `g4x2`-equivalent scratch before the existing join/norm/scale chain) that was not attempted in this pass. Also declined: a shared expert (`Ly.hasShared`; GLM/DeepSeek shape), any per-expert bias table (gpt-oss), and any layer without C′ caching on (the mechanism this exists for).

## Correctness: bit-identical, mutation-checked, on two real fixtures

`TestMoEExpertMajorCUDA_bitIdentical` compares full prefill logits (not just argmax) on/off, on `testdata/qwen3moe-tiny` (topK=2) and a new `testdata/qwen3moe-tiny-k3` (topK=3, `scripts/pin_qwen3moe_tiny_k3.py` — a k=3 variant of the existing pin script). Both pass: **0/512 logits differ**, on both fixtures, with a non-vacuity check (`cudaMoeExpertMajorRuns` must actually increment when the flag is on, and must not when it's off).

**The k=3 fixture is not redundant with k=2 — it is the one that can fail.** Float addition of exactly two operands is commutative in IEEE754 (`a+b == b+a` bit for bit), so reversing the fold order is INVISIBLE at topK=2 by construction, not because the implementation is correct at that shape specifically. Checked directly: reversing the rank-fold loop (`for j := r.topK-1; j >= 0; j--`) passes cleanly at k=2 and fails **482/512 logits** at k=3, confirming the order-preservation mechanism is real and load-bearing, not merely untested. A second mutation (applying the wrong rank's routing weight to a given expert's output — a value-level defect, not an ordering one) fails **512/512** logits at k=2, confirming the (row, rank) bookkeeping itself is also exercised.

## Real-hardware measurement: Mellum2 (the only generic-MoE, non-shared-expert, non-g4moe checkpoint on this box that reaches the batched prefill path)

`mellum2.int4.giw`, 28 MoE layers, hidden=2304, moeInter=7168, `-moe-cache-experts -ctx 8192`, RTX 2070 SUPER, driver 595.91.07. `TestPrefillLongPrompt`, M=512, 2 rounds alternating arms, paired.

**Locality first, because it explains the result rather than being contradicted by it**: this box gives Mellum2 **51 slots/layer** (5.2 GB free after load — a much looser budget than M26's 10), against a median of 57 distinct experts/layer at M=512. The existing per-row LRU already gets **98.1% hit rate** (112,459 hits / 2,229 misses); a perfect once-per-chunk fetch would need only 1,574 (a 1.42x reduction in the already-small 1.9% miss share) — **there is very little headroom left for this lever to capture on this specific model/box combination**, unlike M26's 61.3%/22.8x situation.

| | off | on | ratio |
|---|---:|---:|---:|
| round 1 | 12.454 ms/token | 12.042 ms/token | 1.034x |
| round 2 | 12.466 ms/token | 11.953 ms/token | 1.043x |

**A real, small, reproducible win (~3.5-4.3%), exactly the size the locality data predicts — not the large win this feature was built for.** This is the honest result for the ONLY real checkpoint on this box that exercises the built code path; it is not evidence against the mechanism (the locality numbers explain why the ceiling here is low), and it is not a measurement of what M26 would show (M26 needs the un-built gemma4 extension to be measured at all).

## State

`GOINFER_CUDA_MOE_EXPERT_MAJOR` ships opt-in, default off. It is real, tested, gated infrastructure with a small proven win on the one available real target and a much larger PROJECTED (not measured) win on the model that actually motivated it. Shipping default-on is not proposed here — the measured win (3.5-4.3%) is real but modest, and the model with the large projected opportunity was not tested.

## Not established

The gemma4-specific extension (the actual path to M26's projected ~2.3x) is not built. No model with BOTH a tight VRAM budget (like M26's 10 slots) AND the generic (non-g4moe) MoE shape exists on this box to test the mechanism at its intended operating point — Mellum2's looser budget is a real, honest limitation of what could be measured this pass, not a design flaw. Byte-level DMA counts (vs the call counts used throughout) were not captured.
