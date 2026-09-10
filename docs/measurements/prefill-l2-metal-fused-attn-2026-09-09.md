# Metal prefill L2 — fused (simdgroup_matrix) attention: built, correctness-tested, measured 4.23× on S/K=3900 (2026-09-09)

**`attention_prefill_fused` is real, correct, and measured — but this is NOT a gate pass and
does NOT ship as a default.** It is built, tested at both the isolated-kernel and full-pipeline
level, and measured end to end on the real S checkpoint. It has **not** been through the §3
fidelity/decision-set gate L1 ran (`docs/measurements/prefill-gate-l1-ref-b-2026-09-09.md`) — no
decision-set sweep, no D7 cell, no pooled §3.2 form. It ships behind
`GOINFER_METAL_FUSED_ATTENTION=1` (default OFF) until that gate runs, per the standing rule that
a default change needs its own gate cell.

## Provenance

| | |
|---|---|
| box | MacBook Pro, Apple M1 Pro, macOS 26.6.2 |
| goinfer | `d3eb08f` + this change (metal/prefill.go: `attention_prefill_fused`; metal/backend.go: `metalFusedAttentionEnabled`) |
| model | **S** = `~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`, int4, 28L nH=12 nKV=2 hd=128 — local SSD, not the SMB archive |
| harness (end to end) | `metal/prefill_ttft_test.go` (`TestPrefillTTFT`, `GOINFER_HEAVY_TESTS=1`), sequential vs batched `PrefillLast`, P ∈ {256, 1024, 3900} |
| harness (kernel) | `metal/attention_prefill_bench_test.go` (isolated single-layer dispatch, M=140 and M=3900, synthetic random f16) |
| harness (correctness) | `metal/attention_prefill_fused_test.go` (5 synthetic seam cases vs the exact kernel) + `metal/prefill_gemma_test.go` / `prefill_moe_parity_test.go` (real small-checkpoint parity, run with the flag on) |
| method | same session, baseline then fused arm ~2.5 min apart, box otherwise idle |
| logs | `~/goinfer-logs/l2-metal-ttft-baseline-*.log`, `~/goinfer-logs/l2-metal-ttft-fused-*.log` |

## 1. End-to-end TTFT (`TestPrefillTTFT`, batched arm, real S checkpoint)

| P | baseline batched | fused-attn batched | **ratio (fused vs baseline batched)** | vs sequential (today's shipped default) |
|---|---|---|---|---|
| 256 | 945.04 ms | 941.54 ms | 1.00× | 3.84× → 3.87× |
| 1024 | 5120.82 ms | 3972.26 ms | **1.29×** | 2.96× → 3.82× |
| **3900** | **37462.72 ms** | **17939.10 ms** | **2.09×** | **2.03× → 4.23×** |

P=256 is unaffected, as expected — attention is a small share of a short prefill and this
prompt length sits at the `metalFastPrefillFloor` boundary (overridden to 0 for this sweep, same
as `metal/prefill_gate_ref_test.go` already does, so P=256 could be measured at all). The win
grows with K, matching the O(K²) attention share `docs/task-prefill-gap.md` §4 predicted.

**This lands almost exactly on the doc's own pre-registered projection.** §4's "then, and now
sized" paragraph priced this kernel at "would take the S/K=3900 speedup from 2.02× to ~5×"
before any of it was built. Measured: baseline 2.03× (matches the 2.02× prior figure to within
run-to-run noise) → **4.23× with the fused kernel** — the projection's ballpark, from a real
measurement, not the arithmetic that produced the projection.

## 2. Isolated kernel throughput (`attention_prefill` vs `attention_prefill_fused`, one layer)

Synthetic random f16, nH=12 nKV=2 hd=128 (S model's own geometry), single dispatch, best-of-N:

| M (=K, causal) | exact | fused | **ratio** |
|---|---|---|---|
| 140 | 754.1 μs | 418.5 μs | 1.80× |
| 3900 | 647.17 ms | 118.76 ms | **5.45×** |

The kernel-level ratio (5.45×) is higher than the end-to-end batched-arm ratio (2.09×) because
attention is not 100% of the batched arm's cost — the int4→f16 MMA GEMM (`gemm_w4f16_store`)
is untouched by this change and still costs whatever it cost before. Rough cross-check: 28
layers × (647.17−118.76) ms ≈ 14.8 s of the 19.5 s the end-to-end run actually saved
(37462.72−17939.10 ms) — same order of magnitude, not an exact match (different measurement
paths, dispatch/launch overhead, thermal state); the end-to-end number is the one this doc is
written on, not the extrapolation.

## 3. Correctness

`TestAttentionPrefillFused` — 5 synthetic seam cases (GQA + ragged tail, no-GQA + hd=64 + a
mid-context startPos offset, GQA + sliding window + mid-context, hd=8 single-tile, an
exact-multiple-of-8 M) — **5/5 pass**, cosine 1.000000 in all five, max abs diff 1.2e-4–4.9e-4
(f16-rounding scale, not a divergence). NOT bit-identical by design (the online-softmax rescale
reorders the sum vs. the exact kernel's single final normalize — same P19 category as the CUDA
L2 twin), so this checks closeness, not equality.

Ran the real small-checkpoint parity tests (`TestPrefillParityGemma`, `TestPrefillParityMoE`)
with `GOINFER_METAL_FUSED_ATTENTION=1` through the *actual* `PrefillLast` pipeline (not a
standalone kernel harness) — confirmed via a throwaway debug print that the fused path genuinely
engaged (hd=16 and hd=8 respectively) before removing it:

| test | baseline cosine (exact kernel) | fused-attn cosine |
|---|---|---|
| TestPrefillParityGemma | 0.99979 | 0.99980 |
| TestPrefillParityMoE | 0.99993 | 0.99992 |

No fidelity degradation beyond noise at the last digit — both stay at the same cosine level the
exact batched kernel already sits at versus the sequential reference.

## 4. Scope of what this is NOT

- **Not a §3 fidelity gate.** No decision-set sweep, no D7 cell, no pooled §3.2 form comparing
  against the CPU f32 reference the way L1's gate did.
- **Not a default change.** `metalFusedAttentionEnabled()` defaults OFF; `attention_prefill`
  (exact) stays the shipped path. Opt in with `GOINFER_METAL_FUSED_ATTENTION=1`.
- **hd ≤ 128 only.** `attention_prefill_fused` requires `hd%8==0 && hd<=128` (`ATTN_MAXHD`,
  `metal/prefill.go`); `PrefillLast` falls back to the exact kernel outside that range. Every
  family measured or benched here (qwen2.5-coder-1.5b: hd=128; the Gemma/MoE parity fixtures:
  hd=16/hd=8) is inside it, but a family with hd=256 (some MoE/long-context archs) is not yet
  covered and would need either a wider `ATTN_MAXHD` (more threadgroup memory per simdgroup —
  budget math in the kernel's own comment) or an accepted fallback.
- **Single model measured end to end.** Only S; D7 (7B) unmeasured for this kernel specifically
  (the L1 gate's D7 cell hit a fit-guard on this 16 GB machine — the same constraint would apply
  here).

## Next step

A real §3 gate for this kernel (decision-set sweep, pooled form, D7 with `GOINFER_NO_FIT_GUARD=1`
if memory allows) before considering `metalFusedAttentionEnabled()`'s default. Not run this
session — ask before starting it, per the standing rule that a default change ships only after
its own gate cell.
