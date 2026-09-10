# Metal prefill L2 — fused (simdgroup_matrix) attention: SHIPS and DEFAULT ON since 2026-09-10, measured 4.23× on S/K=3900

**Update 2026-09-10: the §3 gate ran and SHIPS.** §5 below is the gate — S's decision set
(K∈{256,512,1024}, prompt set B, pooled §3.2 form, same reference files and harness L1's own
gate used) passes all three criteria, and fused *beats* exact on every one (fewer hard flips,
slightly higher agreement, lower mean KL). Set A's independent re-score (K∈{256,1024}, informational,
not deciding) reaches the same SHIPS verdict. D7 failed on the fit guard (12.4 GB needed, ~4 GB
available) — the same outcome and the same acceptance L1's gate made; S is sufficient for the
pooled decision. **The gate answered whether the kernel is fit to ship; flipping the default
was then asked of, and approved by, the user separately — `GOINFER_METAL_FUSED_ATTENTION`
defaults ON as of 2026-09-10** (`metalFusedAttentionEnabled`, `metal/backend.go:383`).

The rest of this doc (§1–§4) is the 2026-09-09 write-up: `attention_prefill_fused` built,
correctness-tested (kernel + full-pipeline), and measured end to end (4.23× at S/K=3900) —
all still true and unchanged by the gate.

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

## 4. Scope of what this is NOT (as of 2026-09-09 — superseded by §5)

- **Not yet a §3 fidelity gate.** No decision-set sweep, no D7 cell, no pooled §3.2 form
  comparing against the CPU f32 reference the way L1's gate did. **§5 (2026-09-10) is that gate,
  and it SHIPS.**
- **Not yet a default change.** `metalFusedAttentionEnabled()` defaults OFF; `attention_prefill`
  (exact) stays the shipped path. Opt in with `GOINFER_METAL_FUSED_ATTENTION=1`. **§5 records the
  gate pass and the subsequent default flip — this bullet describes 2026-09-09, not the current
  state.**
- **hd ≤ 128 only.** `attention_prefill_fused` requires `hd%8==0 && hd<=128` (`ATTN_MAXHD`,
  `metal/prefill.go`); `PrefillLast` falls back to the exact kernel outside that range. Every
  family measured or benched here (qwen2.5-coder-1.5b: hd=128; the Gemma/MoE parity fixtures:
  hd=16/hd=8) is inside it, but a family with hd=256 (some MoE/long-context archs) is not yet
  covered and would need either a wider `ATTN_MAXHD` (more threadgroup memory per simdgroup —
  budget math in the kernel's own comment) or an accepted fallback.
- **Single model measured end to end.** Only S; D7 (7B) unmeasured for this kernel specifically
  (the L1 gate's D7 cell hit a fit-guard on this 16 GB machine — the same constraint would apply
  here).

## 5. The §3 gate (2026-09-10) — SHIPS

**`TestPrefillGateVsReference` run with `GOINFER_METAL_FUSED_ATTENTION=1`** — same harness, same
pooled §3.2 form, same `~/goinfer-logs/prefill-ref-b/` reference files L1's own gate scored
against, so the "fast" arm here is `PrefillLast` with the fused kernel selected instead of the
exact one. Log: `~/goinfer-logs/l2-metal-gate-b-20260910-080216-fixed.log` (durable). Run time:
S subtest 2620.75 s (43.7 min); D7 failed the fit guard in 0.12 s.

A pre-existing fragility in the shared harness had to be fixed first, unrelated to this kernel:
`metal/prefill_gate_ref_test.go`'s `decoder.Load` never pinned `ResidentContext`, so the 0
(backend-default) auto-cap sizes context off *available memory*, not Metal's fixed
`metalCtxCapMax` (4096) kernel-score-buffer ceiling — on a box with enough free RAM it picks a
context above 4096 and `BuildResident` declines outright, falling back to CPU/staged and failing
this test's `*metalResident` type assertion ("metal resident not built for this model"). Pinned
`ResidentContext: metalCtxCapMax` — comfortably covers every cell here (K=3900 confirm +
continuationN(64) tops out at position 3962). This would have broken a plain re-run of L1's own
gate too; it is not new to fused attention.

**S, decision set (prompt set B, K∈{256,512,1024}, DECIDING) — pooled verdict: SHIPS.** Fused
beats exact on every criterion, not just ties it:

| criterion | exact | fused | pass? |
|---|---|---|---|
| critA hard flips (fast ≤ exact + 2√exact) | 18 | **15** | true |
| critB agreement (fast ≥ exact − 2√d/N, d=59, N=1920) | 92.76% | **92.81%** | true |
| critC mean KL (fast ≤ exact, lower on ≥half prompts, no cell >1.1×) | 0.0405 | **0.0381** (22/30 prompts lower, ceiling OK) | true |

Per-cell (set B):

| K | exact agree / HF / meanKL | fused agree / HF / meanKL |
|---|---|---|
| 256 | 93.0% / 4 / 0.0347 | 94.1% / 3 / 0.0316 |
| 512 | 95.0% / 4 / 0.0347 | 94.4% / 4 / 0.0314 |
| 1024 | 90.3% / 10 / 0.0520 | 90.0% / 8 / 0.0512 |
| 3900 (confirm, not gating) | 91.2% / 10 / 0.0508 | 92.3% / 10 / 0.0493 |

**Set A, re-scored (K∈{256,1024}, informational, NOT deciding) — same verdict: SHIPS.**
critA 75→73, critB 87.27%→87.03% (d=35, N=1280), critC meanKL 0.3008→0.2982 (17/20 prompts
lower). Set A's K=1024 cell includes a rough patch (prompt 2: both arms collapse to ~20% agree,
KL≈4.24 — a hard prompt for both arms alike, not a fused-specific defect, since exact and fused
move together). Two independent prompt sets landing on the same verdict is corroboration, not a
second decision — only set B's pooled result gates.

**D7 — FAILED the fit guard** (needs 9.3 GB, 3.9–4.2 GB available on this 16 GB Mac mid-session),
same outcome and same acceptance L1's gate made. S is sufficient for the pooled decision.

**What this gate decided, and what happened next.** It answered "is `attention_prefill_fused`
fit to ship" — yes. Per the standing rule ("ask before: changing a default without its gate
cell"), the default flip was asked of the user separately rather than made by the gate itself;
the user said yes, and `metalFusedAttentionEnabled()` defaults ON as of 2026-09-10
(`metal/backend.go:383`).
