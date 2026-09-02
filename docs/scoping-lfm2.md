# Scoping: LFM2.5-2.6B as an EXPERIMENTAL family

**Status (updated 2026-08-31):** scoping COMPLETE, build NOT started, and **no longer
freeze-gated — the core-numerics freeze was LIFTED 2026-08-18/19** (`docs/ollama-chase.md`
§"Formerly freeze-blocked"). This page said "cannot land until the v1.0 tag lifts the freeze" for
thirteen days after it had been lifted; core edits have shipped repeatedly since.

**What that changes and what it does not.** The prohibition is gone; the **cost is not**. §G's blast
radius is unchanged — a realistic LFM2 surface edits `registry.go`/`config.go`/`arch.go`/
`weights.go`/`kvcache.go`, which re-stales `deps_hash` for all 19 enforced families and forces a
goldens re-validation. That was always the real content of the "freeze" objection, and it is now a
**scheduling cost to be paid deliberately** rather than a gate that forbids the work.

**Verdict:** architecturally ~80% composition of operators goinfer already has — no new numeric
kernel, no new correctness frontier. Investigated 2026-08-11 via 5 parallel code+web sweeps.

## A. Same architecture as LFM2? — Yes, a scaled retrain

`LiquidAI/LFM2.5-2.6B` is `model_type: lfm2`, `Lfm2ForCausalLM`; topology byte-identical to
`LFM2-2.6B`: 30 layers, `layer_types` = **22 conv / 8 full_attention** (attn at 2,5,9,13,17,21,24,27),
hidden 2048, GQA 32/8, head_dim 64, `conv_L_cache=3`, `conv_bias=false`, `intermediate_size=10752`,
RMSNorm, SwiGLU, tied embeddings, `block_auto_adjust_ff_dim=false`.

Differences vs LFM2-2.6B are **scale/hyperparameter only** — and two contradict the original brief:
- **vocab 128,000**, not 65,536 (65536 is the older LFM2-2.6B tokenizer).
- rope_theta **1e7** (was 1e6); max_position **131,072**; rope via nested `rope_parameters`
  (`rope_type:"default"`, no scaling); `layer_types` (not `full_attn_idxs`).

## B. Mamba conv reusable? — Composition in design, a new copy in code

Mamba-2's causal depthwise conv1d is **inlined, not factored out**: three verbatim copies —
`decoder/mamba2.go:89-105` (decode), `decoder/mamba2_chunked.go:60-75` (prefill oracle, not wired), `decoder/deltanet.go:176-175`
(bias-free). DeltaNet reusing the identical loop against a **non-SSM** recurrence proves the conv is
algorithmically SSM-independent and its rolling-state prefill/decode boundary is solved. `conv_bias=false`
matches DeltaNet's bias-free form. **New: ~15 lines** (a 4th copy, or extract a shared
`causalDepthwiseConvSiLU` helper — the 3-way dup argues for it) **+ ~10-15 lines** input-dependent gate.

## C. Mixed per-layer cache state? — Yes

`KVCache` (`decoder/kvcache.go:50-146`) already holds parallel per-layer arrays (`keys/vals`, `rings`, `mamba`,
`delta`, `mlaLatent`), populated only on layers of the matching kind. Granite/Nemotron are live
Mamba+attention hybrids; Nemotron's `blockKind []uint8` (`decoder/arch.go:186-174`) is the per-layer-type
template. LFM2's rolling conv state is a **strict subset of `mamba2State`** (the `convWin` field already
exists in both `mamba2State` and `deltaState`). New: `shortConvState` + `conv []*shortConvState` slice +
alloc/reset/truncate, ~30-50 lines verbatim from `mamba`/`delta`. All recurrent hybrids run
**one-token-per-call for prefill AND decode** (no batched prefill) — a simplifier. **In frozen `kvcache.go`.**

## D. Closest interleave — Qwen3.5-MoE

The only candidate with a *true two-mixer* per-layer dispatch (`decoder/forward_qwen35.go:30-34`:
`isLinearLayer(l)` → `gatedDeltaNetStep` vs `qwen35Attention`), driven by `layer_types` — the exact twin
of LFM2's conv-vs-GQA split. Llama 4 / Mellum2 toggle a parameter inside one shared mixer. **Build on
Qwen3.5-MoE**: copy `layerIsLinear→layerIsConv`, `IsLinearLayer→IsConvLayer`,
`qwen35Architecture→lfm2Architecture`, write `runLayersLFM2`.

## E. QK-norm — likely a NON-ISSUE (confirm)

The brief said "QK LayerNorm (not RMSNorm)." The reference `modeling_lfm2.py` uses **`Lfm2RMSNorm(head_dim)`
per-head** — RMSNorm. **If RMSNorm (confirm directly), LFM2 uses goinfer's existing hardcoded RMSNorm
QK-norm path (`decoder/attention.go:97-96`) with ZERO new code.** This is exactly the "quiet wrong answer" risk —
verify before building. Contingency if LayerNorm: the `layerNorm(x, weight, bias, …)` primitive already
exists and handles bias (`decoder/rmsnorm.go:49`; `decoder/config.go:903-706` flags LayerNorm-QK as a known Phase-2
primitive) → ~25 lines forward-selector + `.bias`-tensor plumbing.

## F. Computed FFN dim — moot here (stated 10752)

`block_auto_adjust_ff_dim=false` → FFN = `intermediate_size = 10752` **verbatim**; no 2/3 rule. General
formula (toggle `true`): `ff=int(2/3·intermediate_size)`; `ff=int(block_ffn_dim_multiplier·ff)`;
`ff=block_multiple_of·ceil(ff/block_multiple_of)`. MLP is LLaMA SwiGLU `w2(silu(w1·x)·w3·x)`. **Loader
must read `intermediate_size` directly and GUARD on the toggle** — applying the 2/3 formula when the
toggle is off is the quiet-wrong-answer.

## G. Blast radius — cannot be freeze-safe

Hashed set = 29 `decoder/` files: `core` (`registry.go`, `arch.go`, `config.go`, `model.go`,
`attention.go`, `kvcache.go`, `rmsnorm.go`, …), `loaders` (`weights.go`, `gguf.go`), per-family `own`
forwards. 19/23 families enforced; all `uses` core+loaders.

| Touch | Hashed? |
|---|---|
| `registry.go` — `"lfm2": lfm2Architecture` map entry (no additive seam outside a hashed file) | **YES (core)** → all 19 restale |
| `arch.go` — `lfm2Params` + `isConvLayer` | **YES (core)** |
| `config.go` — fields + `validateLFM2` | **YES (core)** |
| `weights.go` — `lfm2TensorSchema` + `buildLFM2Weights` | **YES (loaders)** |
| `kvcache.go` — `conv` slice + reset/truncate | **YES (core)** |
| `gguf.go` — arch map (optional; safetensors-first skips) | **YES (loaders)** |
| `forward_lfm2.go` (NEW) | **NO — freeze-safe** (LFM2's own entry) |
| `pin_lfm2_tiny.py`, `lfm2_test.go`, `testdata/`, manifest row | **NO — freeze-safe** |

The `registry.go` one-liner alone re-stales all 19. **Post-freeze item, not a side project.**

**Experimental tier** (`parity-coverage-policy.md` + `TestParityManifest_methodTier`): REQUIRES
`status:"experimental"` with a non-empty `method` + a committed T1 tiny-golden; FORBIDS
`status:"validated"` (needs a T3 real-oracle method); EXCLUDED from the supported count. No T3 required to
ship experimental.

## Size estimate

| Component | Size | Notes |
|---|---|---|
| Conv operator | **S** ~25-40 lines | reuse mamba/delta loop + solved boundary; new = bias-free copy + gate |
| Cache work | **S** ~30-50 lines | `shortConvState` + slice + alloc/reset/truncate; frozen `kvcache.go` |
| Interleave | **S** ~40-60 lines | copy Qwen3.5-MoE pattern; frozen `arch.go`/`config.go`/`registry.go` + new forward |
| Loader | **M** ~100-150 lines | `lfm2TensorSchema` + `buildLFM2Weights` (safetensors); frozen `weights.go`. GGUF deferred |
| QK-norm | ~0 if RMSNorm; S if LayerNorm | confirm first |
| Parity (T1) | **S** ~100 lines | `pin_lfm2_tiny.py` + `lfm2_test.go` + manifest row — freeze-safe |

**Total ≈ 350-500 new lines, of which only ~15-40 are novel logic** (the gated short-conv step).
Everything else is mechanical loader/config/schema mapping + forward assembly of existing operators.
By goinfer standards a **SMALL family** — far below DeltaNet/Mamba/MLA, which each introduced a new
mixer with a new correctness frontier. LFM2 introduces **no new kernel and no new boundary problem** —
it composes existing ones.

## Sequencing

1. ~~**Cannot land under the freeze**~~ — **superseded 2026-08-31, the freeze is lifted.** Still
   max blast radius, so it should be **batched with a goldens re-validation run** rather than landed
   on its own: the re-validation is the cost, and paying it once for several families beats paying it
   per family.
2. **Prep already done:** `scripts/pin_lfm2_tiny.py` + the T1 tiny-golden (committed alongside this
   doc). Drafting `forward_lfm2.go` in isolation is no longer necessary as a freeze workaround — the
   wiring can now be written directly.
3. **Confirm E** — **partly discharged.** Two independent in-tree sources now say RMSNorm: §E's read
   of `modeling_lfm2.py` (`Lfm2RMSNorm(head_dim)` per-head) and `scripts/pin_lfm2_tiny.py`'s own
   header, written against the real `LFM2.5-2.6B` config, which states "per-head Q/K RMSNorm". What
   is still owed is **weight-level** confirmation — that no `q_layernorm.bias`/`k_layernorm.bias`
   tensor exists in the real checkpoint — which is a tensor-name check at load, not an experiment.
   No LFM2 checkpoint is present on either box, so it cannot be done from here today.
4. safetensors-first, CPU-only, experimental tier; GGUF deferred (llama.cpp `lfm2` GGUFs exist and are
   first-class, so it's a straightforward follow-on).

GGUF status: llama.cpp natively supports arch `lfm2`; official `LiquidAI/LFM2.5-2.6B-GGUF` exists.
