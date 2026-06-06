# Task (goinfer): Gemma 4 model support

> **For:** Claude Code in `~/tmcode/goinfer`. **Why:** Google released Gemma 4 QAT
> checkpoints (June 2026) as Q4_0 GGUF + W4A16, tuned for local/CPU. goinfer
> already dequantizes Q4_0 GGUF and supports Gemma **3**, but Gemma 4 is a
> meaningfully new architecture, not "Gemma 3 with new dims." This plan is the
> result of a research pass; **it is a sizeable runtime feature, not a weekend
> demo** (see §0 reality check). Parity (`TestDecodeParity` + forward goldens)
> stays the non-negotiable gate.

---

## 0. Reality check (read first)

The headline marketing model — **Gemma 4 E2B in <1 GB** — is the *hardest* one to
support: the "E" (effective) models get their efficiency from **Per-Layer
Embeddings (PLE)** plus **AltUp** and **Laurel** residual mechanisms — net-new
forward-pass compute that **llama.cpp has not implemented yet** (PLE is tracked
open in ggml-org/llama.cpp#22243). There is **no upstream forward graph to diff
against** for those — only `modeling_gemma4`.

The good news, **verified from the real `config.json`**: the **12B dense** model
is **PLE/AltUp/Laurel-free** — a "plain" Gemma-3-style dense net plus three
bounded deltas (variable head_dim, p-RoPE, K=V). So the work splits:

- **A real Gemma 4 dense model (12B) is achievable** without the hard pieces → the
  "goinfer runs Gemma 4, no cgo" proof point. **But 12B ≈ 7 GB at Q4_0 — not the
  tiny single-file story.**
- **The tiny ≤1 GB E2B demo needs PLE/AltUp/Laurel** → high-risk, no upstream
  reference, multi-week, and arguably should wait until upstream lands a reference.

**Recommendation:** this is a roadmap feature, not a launch add-on. If pursued,
**Phase 1 (12B dense)** is the realistic first milestone; the tiny E2B demo is a
later, riskier phase gated on PLE.

---

## The lineup (from the QAT Q4_0 collection)

Text-only (goinfer's wheelhouse): **E2B** (~5B actual / "effective 2B"), **E4B**
(~8B), **12B**. Each ships as `…-gguf` (Q4_0), `…-unquantized` (safetensors, for
goldens), and `…-w4a16-ct`. The **26B-A4B** (MoE) and **31B** are **multimodal
(image-text-to-text)** → their vision towers are **out of scope** for a text
decoder; only their text stack would apply, and they're large.

**Bonus tie-in:** each model has a tiny **`…-assistant`** sibling (78 M – 0.5 B) —
these are **speculative-decoding draft models**. Gemma 4 + its official assistant
is a natural showcase for goinfer's [[task-speculative-decoding]] once goinfer has
a bandwidth-bound (GPU) backend where speculative actually wins.

---

## 1. Scope & sequencing

| Phase | Target | New compute | Demo value |
|---|---|---|---|
| **1** | **12B dense** (PLE-free, verified) | variable head_dim · p-RoPE · K=V global · per-layer KV-head count | "goinfer runs Gemma 4" — real but ~7 GB |
| **2** | **E2B / E4B** (≤1 GB headline) | **+ PLE + AltUp + Laurel** | the tiny single-file demo — **high risk, no upstream ref** |
| **out** | 26B-A4B MoE, 31B | multimodal vision tower; large | — |
| **out** | 2-bit LiteRT mobile (`wNa8o8`) | not GGUF | — |

Phase 1's attention deltas are common to all variants and must land first. Phase 2
adds the E-model residual machinery on top.

---

## 2. Architecture deltas vs Gemma 3 — **verified from `google/gemma-4-12B-it…/config.json`**

`model_type: "gemma4_unified_text"`, `architectures: ["Gemma4UnifiedForConditionalGeneration"]`.
12B dims: hidden 3840, 48 layers, 16 heads, 8 KV heads, intermediate 15360, vocab 262144.

| Knob | Gemma 3 | Gemma 4 (12B, verified) | `Architecture` covers it? |
|---|---|---|---|
| RMSNorm `(1+w)` / Sandwich4 / GeGLU `gelu_tanh` / QK-norm / √hidden embed / tied head / no softcap | yes | **same** | ✅ |
| vocab / tokenizer (Gemma SP) | 262144 | **262144, same** | ✅ |
| sliding window + pattern | 512, 5:1 | **1024, 5:1** (`layer_types`), last global | ✅ `layerIsGlobal` (read `layer_types`) |
| dual RoPE base | 10k / 1M | **10k local / 1M global** | ✅ `RoPELocalBase`/`GlobalBase` |
| **head_dim** | 256 uniform | **256 local / 512 global** (`head_dim` + `global_head_dim`) | ❌ **per-layer head_dim** |
| **global KV heads** | = local (8) | **1** (`num_global_key_value_heads`) | ❌ **per-layer KV-head count** |
| **p-RoPE** | none | **`partial_rotary_factor 0.25` on the GLOBAL/full layers only** | ❌ **partial rotary, per-layer (global-only)** |
| **K = V** | distinct | **`attention_k_eq_v: true`** (V tensor == K on shared layers) | ❌ **loader/forward assume distinct K,V** |
| cross-layer KV share | none | **`num_kv_shared_layers: 0` on 12B** (nonzero on others — verify) | ❌ (not needed for 12B) |
| **PLE / AltUp / Laurel** | none | **ALL ABSENT on 12B** (`vocab_size_per_layer_input:0`, `hidden_size_per_layer_input:0`, no altup/laurel) | n/a for Phase 1; **present on E2B/E4B → Phase 2** |

So **Phase 1 (12B) needs exactly three things**: per-layer head_dim, per-layer
KV-head count, and p-RoPE on the global layers, plus K=V loading. Everything else
is the existing gemma3 descriptor.

---

## 3. File-by-file changes

### Step 0 (still required before code)
12B `config.json` is verified above. Still pin from a **real GGUF dump**: the
`general.architecture` string the GGUF uses (HF `model_type` is `gemma4_unified_text`
— confirm the GGUF spelling), the exact `gemma4*.*` metadata keys, and the tensor
names (esp. how K=V is stored: one tensor or duplicated). Also pull the **E2B**
config to confirm the PLE/AltUp/Laurel fields for Phase 2.

### `decoder/arch.go`
- Per-layer head_dim: add `headDimFor func(i int) int` (mirror `layerIsGlobal`) or
  `LocalHeadDim`/`GlobalHeadDim`; route every `arch.HeadDim` read (kvcache sizing,
  both loaders' qDim/kvDim, attention, RoPE table width) through it. **Most
  invasive change** — `HeadDim` is assumed uniform today.
- Per-layer KV-head count: similarly, global layers use `num_global_key_value_heads`.
- `RoPEPartialFactor float64` (or a per-layer rotary-dim), applied to **global**
  layers only — distinct from `RotaryDim` (verify which dims p-RoPE prunes).
- `GlobalKVShared bool` (K==V on shared layers).
- (Phase 2 only) a nil-able `*Gemma4Extras` for PLE/AltUp/Laurel so other families
  pay nothing.

### `decoder/registry.go`
- `gemma4Architecture(cfg)` — start from `gemma3Architecture`, add the four knobs
  above, populate `layerIsGlobal` from `cfg.LayerTypes`. Register `"gemma4_unified_text"`
  (+ whatever the GGUF arch string turns out to be).

### `decoder/config.go`
- Add fields: `GlobalHeadDim`, `NumGlobalKVHeads`, `PartialRotaryFactor`,
  `AttentionKEqV`, `LayerTypes`, `NumKVSharedLayers`, plus the Phase-2 PLE fields.
- `validateGemma4()` — accept the Phase-1 stack; **loudly reject** what a phase
  doesn't implement yet (PLE/AltUp/Laurel present → error in Phase 1; multimodal /
  MoE variant → error). Mirrors `validateMellum` / the Gemma-2 softcap rejection.

### `decoder/weights.go`
- `gemma4TensorSchema` from `gemma3TensorSchema`; compute qDim/kvDim **per layer**
  inside `loadLayer` (variable head_dim + KV-head count). Handle K=V (load V from K
  or skip the V tensor on shared layers).

### `decoder/gguf.go`
- Add the `case` for the Gemma 4 arch string; extend the unsupported-arch error.
- `ggufGemma4Config(g)` from `ggufGemmaConfig`: read dims, `…attention.sliding_window`,
  the per-layer types/pattern, dual `rope.freq_base`, the global head_dim + global
  KV-head keys, the partial-rotary key. Gemma stays on the NEOX no-permute path
  (`ggufQKPermuted` false) — verify p-RoPE doesn't change stored layout. The `(1+w)`
  norm subtraction already keys off `RMSAddOne`. Q4_0 dequant unchanged.

### Forward pass (`attention.go`, `rope.go`, `kvcache.go`, `model.go`, `forwardn.go`)
- Per-layer head_dim + KV-head count in `causalAttention` and **KV-cache slot
  sizing** (`kvcache.go` sizes by head_dim — must be per-layer).
- p-RoPE in `rope.go` (prune low-freq dims, global layers only).
- K=V in attention/cache.
- (Phase 2) PLE/AltUp/Laurel in `runLayers`, gated on `gemma4Extras != nil`.

### `tokenizer/`
- **No change expected** (Gemma SP 262k, byte-fallback exists). Confirm GGUF
  `tokenizer.ggml.*` ids + special/EOS tokens (handled generically).

---

## 4. Parity & test plan (the gate)

1. `scripts/pin_gemma4_forward.py` (mirror `pin_*`): run the **`-unquantized`**
   12B (or a smaller dense if added) through `transformers` on a fixed short
   prompt; dump `{IDs, Argmax, Logits}` → `testdata/gemma4_forward_golden.json`
   (+ `_full.json` cosine oracle). One golden per phase (12B dense; later E2B PLE).
2. `TestGGUF_gemma4_parity` (clone `TestGGUF_gemma3_parity`): load the Q4_0 GGUF,
   assert `argmax == golden` + cosine ≥ ~0.99 (Q4_0 floor), and that the new knobs
   resolved (per-layer head_dim present, K=V wired).
3. Safetensors parity through `buildGemma4Weights` against the same golden (both
   load paths share the contract).
4. `TestDecodeParity`-style greedy continuation pinned to HF greedy (catches
   KV-cache / per-layer-head_dim / K=V bugs the single forward won't).
5. Smoke/bench: confirm it loads + generates on CPU; capture tok/s. (12B is big;
   use the smallest dense available for routine CI.)

Regenerate goldens only on an intentional, parity-reviewed numerics change.

---

## 5. Risks & open questions

- **PLE/AltUp/Laurel (Phase 2 / the tiny demo):** net-new compute, **no upstream
  reference** (llama.cpp#22243 open). Highest risk; derive from `modeling_gemma4`
  + a hand-checked micro-golden. Consider waiting for an upstream impl to diff.
- **Variable head_dim ripple:** touches kvcache sizing, both loaders, attention,
  RoPE width — the most error-prone struct change (subtle silent logit corruption).
- **p-RoPE:** confirm exactly which dims `partial_rotary_factor 0.25` prunes and
  that it's global-layers-only; it is NOT `RotaryDim` trailing-dim drop.
- **K=V + global 1-KV-head:** verify GGUF storage (single vs duplicated tensor)
  and the attention math under 1 global KV head + 512 head_dim.
- **GGUF arch string + key names:** HF `model_type` is `gemma4_unified_text`; the
  GGUF spelling + `gemma4*` keys must be pinned from a real dump (wrong keys = silent
  zero dims).
- **26B/31B multimodal:** vision tower out of scope; text-only path is large + MoE
  (26B reuses existing `MoEConfig`/Qwen2-MoE shared-expert) — defer.
- **2-bit LiteRT mobile (`wNa8o8`):** out of scope (not GGUF).

---

## 6. Effort (checkpoint in hand, Step 0 done)

- **Step 0** (GGUF dump + E2B config + `modeling_gemma4` read): 0.5–1 day.
- **Phase 1 — 12B dense** (variable head_dim + per-layer KV-heads + p-RoPE + K=V):
  **~3–5 days**, dominated by the per-layer-head_dim ripple + parity. Verified
  PLE-free, so no AltUp/Laurel surprise.
- **Phase 2 — E2B/E4B + PLE/AltUp/Laurel** (the ≤1 GB demo): **~1–2 weeks**, high
  variance, no upstream reference.

**Headline tiny demo ≈ Step 0 + Phase 1 + Phase 2 ≈ 2–3 weeks.** A "goinfer runs
Gemma 4 (12B) dense, no cgo" milestone is the cheaper **~1 week** deliverable.

### Critical files
`decoder/arch.go` · `decoder/registry.go` · `decoder/gguf.go` · `decoder/weights.go`
· `decoder/config.go` · forward: `attention.go`/`rope.go`/`kvcache.go`/`model.go` ·
gate: `gguf_forward_test.go` + `scripts/pin_gemma4_forward.py` + `testdata/gemma4_forward_golden.json`.

### Sources
Gemma 4 QAT blog · transformers `gemma4` docs · `google/gemma-4-*-qat-q4_0`
collection (configs + GGUFs) · llama.cpp PLE issue #22243 · `convert_hf_to_gguf.py`.
