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

### Step 0 — partly done

**Landed (this pass — no model needed, fully unit-tested):**
- **GGUF `general.architecture` = `"gemma4"`** (confirmed via the HF API; note it
  differs from the HF `model_type` `gemma4_unified_text`). File:
  `gemma-4-12b-it-qat-q4_0.gguf` (~11.9 GB), ships an `mmproj-*` sibling (optional
  vision — ignore for text decode).
- `Config` fields `GlobalHeadDim` / `NumGlobalKVHeads` / `AttentionKEqV`
  (`config.go`); `Architecture.gemma4 *gemma4Params` (`arch.go`).
- **`gemma4Architecture(cfg)`** descriptor (`registry.go`) encoding the verified
  12B arch, with `TestGemma4Architecture` pinning every knob (head_dim 256/512,
  p-RoPE 128 = 0.25·512, K=V, dual RoPE, 5:1 interleave, QK-norm, Sandwich4, …).
  **Not registered yet** — the GGUF loader returns a clear WIP error for `"gemma4"`
  so nothing mis-runs on the uniform-head path.

**Still required before the forward pass (needs a real checkpoint):** the exact
`gemma4.*` GGUF metadata key names + tensor names (esp. how K=V is stored — one
tensor or duplicated), and the **E2B** config to confirm the PLE/AltUp/Laurel
fields for Phase 2.

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

---

## UPDATE — source-grounded spec (read `transformers/modeling_gemma4.py`)

Pivoted the target to **E2B** (it's the only Gemma 4 that fits 16 GB end-to-end:
goinfer-run at int4 ~2.5 GB **and** the HF bf16 golden ~10–12 GB). The reference
is **HF transformers**, not llama.cpp — so llama.cpp's PLE gap is irrelevant. Read
the real forward pass from `modeling_gemma4.py` (2645 lines; key classes below).

### Big correction: there is **NO AltUp and NO Laurel**
`grep` of the source finds neither — those are **Gemma 3n** mechanisms the earlier
research conflated. Gemma 4's only E-model addition is **PLE** (per-layer
embeddings). The forward pass is a standard Gemma sandwich block + a bounded PLE
residual branch + a per-layer learned scalar. Much smaller than feared.

### Decoder layer (`Gemma4TextDecoderLayer.forward`, ~L1398) — per layer i
```
# attention sandwich
r = h;  x = input_layernorm(h);  x = self_attn(x, layer_type);
x = post_attention_layernorm(x);  h = r + x
# MLP sandwich (E2B dense GeGLU gelu_tanh; MoE only if enable_moe_block → 26B)
r = h;  x = pre_feedforward_layernorm(h);  x = mlp(x);
x = post_feedforward_layernorm(x);  h = r + x
# PLE branch (only if hidden_size_per_layer_input > 0)
r = h;  x = per_layer_input_gate(h)          # Linear hidden→256, no bias
x = gelu_tanh(x);  x = x * per_layer_input_i # elementwise gate by layer i's PLE vec [256]
x = per_layer_projection(x)                  # Linear 256→hidden, no bias
x = post_per_layer_input_norm(x);  h = r + x # RMSNorm
h = h * layer_scalar                         # per-layer learned scalar (buffer, init 1)
```
All norms are Gemma RMSNorm with the `(1+w)` offset.

### Attention (`Gemma4TextAttention`, L1177)
```
is_sliding = layer_type=="sliding_attention"
head_dim   = (!is_sliding && global_head_dim) ? global_head_dim(512) : head_dim(256)
alt        = attention_k_eq_v && !is_sliding          # K=V on global layers
n_kv       = alt ? num_global_key_value_heads(1) : num_key_value_heads(8)
q = q_norm(q_proj(h));  q = rope(q, layer_type)
k = k_proj(h);  v = alt ? k_proj_output : v_proj(h)   # alt → V shares K's projection
k = k_norm(k);  k = rope(k, layer_type)
v = v_norm(v)                                         # NEW: RMSNorm WITHOUT scale (with_scale=False)
attn = softmax(qkᵀ * scaling) · v;  out = o_proj(attn)
```
- **`self.scaling = 1.0`** (NOT 1/√head_dim, NOT query_pre_attn_scalar^-0.5 like
  Gemma 3) — ⚠️ verify against the golden where the query scaling actually lives
  (q_norm? folded?). This is the #1 parity risk.
- New vs Gemma 3: **v_norm** (scale-less RMSNorm on V), variable head_dim/KV-heads,
  K=V on global.

### RoPE (`Gemma4TextRotaryEmbedding`, L1093) — per layer_type
Separate inv_freq per type. `sliding` = default RoPE (base `rope_local`, dim 256).
`full` = **`proportional`** rope (`ROPE_INIT_FUNCTIONS["proportional"]`, base
`rope_global` 1e6, `head_dim_key="global_head_dim"` 512, `partial_rotary_factor`
0.25) — this is the "p-RoPE". **TODO: pull the exact `proportional` formula from
`transformers/modeling_rope_utils.py`.**

### PLE inputs (`Gemma4TextModel`, L1586 + `get/project_per_layer_inputs`)
```
embed_tokens_per_layer: ScaledWordEmbedding(vocab_per_layer 262144 → L*256, scale √256=16)
token_identity   = embed_tokens_per_layer(input_ids).reshape(.,L,256)
context_aware    = per_layer_projection_norm( per_layer_model_projection(inputs_embeds) * hidden^-0.5 ).reshape(.,L,256)
per_layer_inputs = (token_identity + context_aware) * 2^-0.5    # per_layer_input_scale
# layer i consumes per_layer_inputs[:,:,i,:]
```
main embed: `embed_tokens(ids) * √hidden`. Final: `norm` then tied `lm_head`.

### New tensors for E2B (vs gemma3TensorSchema)
Per layer: `per_layer_input_gate` [256,H], `per_layer_projection` [H,256],
`post_per_layer_input_norm` [H], `v_norm` [head_dim] (scale-less → may be absent),
`layer_scalar` [1]. Model: `embed_tokens_per_layer` [262144, L*256],
`per_layer_model_projection` [L*256, H], `per_layer_projection_norm` [256].
(GGUF names per `convert_hf_to_gguf.py` — pin from a real dump in Step 0.)

### Revised effort (E2B, against HF spec, validated on 16 GB)
- Resolve the 2 TODOs (proportional rope formula; the scaling=1.0 query question)
  + get the E2B config dims + a GGUF tensor-name dump: ~0.5 day.
- Forward pass (variable head_dim/KV/K=V, v_norm, per-type RoPE, PLE branch +
  inputs, layer_scalar): **~4–7 days** — bounded, no AltUp/Laurel.
- HF golden (E2B fits 16 GB) + parity gate: ~1 day. Total ~1–1.5 weeks.

---

## STEP 0 COMPLETE — real E2B GGUF dump (`gemma-4-E2B_q4_0-it.gguf`, 3.1 GB)

`general.architecture = "gemma4"`. The two TODOs are resolved and several more
parity-critical deltas surfaced. **E2B differs from the 12B** (no K=V; heavy KV
sharing; PLE present). Descriptor fixes already applied: `AttnScale=1.0`,
`RMSAddOne=false`. **Still to encode (forward-pass phase):**

**Metadata (`gemma4.*`):**
- dims: `embedding_length 1536`, `block_count 35`, `attention.head_count 8`,
  `head_count_kv 1`, `embedding_length_per_layer_input 256` (PLE dim).
- **variable head_dim**: `attention.key_length/value_length 512` (global),
  `*_swa 256` (sliding); `rope.dimension_count 512` / `_swa 256`.
- **variable FFN**: `feed_forward_length = [6144×15, 12288×20]` (per-layer!).
- **`attention.shared_kv_layers 20`** → first-shared = 35−20 = layer 15.
- `attention.sliding_window 512`; `sliding_window_pattern` = explicit per-layer
  bool array, **4 sliding : 1 global** (global at i where (i+1)%5==0: 4,9,…,34).
- **`final_logit_softcapping 30`** ⚠️ (Gemma 4 re-added it; verify HF applies it).
- RoPE: `rope.freq_base 1e6` (global), `freq_base_swa 10000` (sliding).
- `tokenizer.ggml.eos_token_id 1`.

**Per-layer tensors (llama.cpp names):** `attn_norm` (input), `post_attention_norm`,
`ffn_norm` (pre-FFN), `post_ffw_norm` (post-FFN), **`post_norm`** (= the PLE branch's
post_per_layer_input_norm), `attn_q/k/v`, `attn_q_norm`/`attn_k_norm` (**no v_norm
weight** — scale-less), `ffn_gate/up/down`, **`inp_gate`** [H,256] (per_layer_input_gate),
**`proj`** [256,H] (per_layer_projection), **`layer_output_scale`** [1] (layer_scalar).
- **Layers 0–14**: own `attn_k/v` + `k_norm` (k/v length 256 sliding / 512 global).
- **Layers 15–34 (KV-shared)**: NO `attn_k`/`attn_v`/`attn_k_norm` — reuse the KV
  of the last non-shared layer *of the same type* (sliding vs global), per the HF
  `shared_kv_states[layer_type]` + `store_full_length_kv` logic.

**Model-level:** `token_embd`, `output_norm`, the PLE inputs (`per_layer_token_embd`
[262144, 35·256], `per_layer_model_projection`, `per_layer_projection_norm`) —
confirm exact names on the next dump.

**Descriptor additions still needed:** `FinalLogitSoftcap=30`; per-layer
`IntermediateDim` (variable FFN); per-layer head_dim/KV via the bool pattern;
`SharedKVLayers=20` (+ the same-type reuse rule); PLE dims. The K=V flag is OFF
for E2B (distinct V) — it's the 12B that sets `attention_k_eq_v`.
