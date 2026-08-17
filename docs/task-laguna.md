# Laguna (poolside) — G6

Target: **three generations under one `laguna` adapter** — `Laguna-XS-2.1`, `Laguna-XS.2`,
`Laguna-M.1`. Filed from the G6 queue entry; this doc is the build record.

## Phase 0 — config-verified against the real releases (2026-08-17, `linux-62gb`)

Verified before estimating, per the discipline G4 earned (G4's assumptions were wrong twice).
All three configs and the vendor `modeling_laguna.py` were read from the released repos.

### The three generations

| | XS-2.1 | XS.2 | M.1 |
|---|---|---|---|
| layers / hidden | 40 / 2048 | 40 / 2048 | 70 / 4096 |
| heads (q/kv) × head_dim | 48/8 × 128 | 48/8 × 128 | 64/8 × 128 |
| `num_attention_heads_per_layer` | `[48,64,64,64,…]` | `[48,64,64,64,…]` | **absent (uniform)** |
| `layer_types` | 10 full / 30 sliding | 10 full / 30 sliding | **absent (all full)** |
| `sliding_window` | 512 | 512 | **0** |
| experts / top-k | 256 / 8 | 256 / 8 | 256 / **16** |
| `moe_intermediate_size` / shared | 512 / 512 | 512 / 512 | 1024 / 1024 |
| `moe_routed_scaling_factor` | 2.5 | 2.5 | **1.0** |
| dense (`mlp_only_layers`) | `[0]` | (absent) 1 dense | `[0,1,2]` |
| `gating` spelling | `"per-head"` | **`true`** | `"per-element"` |
| top-level `partial_rotary_factor` | absent | 0.5 | absent |
| est. params | ~33B-A3B | ~33B-A3B | **~220B** |

Common to all three: `model_type: "laguna"`, vocab 100352, `rms_norm_eps` 1e-6,
`attention_bias` false, `tie_word_embeddings` false, bf16, `max_position_embeddings` 262144,
eos `[2,24]`, `moe_apply_router_weight_on_input` false, `router_aux_loss_coef` 0.0.

**The vendor modeling code is byte-identical across generations** (only relative-import paths
and a `transformers>=5.12` conversion-mapping shim differ) — so one adapter genuinely serves all
three, and the generational differences are entirely config.

**M.1 is structurally SIMPLER, not harder**: no sliding window, no per-layer head counts, no
routed scaling. It exercises the per-element gating path at depth and nothing else new.

### What the scoping missed

1. **Per-layer QUERY head count.** `num_attention_heads_per_layer` = `[48,64,64,64,48,…]` on the
   XS line — full-attention layers use 48 heads, sliding layers 64 (confirmed by grouping heads
   by layer type: `{full: [48], sliding: [64]}`). goinfer has `headDimAt`/`kvHeadsAt`/`ffnAt`
   per layer but **no per-layer query-head count**; that is a new accessor plus every site that
   derives `qDim` from the scalar `NumHeads`.
2. **Three spellings of `gating` across three releases of one `model_type`.** `"per-head"` (XS-2.1),
   `true` (XS.2), `"per-element"` (M.1). The vendor resolves them as
   `self.gating = bool(gating); self.gate_per_head = (gating == "per-head")` — so `true` and
   `"per-element"` are the SAME path and only `"per-head"` differs. Accept all three from the
   start; this is G4's schema-drift lesson repeating within one family.

### The one genuinely new primitive, exactly

The vendor's own comment: Laguna attention *"is identical to Qwen2MoE attention except"* — no QKV
bias, explicit `head_dim`, per-layer SWA, and **output gating applied BEFORE `o_proj`**:

```python
gate = F.softplus(self.g_proj(hidden_states).float()).to(attn_output.dtype)   # hidden_states = POST input_layernorm
if self.gate_per_head:      # gate [.., num_heads], broadcast across head_dim
    attn_output = (attn_output.view(..., num_heads, head_dim) * gate.unsqueeze(-1)).view(attn_shape)
else:                       # gate [.., num_heads*head_dim], elementwise
    attn_output = attn_output * gate
attn_output = self.o_proj(attn_output)
```

`g_proj: Linear(hidden → num_heads | num_heads*head_dim, bias=False)`, per layer. Two parity
details that must be preserved: the gate reads the **post-input_layernorm** hidden states (the
same tensor q/k/v read, so no extra tap is needed), and **softplus is computed in float32** and
cast back.

### What reuses shipped primitives

- **Router**: sigmoid scoring + `e_score_correction_bias` added to SELECTION scores only (weights
  stay unbiased) + `norm_topk_prob` + `routed_scaling_factor` — this is exactly goinfer's
  `MoEConfig{RouterSigmoid, RoutedScale, NormTopKProb}` (deepseek / glm4_moe).
  `moe_router_logit_softcapping` is 0.0 in all three releases (path exists in vendor code, unused).
- **Shared expert**: `LagunaMLP` — a normal gated SwiGLU, so `SharedUngated: false`
  (GLM's is ungated; Laguna's is not — do not copy that field from glm4_moe).
- **Mixed dense/MoE prefix**: `mlp_only_layers` / `mlp_layer_types` → `FirstKDense` (dense layers
  are a contiguous prefix in all three: `[0]`, `[0]`, `[0,1,2]`).
- **Partial rotary, GLM-style non-interleaved** (`rotate_half` over the first `rotary_dim`,
  pass-through beyond) — glm4/phi3 already do this.
- **Layer-type-keyed RoPE** is *mostly existing machinery*: `RoPELocalBase`/`RoPEGlobalBase` plus
  `ropeScaling`/`ropeScalingLocal` (arch.go already carries a separate local scaling). Only the
  per-layer-type **rotary dim** needs generalizing beyond the gemma4-gated `GlobalRotaryDim`.
- **SWA/global interleave**: `SlidingWindow` + `layerIsGlobal` (gemma3/gemma4/mellum2/gpt-oss).
- **Attention sinks are NOT enabled** in any release (`swa_attention_sink_enabled` absent), so the
  vendor's sink branch is dead weight here — do not implement it.

### Traps to carry into Phase 1

- **`partial_rotary_factor` precedence.** It appears BOTH top-level and inside each
  `rope_parameters[layer_type]`. The vendor warns that HF's `standardize_rope_params`
  *"unconditionally overwrites `rope_parameters["partial_rotary_factor"]` with
  `self.partial_rotary_factor`"*, and works around it by aligning the top-level field to the SWA
  value on a cloned config. Read the per-layer-type value; treat the top-level as a fallback only.
- **RoPE params differ per generation**, not just per layer type: full-attention YaRN is
  `factor 32 / original_max 8192` on XS-2.1 but `factor 64 / original_max 4096` on XS.2 and M.1
  (`attention_factor` differs to match). Config-driven — do not hardcode.
- **Sliding layers use `rope_type: "default"`, theta 10000, partial 1.0**; full layers use YaRN,
  theta 500000, partial 0.5 (M.1: partial 1.0). So on the XS line the two layer types differ in
  base, scaling, AND rotary width simultaneously.
- **Experts ship as stacked fused 3D tensors**: `gate_up_proj [E, 2*inter, hidden]` (gate‖up) and
  `down_proj [E, hidden, inter]` — close to the existing llama4/GGUF stacked-expert handling.
- **`e_score_correction_bias` is remapped** by `_checkpoint_conversion_mapping` from
  `mlp.experts.e_score_correction_bias` (vLLM-trained) to `mlp.gate.e_score_correction_bias`.
  Accept both keys.

### Real-checkpoint gate feasibility

- **XS.2 (~63GB bf16, 14 shards)** — downloadable and runnable on this box; **this is the T3 gate**.
  Its per-element gating is also the more demanding of the two granularities.
- **XS-2.1** — same shape; covered by a tiny golden that exercises the `"per-head"` path.
- **M.1 (~400GB+, 89 shards)** — does not fit this box (62GB RAM). Tiny-golden parity only, with
  the real gate deferred and disclosed, the same call made for Kimi K2. Its code path is identical
  to XS.2's apart from config, and XS.2 gates that path against a real checkpoint.

### Verdict

Estimate holds at **"otherwise cheap"**: one genuinely new primitive (softplus output gating, two
granularities) plus one new accessor (per-layer query heads), on an otherwise-familiar
sigmoid-routed MoE with a shared expert. The layer-type-keyed RoPE that looked new is mostly
existing machinery.

**Strategic tie-in is real, not prospective:** `poolside/Laguna-XS.2-speculator.dflash` and
`poolside/Laguna-S-2.1-DFlash` are published, and P10's block drafting shipped today
(`serve --drafter`). Laguna would be the first pairing with a vendor-blessed drafter.
