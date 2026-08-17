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

**The vendor modeling code is byte-identical between XS-2.1 and M.1** (only relative-import paths
and a `transformers>=5.12` conversion-mapping shim differ). **CORRECTED IN PHASE 1: XS.2's module
is NOT the same file** (34KB vs 41KB) — see the corrections section below. One adapter still serves
all three, but "generational differences are entirely config" was too strong.

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
- **Shared expert**: `SharedUngated: true`, same as glm4_moe. **Read this flag carefully** — it
  does NOT mean "not a SwiGLU". Laguna's shared expert IS a normal gated SwiGLU (`LagunaMLP`),
  but goinfer's `SharedUngated` refers to the OUTER `sigmoid(shared_gate·h)` scalar gate that
  Qwen2-MoE applies to the shared expert's output. Laguna has no such gate — it adds the shared
  output raw (`expert_output = expert_output + shared_expert_output`) — so `SharedUngated: true`.
  Mapping "LagunaMLP is gated" onto `SharedUngated: false` is the natural-looking read and is
  wrong; it would silently multiply the shared branch by a sigmoid the model never trained with.
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


## Phase 1 corrections — what the real checkpoint and XS.2's own module changed

Phase 0 read `config.json` for all three and `modeling_laguna.py` for XS-2.1 and M.1. Increment 1
added two sources Phase 0 did not consult — **XS.2's own module** and the **real XS.2 checkpoint's
tensor index and shapes** — and both overturned assumptions. Recording them because the pattern is
now three-for-three in this family: the released artifact disagrees with the prose.

1. **QK-norm is UNCONDITIONAL, and Phase 0 missed it entirely.** All three modules construct
   `q_norm`/`k_norm` as `LagunaRMSNorm(head_dim, eps=rms_norm_eps)` with no config flag, and the
   real checkpoint ships `self_attn.{q,k}_norm.weight` of shape `[128]` on every layer. Nothing in
   `config.json` mentions it and the vendor's "identical to Qwen2MoE attention except …" list omits
   it. Phase 0 grepped for gating and softplus, not for norms — so the first adapter had
   `QKNorm: false`, which would have been a silent parity failure. Caught before any parity run,
   by reading the checkpoint's tensor names.

2. **The gate's granularity comes from the TENSOR SHAPE, not `config.gating`.** XS.2 declares
   `gating: true`, which the XS-2.1/M.1 module resolves to per-ELEMENT — but XS.2's own module
   hardcodes `nn.Linear(config.hidden_size, self.num_heads)` and never reads the field, and its
   shipped `g_proj` is `[64, 2048]`, i.e. per-HEAD. **The vendor's spelling→granularity rule is
   generation-specific.** So `applyAttnGate` selects on `GProj.Rows()` (`nH` ⇒ per-head,
   `nH*head_dim` ⇒ per-element; they can never collide since `head_dim > 1`), and the config value
   is kept only as the declared expectation. This is the safest possible reading and is immune to a
   fourth spelling.

3. **Experts ship PER-EXPERT, not stacked.** Phase 0 recorded the module's fused 3D parameters
   (`gate_up_proj [E, 2*inter, hidden]`). The checkpoint stores per-expert 2D tensors —
   `mlp.experts.{0..255}.{gate,up,down}_proj.weight`, 9984 = 39 MoE layers × 256 experts of each —
   which HF re-packs at load. That is the form goinfer already reads, so **no stacked-expert
   handling is needed at all** and the estimate got cheaper, not dearer.

4. **The shared expert is SINGULAR on disk.** The module says `self.shared_experts` (plural, as GLM
   and DeepSeek spell it); the checkpoint keys are `mlp.shared_expert.*`.

5. **`e_score_correction_bias` ships under `mlp.experts.*`**, not `mlp.gate.*` — the vLLM-trained
   spelling, which HF rewrites at load. Reading the checkpoint directly means taking the experts
   spelling, as Phase 0 predicted.

6. **Per-layer query heads confirmed in real weights**, not just config: `q_proj` is `[6144, 2048]`
   (48 heads) on layer 0 and `[8192, 2048]` (64 heads) on layer 1.

### Resident-backend safety

Laguna declares a new `FeatAttnOutputGate` resident feature covering both the softplus gate and the
per-layer query heads, so **every resident backend declines it** (`laguna → admitted by []`).
Without that declaration the family needs nothing CUDA lacks and would have been
*admitted-but-mis-run*: the resident path would skip the gate silently and still emit plausible
logits. This is the same failure shape `FeatGemma4EModel` and `FeatAttnSink` exist to prevent.

### Increment 1 — landed

Config resolution (all three real configs), the `laguna` architecture adapter, per-layer query-head
accessor (`headsAt`/`maxHeads`, threaded through `causalAttention`/`attendQuery`/
`attendBatchedHeads` and decode-scratch sizing), `RotaryDimLocal` completing the local/global RoPE
triple, the softplus output gate, the tensor schema, and `GProj` loading with shape-selected
granularity. Gated by `TestLagunaArchitecture_realConfigs` (three real configs; mutation-tested on
rotary width, QK-norm, and per-layer heads), `TestLagunaGating_allThreeSpellings`, and
`TestLagunaFirstKDense_contiguousOnly`. `GProj` is excluded from quantization: it is <1% of a
layer's attention weights and its output multiplies the entire attention context.

**Not yet gated numerically** — tiny goldens and the real XS.2 oracle are the next increments.


## Increment 3 — the real XS.2 gate, and the bug it found

**The real gate failed on its first run, for a real reason**, which is the argument for having
built it. It crashed inside batched prefill with a shape check: `344064 vs 458752` = `56×6144`
vs `56×8192`, i.e. **48 heads vs 64**. Increment 1 threaded per-layer query heads through the
DECODE path (`causalAttention`, `attendQuery`, `attendBatchedHeads`, decode scratch) but not
through `runLayersFromEmbedN`, whose `q`/`ctx` are allocated once from `NumHeads`.

**T1 did not catch it because T1 only drove the sequential path.** The tiny test looped
`m.forward` per token; batched prefill was never exercised. So T1 now runs the prompt through
`prefillLogits` as well and compares against the same golden — the tiny fixtures already carry
the per-layer geometry (4 heads full / 8 sliding), so this class of bug is now caught in 0.02s
instead of after a two-minute 63GB load.

Fixing the crash exposed a second, quieter bug: with the buffers right, batched prefill returned
cosine **0.957021** — byte-identical to the earlier "gate disabled" mutant. The gate was applied
in `causalAttention` and **nowhere in the batched path**. That is a defect that reads as a
plausible-looking number rather than a failure, so the gate math now lives in ONE place
(`applyGateRow`), called by both paths, exactly as `attendQuery`'s own comment argues for.

### Result

Real `poolside/Laguna-XS.2` (33B-A3B, 14 shards, loaded at int4) generates:

> 1. Eiffel Tower
> 2. Notre-Dame Cathedral
> 3. Louvre Museum

distinct-trigram 1.000, three real landmarks, through the vendor's own chat template.

### Why the manifest says `experimental`, not `validated`

The parity policy reserves `validated` for a **T3 method** — cosine/argmax against a full
reference forward of a released checkpoint. This gate is **coherence + structure** on real
weights, which is a genuinely different and weaker claim, so the row stays `experimental` with
method `tiny-golden` and the real gate described in its note. A true T3 would need a bf16 forward
of a 33B model alongside the int4 one; 62GB of RAM does not hold both, so it would have to be a
layer-slice oracle. That is the one open item for this family, and it is a nice-to-have.
