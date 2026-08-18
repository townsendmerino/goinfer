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


## Correction: Laguna GGUFs DO exist, and llama.cpp supports the family natively

Increment 3's write-up said no Laguna GGUFs existed. That was wrong and is worth recording,
because it would have parked a tractable, high-value task indefinitely.

`general.architecture` in these files is literally **`laguna`** — llama.cpp has first-class
support — and official `poolside/Laguna-XS-2.1-GGUF` and `poolside/Laguna-S-2.1-GGUF` exist
alongside many community conversions (bartowski, lmstudio-community, mradermacher, and
`Lucebox/Laguna-XS.2-GGUF`).

The metadata carries everything the adapter needs, in some places more cleanly than the
safetensors config:

| GGUF key | value (XS-2.1) | maps to |
|---|---|---|
| `laguna.attention.head_count` | **array[40]** `[48,64,64,64,…]` | per-layer QUERY heads |
| `laguna.rope.dimension_count` / `_swa` | 64 / 128 | `RotaryDim` / `RotaryDimLocal` |
| `laguna.rope.freq_base` / `_swa` | 500000 / 10000 | `RoPEGlobalBase` / `RoPELocalBase` |
| `laguna.rope.scaling.*` | yarn, factor 32, orig 8192 | `ropeScaling` (full layers only) |
| `laguna.expert_gating_func` | 2 | `RouterSigmoid` |
| `laguna.expert_weights_norm` / `_scale` | 1 / 2.5 | `NormTopKProb` / `RoutedScale` |
| `laguna.leading_dense_block_count` | 1 | `FirstKDense` |
| `laguna.attention.sliding_window` | 512 | `SlidingWindow` |
| `tokenizer.chat_template` | laguna_glm_thinking_v8 | **the template the safetensors dir does NOT carry** |

Tensors are standard llama.cpp naming and land on machinery goinfer already has (stacked
`ffn_*_exps`, `*_shexp`, `exp_probs_b` — the GLM spelling), plus one new one:
**`blk.N.attn_gate.weight`**, stored `[2048, 48]` (GGUF's transposed convention), i.e. per-head
at that layer's own head count. Note there is NO gating-granularity key — so the GGUF path must
read granularity from the tensor shape too, which is what the safetensors path already does.

**One open question for the loader:** the GGUF has no `layer_types` and no sliding-window
PATTERN key, so which layers are full must be derived. The head-count array encodes it (48 on
full, 64 on sliding), which is derivable but an inference — the loader should assert the
resulting 10/30 split rather than trust it silently.


## The DFlash drafter pairing — specified, NOT built

`poolside/Laguna-XS.2-speculator.dflash` is downloaded and inspected. It is **not** a
drop-in for goinfer's shipped DFlash path, and the reasons are specific enough to write down
so this can start cold.

### What the drafter actually is

62 tensors, 5 layers, Qwen3-shaped (q/k/v/o + gate/up/down + per-head `q_norm`/`k_norm`):

| tensor | shape | note |
|---|---|---|
| `fc.weight` | `[2048, 10240]` | 5 taps × hidden — matches `blockTrunk`'s fusion projection |
| `hidden_norm.weight` / `norm.weight` | `[2048]` | context norm / final norm |
| `embed_tokens.weight` | `[100352, 2048]` | **its OWN embedding** (target vocab) |
| `lm_head.weight` | `[32000, 2048]` | **its own REDUCED-vocab head** |
| `d2t` / `t2d` | `[32000]` / `[100352]` | draft↔target id translation |

`block_size: 8`, `mask_token_id: 12`, `aux_hidden_state_layer_ids: [1, 9, 17, 36, 39]`.

### The two mismatches

1. **Head/embedding ownership.** goinfer's DFlash drafters carry no `lm_head` and borrow the
   target's (`DrafterHeadLogits` exists precisely for that). This one ships both an embedding
   and a reduced-vocab head, so drafted ids come out in DRAFT space and must be mapped with
   `target = i + d2t[i]`. **goinfer already implements exactly that** — in `decoder/eagle.go`
   (`d2t`, `DraftVocabSize`). So the work is joining two existing paths, not inventing one.

2. **Config schema.** goinfer's DFlash loader reads a nested `dflash_config`
   (`block_size` / `mask_token_id` / `target_layer_ids`). This file is in vLLM's
   **speculators v0.5** format: `speculators_config` (algorithm `dflash`, proposal method,
   `verifier.name_or_path`), `transformer_layer_config` for the layer geometry
   (`model_type: llama`, 5 layers, 16 heads / 8 KV, head_dim 128, hidden 2048, inter 8192),
   and the block fields at top level under different names
   (`aux_hidden_state_layer_ids` rather than `target_layer_ids`).

### Why it is worth doing, and the honest caveat

It would be the first **vendor-blessed** pairing for the block drafting that shipped in P10
(every prior pairing was third-party). The caveat: the Laguna target is CPU-only in goinfer
today — `FeatAttnOutputGate` makes every resident backend decline — so any speedup measured
now is CPU-side. P10's own kill-gate lesson was that **the draft, not the verify, was the
wall**, and a CPU draft against a CPU target is a different regime from the GPU-resident
numbers gate 3 reported. Measure before believing.

**Estimated: a few hours.** It is a feature, not a leftover, and is deliberately left unbuilt
rather than half-built.


## Leftovers — results

**GGUF (poolside's own `Laguna-XS-2.1-Q4_K_M.gguf`, 20GB): PASSES.** Real generation:

> Here are three iconic landmarks in Paris: **1. Eiffel Tower (Tour Eiffel)** … **2. Louvre
> Museum (Musée du Louvre)** … **3. Notre-Dame Cathedral** …

Two of my own test bugs had to be cleared first, and both are worth recording because neither
was a loader fault:

- `Options{}` on a Q4_K_M file makes goinfer **dequantize to f32** — ~132GB for a 33B model,
  so the process was OOM-killed with only `signal: killed` to show for it. The safetensors
  gate had used `Quant: "int4"`; this one had not. Loading a quantized file does not imply a
  quantized load.
- The first passing run then failed my own landmark assertion: XS-2.1 answers as a verbose
  numbered list with a paragraph per entry, so a 48-token budget stopped inside item 1. Fixed
  by raising the budget to 160, **not** by lowering the bar to one landmark — that check
  exists precisely to separate a right answer from a fluent wrong one.

**Layer-slice oracle now covers BOTH locally-available generations** (XS.2 and XS-2.1), each at
**cosine 1.00000000** on the sequential and batched-prefill paths. Worth having both: XS-2.1
DECLARES per-head gating where XS.2 declares per-element and ships per-head, and their YaRN
factors differ (32 vs 64 ⇒ different mscale and inv_freq). Those are config-driven paths that
only real weights exercise end to end.

**CI went red on the GGUF loader commit** — `TestDispatchCensus` caught
`switch v := g.Metadata["laguna.expert_weights_norm"].(type)`, a type switch on a value's
ENCODING, which is exactly what that census exists to stop. Replaced with plain assertions in
glm4moe's existing style. The process failure was mine: after that commit I ran only targeted
tests instead of CI's own command. Since then every push is preceded by
`go test -race -timeout 25m -tags goinfer_testhooks ./...` locally — the same combination CI
runs, including `-race`, which I had not been using at all.

**The DFlash pairing is NOT done, and it is bigger than first estimated.** Beyond the
speculators-v0.5 config dialect and the reduced-vocab `d2t` head, there is a structural gap:
**only CUDA implements `ResidentBlockDrafter` / `ResidentDrafterHost`.** There is no CPU
block-drafting orchestration, and Laguna is CPU-only by construction (`FeatAttnOutputGate`
makes every resident backend decline it). So a pairing needs a CPU BlockSpec — draft through
`blockTrunk.DraftBlock` (which already works on CPU), verify through the existing batched
`forwardN`, plus the accept/burst and guard logic. Realistically 4–5h if clean, 6–7h with the
debug loop, since each real-model iteration costs ~5 minutes of load before it can fail.


## DFlash pairing — measured

`poolside/Laguna-XS.2-speculator.dflash` against real `Laguna-XS.2` at int4:

**2.86 tokens/round** — 22 rounds, 41 accepted drafts, on a code prompt ("reverse a linked
list"), producing coherent Python. P10's calibrated break-even is **2.5** tok/round
(`breakEvenTokensPerRound`, blockspec.go), so the pairing clears it.

That number is the only thing that can tell a good drafter from a broken one here: block
drafting is lossless by construction, so the output looks identical either way. P10 learned
this twice — a wrong mask token turned a 1.60× pairing into 0.66× while the text stayed
perfectly valid.

### What 2.86 does and does not buy

It does NOT yet buy a speedup, and the reason is structural rather than a tuning problem.
The measurement harness **verifies sequentially** — it feeds the anchor and each drafted
token through `ForwardCapture` one at a time. That is the right instrument for acceptance
(it is exactly what the target would do) but it does no fewer target forwards than plain
decoding, so it can only ever measure whether drafts are accepted, never save time.

A real speedup needs the drafted block verified in ONE batched pass — the CPU analogue of
`PrefillLastNArgmax`, which today exists only on the resident (CUDA) interface. goinfer has
the batched machinery (`runLayersFromEmbedN` / `forwardN`); what is missing is a CPU host
exposing `ResidentDrafterHost` / `ResidentBlockDrafter` so `BlockSpec` can drive it.

**So the corrected picture of my own earlier estimate:** I said step 2 was "build a CPU
BlockSpec, ~2h" and it turned out to be two small seams, because the CPU draft/verify loop
already existed in `dflash_accept_test.go` from the P10 sweeps. I had looked at the
INTERFACES, seen only a CUDA implementation, and concluded no CPU path existed — when what
was missing was a CPU path *exposed through those interfaces*. That distinction is the whole
of the remaining work.


## DFlash pairing — the end-to-end verdict: DO NOT SHIP (0.82x on CPU)

With the CPU BlockSpec doing a real BATCHED verify, the full pairing on real Laguna-XS.2
@int4:

| | value |
|---|---|
| acceptance | **3.20 tok/round** (15 rounds, 48 tokens) |
| break-even constant | 2.5 (`breakEvenTokensPerRound`) |
| speculative wall-clock | 5m58.7s |
| plain greedy wall-clock | 4m54.8s |
| **speedup** | **0.82× — SLOWER** |

Output was token-identical to plain greedy, so the path is correct. It is simply not worth
running.

### Why acceptance above break-even still loses

`breakEvenTokensPerRound = 2.5` was calibrated in P10 for the GPU-resident regime. It does
not transfer here, for a reason that is specific and worth stating: **a batched verify over a
sparse MoE does not amortize the way a dense one does.** Each of the 8 verified rows routes
to its OWN top-8 of 256 experts, so an 8-row verify touches far more expert weight than one
decode step does — approaching 8× the expert traffic rather than the near-free extra rows a
dense model gives you. Laguna is A3B: a single decode step activates ~3B parameters, and the
batching that makes speculation pay elsewhere is exactly what MoE routing defeats.

Working backwards from the measurement, the real break-even on this target is ≈ 3.9
tok/round (3.20 / 0.82), not 2.5.

### The guard would NOT have caught this

This is the part that matters beyond Laguna. P10's runtime acceptance guard disables
drafting when tok/round falls below `breakEvenTokensPerRound`. At 3.20 observed against a
2.5 constant, **the guard would have happily kept a configuration running that loses 18%**.
The guard is sound in shape and wrong in calibration for a CPU MoE target — its constant
encodes an economics that this regime does not share.

So the honest conclusion is threefold:

1. **The pairing works.** Loads, drafts, accepts at 3.20 tok/round, lossless.
2. **It must not be wired into `serve --drafter` for CPU MoE targets**, which is why that
   wiring is deliberately absent rather than merely unfinished.
3. **`breakEvenTokensPerRound` should be measured per (target, backend), not a constant** —
   or the guard should compare wall-clock directly rather than inferring from a rate. Filed
   here rather than fixed, because changing a shipped guard's semantics on the strength of
   one target's measurement would be the same over-generalization that put a GPU-calibrated
   constant in the CPU path to begin with.

The CPU BlockSpec itself is still worth having: it is the first non-CUDA implementation of
those interfaces, it is gated lossless (6.67 tok/round on Qwen3-4B, token-identical), and it
makes block drafting available to CPU-only families whose economics may differ — a DENSE
CPU-only target is precisely the case where batched verify should amortize.
