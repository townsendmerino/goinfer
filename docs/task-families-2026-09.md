# Task: four families, 2026-09 (F1–F4)

> **Status: DONE, 2026-09-06.** All four items landed in order (F1 → F2 → F3 → F4), each committed
> and pushed independently. Summary below; read each section's own status line for the detail.
>
> - **F1 `qwen3_moe`** — new registry key, T1 tiny-golden (cosine 0.9999999999999462, argmax
>   exact), GGUF support, resident-admitted cuda/metal/webgpu. T3 real-checkpoint owed (Linux box).
> - **F2 Nemotron 3.5 Lightning** — confirmed NOT a new family: config-identical to the already-T3'd
>   Nemotron 3 Nano in every architecturally meaningful field. No registry change; a real-checkpoint
>   gate + pin script + asset were added, execution owed (Linux box, bf16 ~60GB).
> - **F3 `granite`** (dense) — new registry key, reuses `llamaTensorSchema` verbatim. Of Granite's
>   four scalar multipliers: `embedding_multiplier` → `EmbedScale`, `attention_multiplier` →
>   `AttnScale`, `logits_scaling` → `LogitScale` (all three already-generic fields, all exercised
>   non-trivially by the T1 fixture below); `residual_multiplier` has no generic hook and is
>   HARD-REJECTED unless 1.0 (the identity value every released 4.2 size ships — see the F3 section
>   below for the check). T1 tiny-golden with the three non-trivial multipliers (cosine
>   0.999999999999987, argmax exact), GGUF support, resident-admitted cuda/metal/webgpu. T3
>   real-checkpoint owed (this Mac or Linux box; 8B is the sized target).
> - **F4 Ling-3.0-tiny (`bailing_hybrid`)** — Phase 0 only, by design: no stop condition fired, but
>   the brief scoped this item to a synthetic rehearsal, not a shipped family. MLA + MoE are pure
>   `deepseekArchitecture` composition; KDA (Kimi Delta Attention) is a genuinely new primitive — a
>   per-channel-gated delta rule, one row of the existing Gated DeltaNet's decay generalized from a
>   scalar to a vector — proven against `fla-org/flash-linear-attention`'s real reference at cosine
>   1.00000000 (maxAbsDiff 2.98e-08, float32 rounding). No registry key, no CHANGELOG entry. Filed
>   explicitly as the KDA rehearsal for Kimi K3's own eventual bring-up.
> - **Bench rows**: none this pass for any of the four — every family's T3 (the gate a peer-bench
>   row is conditioned on, per this doc's own rule for F1/F2/F3) is owed, and F4 shipped no loadable
>   family at all. Zero peer-bench claims were made, which is the correct state given that.

## F1 · qwen3_moe (Qwen3-30B-A3B / Qwen3-Coder-30B-A3B-Instruct) — DONE at T1 (mac, 2026-09-05/06)

**The hole this closes:** `decoder/registry.go` had `qwen3` (dense), `qwen2_moe`, `qwen3_5_moe` and
`qwen3_next`, but no `qwen3_moe` — Qwen3-30B-A3B and Qwen3-Coder-30B-A3B-Instruct, both released
under that exact `model_type`, had no adapter at all.

### Phase 0 (config-verified against the real releases, not a model card)

Fetched both real `config.json` files directly from the hub:

- **`Qwen/Qwen3-30B-A3B`**: `architectures: ["Qwen3MoeForCausalLM"]`, `model_type: "qwen3_moe"`,
  `head_dim: 128`, `hidden_size: 2048`, `num_attention_heads: 32`, `num_key_value_heads: 4`,
  `num_hidden_layers: 48`, `num_experts: 128`, `num_experts_per_tok: 8`,
  `moe_intermediate_size: 768`, `norm_topk_prob: true`, `decoder_sparse_step: 1`,
  `mlp_only_layers: []`, `rope_theta: 1000000.0`, `attention_bias: false`, `tie_word_embeddings:
  false`. **No `shared_expert_intermediate_size` field at all.**
- **`Qwen/Qwen3-Coder-30B-A3B-Instruct`**: byte-identical on every field above except
  `max_position_embeddings` (262144 vs 40960) and `rope_theta` (1e7 vs 1e6). Confirms the scoping
  assumption directly: **the Coder variant is the same registry key**, a config delta only.

This is exactly the hypothesized composition: qwen3 dense's attention (per-head QK-norm, GQA, no
q/k/v bias, single-base RoPE) with `qwen2_moe`'s router shape (top-k of `num_experts` at
`moe_intermediate_size`, `norm_topk_prob`) — **minus the shared expert**, confirmed by the absent
config field rather than assumed from the "no shared expert" phrasing in the original scoping note.

### GGUF — header and tensor names/dims verified against a real file; VALUES are not (T3 item)

Fetched the first 30 MB of `unsloth/Qwen3-30B-A3B-GGUF`'s `Q2_K.gguf` via HTTP Range (well under
the ~12 GB full file — GGUF metadata + tensor-info lives at the front) and parsed it with goinfer's
own `aikit/embed.GGUFFile` reader, the same ground-truth method the Nemotron 3 Nano T3 handoff used
— but be precise about what that method proves here: it read `general.architecture`, the metadata
keys, and every tensor's NAME and DIMS. It did **not** dequantize a single tensor or run a forward
pass, so it cannot rule out the exact failure mode `docs/completed/queue-correctness.md` G4 already
named for this shape of loader: "a fused expert stack read with the wrong stride gives finite
values, correct shapes and confident nonsense." That check — hand-dequantizing a real
`ffn_*_exps` tensor's expert slices, or a real forward — is not done and belongs to T3, not this
pass.

What WAS confirmed this way:

- `general.architecture` is literally `"qwen3moe"`.
- Metadata is the plain, un-exotic `{arch}.attention.head_count/head_count_kv/key_length/
  layer_norm_rms_epsilon`, `{arch}.expert_count/expert_used_count/expert_feed_forward_length`,
  `{arch}.rope.freq_base` — **no sliding-window or YaRN keys at all** (unlike Mellum's per-layer-type
  split; qwen3_moe is single global RoPE).
- Tensor set and dims: `blk.N.attn_{q,k,v,output}`, `attn_{q,k}_norm`, `ffn_norm`, `ffn_gate_inp`,
  `ffn_{gate,up,down}_exps` (dims `[768 2048 128]` / `[2048 768 128]`, matching
  `moe_intermediate_size`/`hidden_size`/`expert_count`) — **no `ffn_*_shexp` tensors**, confirming
  the no-shared-expert finding independently on the GGUF side, at the name/shape level.

Because both findings are "this is exactly the generic shape, just missing the shared-expert
piece," **zero new code was needed in the generic GGUF `loadLayer` path** (`decoder/gguf.go`) — it
already gates QK-norm on `arch.QKNorm`, the router+experts on `arch.MoE != nil`, and the shared
expert on `arch.MoE.SharedIntermediateDim > 0`. Only a per-family `Config` builder
(`ggufQwen3MoeConfig`) and a `case "qwen3moe":` dispatch line were needed. That reasoning is sound
independent of the T3 gap above — it follows from the generic loader's existing gating logic, not
from this file's tensor values — but the T3 gap is the honest place a stride/layout bug would
actually be caught, and isn't closed yet.

### Adapter decision: pure composition, no new primitive, no new forward path

- `decoder/registry.go`: `qwen3MoeArchitecture` — qwen3's `Architecture` literal (`QKNorm: true`,
  `AttnScale: 1/√head_dim`, single-base RoPE, `NormPlacement: NormPre2`) with a `MoEConfig` shaped
  like `qwen2_moe`'s but with `SharedIntermediateDim` left at its zero value.
- `decoder/config.go`: `validateQwen3Moe` — qwen3's dense checks plus MoE bounds, no
  `shared_expert_intermediate_size` requirement.
- `decoder/weights.go`: `qwen3MoeTensorSchema` — qwen3's attention tensor names + `qwen2_moe`'s
  router/expert names, with every `Shared*` field left empty.
- `decoder/gguf.go`: `ggufQwen3MoeConfig` + one dispatch line, per the GGUF section above.

**Because qwen3_moe is a standard uniform-layer family** (not added to `decoder/arch.go`'s
`ownForwards` table), every dispatch surface named in the task brief answers correctly **for
free**, the same way `qwen2_moe`/`mixtral`/`mellum` already do:

- `canBatchN`/`specRollbackSafe`/`hasRecurrentState` (`decoder/forwardn.go`) all derive from
  `ownForward()`, which qwen3_moe never enters — confirmed by inspection, not a new test. (One of
  `spec_adaptive.go`'s own comments literally uses "Qwen3-30B-A3B" as its worked example of a
  resident MoE with `specRollbackSafe` true — this family was already an implicit assumption in
  that code before it had an adapter.)
- `decoder/features.go`'s `archFeatureProfile`/`FeatMoEGatedShared` derivation is generic on
  `arch.MoE.SharedIntermediateDim > 0`, so the absent shared expert falls out correctly with no new
  branch.
- `canSerialize` (`decoder/serialize.go`) and the CUDA `nonBatchableKind` (`cuda/prefill.go`) are
  both keyed on architecture SHAPE (MoE/MLA/Mamba presence), not a per-family name list, so neither
  needed an entry.

### Chokepoints exercised directly (not just via the family's own validator)

Registering the family surfaced three registry-wide tests that fail until a new `model_type` is
classified everywhere required — exactly the "walk the registry" guard the brief asked for,
already built into this repo rather than new:

1. `TestResidentAdmission_registryCovered` / `archFeatureProfile` (`decoder/features_test.go`) —
   classified as `{FeatMoE, FeatQKNorm}`, the union of `qwen3`'s and `mixtral`'s profiles.
2. `TestResidentAdmission_matrix` / `admissionGolden` — hand-verified `{cuda, metal, webgpu}`: this
   is a strict SUBSET of what `qwen2_moe` (`{FeatMoE, FeatMoEGatedShared}`) and `qwen3`
   (`{FeatQKNorm}`) each already require and are already admitted on, on all three backends, so no
   new kernel work is implied by the admission.
3. `TestParityManifest_fresh` / `TestCapabilityMatrix_CoverageComplete` — a manifest row and a
   `representativeConfig`/`familyDoc` entry are both required before CI is green; both added.

`resolveArchitecture` → `validateResolved()` runs for `qwen3_moe` like every family (not a new
test — the existing generic chokepoint test already covers every registered `model_type`).

### T1 — tiny-golden, DONE

`scripts/pin_qwen3moe_tiny.py` (transformers 5.15.0, `Qwen3MoeForCausalLM`, 4 layers / 8 experts /
top-2 / head_dim 8): `TestQwen3Moe_forwardParity` — **argmax exact, cosine 0.9999999999999462.**
Exercises QK-norm composed with a no-shared-expert routed MoE together for the first time.

**One real, if minor, finding along the way:** transformers 5.15.0's
`Qwen3MoeConfig.save_pretrained` writes the field as `num_local_experts`, but the real released
`Qwen/Qwen3-30B-A3B` config.json (fetched directly, not from a save) uses `num_experts`. goinfer's
adapter reads `num_experts` to match the real release, so the pin script rewrites the freshly-saved
config's field name post-save — the same class of transformers-serialization-drift-vs-release
mismatch `docs/completed/queue-correctness.md`'s G4 (`layers_block_type` canonicalization) and G5
(fused `in_proj_qkvz`) already catalogued, caught here before it could cause a silent config-parse
failure on anyone else's freshly-saved fixture.

`parity_manifest.json` row: `status: experimental`, `method: tiny-golden`, `argmax_pct: 100.0`,
`cosine_min/mean: 1.0`. Added to `cmd/gate/parity.go`'s required `parityGates` list
(`{"qwen3moe", "TestQwen3Moe_forwardParity"}`) so a regression here fails the sweep like every
other family's tiny gate.

### What was deliberately not done

- **T3 real-checkpoint parity — infra built, not yet run.** Qwen3-30B-A3B bf16 is ~61 GB; the
  doc's own order-of-operations ("if F1's T3 cannot run, ship at T1 and move on") applied at first
  landing. `scripts/pin_qwen3moe_real.py` + `decoder/qwen3moe_real_test.go` (tag `realckpt`,
  `TestQwen3MoeReal_oracle`, `GOINFER_QWEN3MOE_HF` asset registered) now exist, mirroring
  `decoder/nemotron3nano_real_test.go`'s shape — starting quant `int8` (weights) / f32
  (activations), not `int8int8`, per Nano's own measured router-flip cliff at similar sparsity
  (G4). Added to both `parityRealckptGates` and `emitGates` (unlike F2's Lightning gate, this one
  has no prior real-checkpoint evidence in the manifest to protect, so a passing sweep may
  legitimately promote the row). Row stays `experimental` until the run actually happens.
- **One decode benchmark row** (the brief's "one cell each" bench requirement) is gated on T3 —
  measuring a resident quant path before the numerics are validated against a real oracle would be
  exactly the kind of ungated claim `docs/benchmarks.md`'s Methodology section exists to prevent.
- **GGUF real-file round-trip test — this is a T3 item, not a T1 gap.** The header, metadata, and
  tensor names/dims were verified against a real file (above); the tensor VALUES were not — no
  dequantization, no forward pass. A full tiny-GGUF fixture + parity gate (mirroring
  `TestGGUF_qwen3_parity`) is not built, and neither is the hand-dequant spot-check G4 used for
  Nemotron 3 Nano's fused expert tensors. The fused-expert stride/layout is exactly where a GGUF
  MoE loader goes wrong per that precedent, so this is named as an open correctness question, not
  a formality.
- **CUDA/Metal/WebGPU resident kernel work.** None is needed — see the admission section above —
  but no resident-parity test was run on real hardware for this family specifically; it rides the
  same generic MoE dispatch `mixtral`/`qwen2_moe` already exercise on real backends elsewhere.

## F2 · Nemotron 3.5 Lightning 30B-A3B — Phase 0 DONE: same family, not a new one; T3 code ready, run pending

**The question this section answers, per the brief's own framing ("probably a parity pin, not a
family"): is it?** Yes — confirmed, not assumed.

### Phase 0 (config-verified against both real releases, field-by-field, not from the brief's own transcription)

Fetched `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`'s real `config.json` directly (the
brief's own quoted fields checked out, but transcriptions get re-verified here the same as every
other family in this doc) and diffed it programmatically, field-by-field, against
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`'s real `config.json` — the checkpoint `nemotron_h`'s
existing T3 (`docs/completed/queue-correctness.md` G4) already validated.

**Every architecturally meaningful field is identical**, not merely similar: `hidden_size` 2688,
`num_attention_heads` 32, `num_key_value_heads` 2, `head_dim` 128, `mamba_num_heads` 64,
`mamba_head_dim` 64, `n_groups` 8, `ssm_state_size` 128, `n_routed_experts` 128,
`num_experts_per_tok` 6, `moe_shared_expert_intermediate_size` 3712, `routed_scaling_factor` 2.5,
`n_group`/`topk_group` both 1 (degenerate, plain top-k — same as Nano), `rope_theta` 10000,
`partial_rotary_factor` 1.0 (present in BOTH configs, and — checked directly against mainline
transformers' `modeling_nemotron_h.py`, not assumed from Nano's T3 alone — genuinely unused by the
reference forward; Nemotron-H's attention is NoPE in practice regardless of these fields, which is
why `nemotronhArchitecture`'s `Architecture` literal hardcodes `RotaryDim: 0` unconditionally).

**Every one of those matching values is READ FROM CONFIG, not hardcoded** — which is the actual
claim "no registry change" rests on, not just that Lightning happens to match Nano. Named against
the Go `Config` field each maps to, all already wired by the existing `nemotronhArchitecture` /
`MoEConfig` construction:

| config.json field | Go `Config` field | consumed as |
|---|---|---|
| `n_routed_experts` | `NRoutedExperts` | `MoEConfig.NumExperts` |
| `num_experts_per_tok` | `NumExpertsPerTok` | `MoEConfig.TopK` |
| `routed_scaling_factor` | `RoutedScalingFactor` | `MoEConfig.RoutedScale` |
| `n_groups` (mamba) | `NGroups` | `nemotronParams.NGroups` (Mamba-2 group count — a DIFFERENT
  field from the MoE router's own `n_group`/`topk_group`, which both configs set to 1) |
| `moe_shared_expert_intermediate_size` | `MoeSharedExpertIntermediateSize` | `MoEConfig.SharedIntermediateDim` |

None of these is a value the adapter assumes or defaults to — if Lightning's release had shipped
different numbers here, the loader would have read and used THOSE, not Nano's. The identity is a
fact about the two checkpoints, not about what the code does with whatever it's given.

**Decoded and diffed the actual layer PATTERN, not just its summary counts.** Nano ships the
compact `hybrid_override_pattern` string (`"MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"`,
52 chars); Lightning ships the expanded `layers_block_type` array directly. Decoded Nano's string
with the same M/E/*/− alphabet `decoder/config.go`'s `normalizeNemotronBlocks` uses and compared
position-by-position against Lightning's array: **byte-identical sequence**, 23 mamba / 23 moe / 6
attention, same order. `normalizeNemotronBlocks` already no-ops when `layers_block_type` is
present (its very first line), so Lightning's array-shaped release needs no loader change either.

**The only field-level differences are vestigial or presentational, each checked against source
rather than assumed inert:**

- `moe_latent_size: null` on both — checked against `modeling_nemotron_h.py` directly: non-null
  would add an `fc1_latent_proj` bottleneck before the expert FFN; null (both checkpoints) means
  that branch never executes. No LatentMoE on either family member.
- `moe_shared_expert_overlap: true` (Lightning only) — grepped mainline transformers'
  `modeling_nemotron_h.py`: **zero occurrences**. Not read by the HF reference forward at all — an
  inference-engine (vLLM/TensorRT-LLM-class) scheduling hint to overlap the shared-expert compute
  with routed-expert dispatch, with no effect on the computed result.
- `num_nextn_predict_layers: 1` / `mtp_layers_block_type` (Lightning only) — also zero occurrences
  in `modeling_nemotron_h.py`; the MTP head isn't implemented in the mainline forward class at all.
  **This is the one field pair the Nano-shaped tiny fixture cannot exercise**, since it never had
  an MTP head to begin with — checked what actually happens, not assumed benign:
  `decoder/config.go`'s `Config` struct has **no field mapped to either JSON key**, so Go's
  `encoding/json` silently drops both on unmarshal (unknown-field behavior, not a parsing error).
  `cfg.NumLayers` is derived purely as `len(cfg.LayersBlockType)` (52) inside
  `nemotronhArchitecture` — a count that never reads `num_nextn_predict_layers`, so its presence
  or value cannot inflate or shrink the layer count the loader iterates. That reasoning is
  code-verified (both fields are genuinely absent from `Config`, checked by grep), but **it is
  reasoned from code, not confirmed against the real checkpoint's tensor index** — whether the
  actual safetensors shards carry a 53rd layer's worth of MTP-head tensors under some naming the
  main load loop could accidentally touch is unknown until the checkpoint is actually inspected or
  loaded. If T3 fails on a tensor-count or shape mismatch, this is the first place to look.

### Adapter decision: no registry change

Because every field above already round-trips through `nemotronhArchitecture` /
`validateNemotron` / `nemotronTensorSchema` unmodified, **this is not a new registry entry** — the
manifest's `nemotron_h` row (keyed by registry `model_type`, per G4's own precedent: "recorded on
the existing nemotron_h manifest row rather than a sibling") already covers this checkpoint's
architecture. The existing `nemotron3nano-tiny` T1 fixture already exercises this exact geometry
(identical shape), so no new tiny fixture was built — doing so would duplicate a fixture that
tests nothing new, which the repo's own anti-duplication instinct argues against.

### What T3 here actually checks (and what's ready, vs run)

A tiny fixture proves the *shape*; it cannot catch a wrong tensor name, a transposed expert stack,
or a router-bias read from the wrong key on **these specific trained weights** — every one of
which produces correct shapes and plausible values regardless of the training run behind them.
That is the one thing Phase 0's config-identity finding cannot establish, so it is still worth a
real-checkpoint gate.

**Built and merged, not yet run:**

- `scripts/pin_nemotron35lightning_real.py` — `pin_nemotron3nano_real.py` with the checkpoint path
  swapped; same prompt, same greedy-continuation-6 shape.
- `decoder/nemotron35lightning_real_test.go` (tag `realckpt`), `TestNemotron35LightningReal_oracle`
  — reuses the same `realLogitOracleQuant` helper Nano's own gate uses, same `int8`
  weights/f32-activations quant (Nano's own measured finding: int8 *activations* cliffed this
  family's router from cosine 0.997668 to 0.978086 at 6-of-128 sparsity — expected to transfer
  given the identical router shape, but that transfer is itself unverified until this gate runs).
  **Deliberately left out of `cmd/gate/parity.go`'s `emitGates`** (added to `parityRealckptGates`
  instead): the manifest's `nemotron_h` row is already `validated` from Nano specifically, and a
  routine sweep re-emitting this gate's `PARITY_ROW` would silently overwrite Nano's own
  already-recorded numbers with Lightning's on every `EMIT_MANIFEST=1` sweep, discarding real
  evidence for no gain. `TestNemotron3NanoReal_oracle` itself sets the precedent — it isn't in
  `emitGates` either.
- Asset registered: `GOINFER_NEMOTRON35LIGHTNING_HF` → `$MODELS/nemotron35lightning-30b-bf16`
  (`testdata/assets.json`), same `dir` shape as Nano's own entry.

**Not run this pass:** the checkpoint is bf16 ~60 GB+, well past this Mac's 29 GB free disk / 16 GB
RAM. The Linux box (`nobara-pc`, 229 GB free / 62 GB RAM) is the intended target per the brief's
own "30B-A3B class on the Linux box" rule, matching where Nano's own T3 ran. **No CHANGELOG entry
for F2**: nothing loadable changed (no registry key, no adapter code, no tensor schema) — only
test/asset infrastructure, which is not a capability change.

## F3 · Granite 4.2 (dense, 3B/8B/30B) — DONE at T1 (mac, 2026-09-06)

### Phase 0 (all three released sizes fetched and diffed)

`ibm-granite/granite-4.2-{3b,8b,30b}`: `model_type: "granite"` (`GraniteForCausalLM` — distinct
from `granitemoehybrid`'s `GraniteMoeHybridForCausalLM`), no key registered for it. All three
carry `attention_bias: false`, `rms_norm_eps: 1e-05`, both a nested `rope_parameters` object and a
redundant flat `rope_theta` (transformers 4.57.1 writes both), `tie_word_embeddings: false` on
**every size** (the brief's own caution — "confirm tie_word_embeddings per size" — checked and
found NOT to vary here), and Granite's four scalars: `embedding_multiplier`, `attention_multiplier`
(the only one that varies meaningfully: 0.0078125 / 0.015625 / 0.0078125 for 8b/3b/30b — NOT
1/√head_dim, e.g. 8b's head_dim 128 would give 0.0884), `residual_multiplier`, `logits_scaling`.

**`embedding_multiplier`, `residual_multiplier` and `logits_scaling` are 1.0 on all three sizes.**
Only `attention_multiplier` deviates from its HF default. This is a real, checked-not-assumed
finding with a direct consequence below.

**Confirmed the tensor names are byte-identical to llama's**, not by reading `modeling_granite.py`
and eyeballing similarity but by instantiating a real `GraniteForCausalLM` (transformers 5.15.0)
and reading its `state_dict()` keys directly: `model.embed_tokens.weight`,
`model.layers.N.self_attn.{q,k,v,o}_proj.weight`, `model.layers.N.mlp.{gate,up,down}_proj.weight`,
`model.layers.N.{input,post_attention}_layernorm.weight`, `model.norm.weight`, `lm_head.weight` —
no bias tensors, no QK-norm tensors. `llamaTensorSchema` is reused verbatim.

### GGUF — verified against a real file

Fetched the first 15 MB of `bartowski/granite-4.2-3b-GGUF`'s `Q2_K.gguf` via HTTP Range and parsed
it with `aikit/embed.GGUFFile` directly: `general.architecture` is `"granite"` (distinct from the
hybrid's `"granitehybrid"`), and llama.cpp bakes the four scalars directly into metadata —
`granite.attention.scale` (the **resolved** attention_multiplier, 0.015625 for the 3B, matching
its released config exactly — no 1/√d default-resolution needed on the GGUF side),
`granite.embedding_scale` / `granite.logit_scale` / `granite.residual_scale` (all 1, confirming the
Phase 0 finding independently). Tensor set: `blk.N.attn_{q,k,v,output}`, `ffn_{gate,up,down}`,
`attn_norm`, `ffn_norm`, `token_embd`, `output`, `output_norm` — exactly llama's, no bias, no
QK-norm tensors.

### Adapter decision: pure composition on three already-generic scalar fields, one hard guard

`EmbedScale` (embedding_multiplier), `AttnScale` (attention_multiplier, in place of the
1/√head_dim default), and `LogitScale` (logits_scaling) are **already generic** on `Architecture` —
each is just "multiply by this scalar," applied at the same call sites regardless of which family
supplies the value or how that value was derived (Gemma's √hidden `EmbedScale` and Cohere's
`LogitScale` already prove the mechanism is family-agnostic). **`residual_multiplier` is the one
exception**: `granitemoehybrid`'s own dedicated forward (`runLayersGranite`) applies it via
`graniteParams.ResidMul`, and the generic uniform-layer path this family rides has no such hook.
Since every released 4.2 size ships it at the identity value 1.0, `validateGraniteDense` **rejects
any other value** rather than silently dropping it — the same discipline `validateLlama` already
applies to scaled RoPE, and cheap today (it costs nothing on any currently-published checkpoint).

`decoder/registry.go`: `graniteDenseArchitecture` (reuses `ropeBaseFlatOrNested`, same as the
hybrid). `decoder/config.go`: `validateGraniteDense`. `decoder/gguf.go`: `ggufGraniteDenseConfig` +
one dispatch line. No new tensor schema (reuses `llamaTensorSchema`).

### Chokepoints and resident admission

Registering `"granite"` surfaced the same three registry-wide gates F1 did:
`TestResidentAdmission_registryCovered`/`archFeatureProfile`, `TestParityManifest_fresh` (manifest
row), `TestCapabilityMatrix_CoverageComplete` (`representativeConfig`/`familyDoc`) — all satisfied.

**`archFeatureProfile["granite"]` is `{}` — empty on purpose, not by omission.** Checked against
`FeatEmbedScale`'s and `FeatLogitScale`'s own derivation (`add(a.EmbedScale > 1, ...)`,
`add(a.LogitScale != 0 && a.LogitScale != 1, ...)`): since every released size's
`embedding_multiplier`/`logits_scaling` are exactly 1.0, **neither feature flag ever fires for a
real checkpoint of this family** — the derivation and the Phase 0 finding agree independently.
`attention_multiplier` isn't gated by any `ResidentFeature` at all; it's baked into `AttnScale`,
which every backend reads generically regardless of value. So `admissionGolden["granite"]` is
`{cuda, metal, webgpu}` — the same set `llama` gets, for the same reason: nothing this family needs
is missing anywhere.

### T1 — tiny-golden, DONE

`scripts/pin_granite_dense_tiny.py` (transformers 5.15.0, `GraniteForCausalLM`, 4 layers, **all
three tunable multipliers set to non-trivial values** — 12.0 / 0.5 / 6.0 for embedding/attention/
logits, residual pinned to 1.0 as the adapter requires): `TestGraniteDense_forwardParity` —
**argmax exact, cosine 0.999999999999987**, plus direct assertions that the resolved
`EmbedScale`/`AttnScale`/`LogitScale` equal the fixture's non-default values (not just that the
final logits happen to match, which a dropped-then-cancelled multiplier could also produce).

`parity_manifest.json` row: `status: experimental`, `method: tiny-golden`. Added to
`cmd/gate/parity.go`'s `parityGates` (`{"granite-dense", "TestGraniteDense_forwardParity"}`) and
`awaitingFirstConfirmation` (learned from F1's CI break — see below).

### What was deliberately not done

- **T3 real-checkpoint parity.** Same order-of-operations as F1: ship at T1, move on. Granite 4.2
  8B (bf16 ~16 GB) is the natural T3 target for this Mac per the brief's own sizing note, but
  wasn't run this pass — no CHANGELOG claim is made about real-checkpoint validation.
- **30B real-checkpoint / load check.** Explicitly out of scope per the brief ("30B is a load
  check only") and not attempted.
- **GGUF real-file round-trip test.** Same posture as F1: the header, metadata, and tensor
  names/dims were verified against a real file; the tensor values were not (no dequant, no
  forward). No committed tiny-GGUF fixture exercises this in CI.

### Process note carried over from F1

F1's push broke CI: `TestParity_everyRequiredGateIsConfirmed` (audit G-04's ledger guard) correctly
flagged `TestQwen3Moe_forwardParity` as a required gate with no ledger entry. Fixed same-day by
adding it, and `TestNemotron35LightningReal_oracle`, to `cmd/gate/parity.go`'s
`awaitingFirstConfirmation`. F3's own new gate was registered there in the same commit that added
it, closing the loop this once cost a red `main` to discover.

## F4 · Ling-3.0-tiny (inclusionAI, model_type "bailing_hybrid") — Phase 0 DONE, KDA rehearsal DONE, NO registry work (mac, 2026-09-06)

**Verdict up front, per the brief's own framing: does not STOP, but does not become a registered
family this pass either.** No stop condition fires (weights are posted, real, and BF16-readable;
the checkpoint composes from exactly three primitives — nothing beyond KDA+MLA+MoE), so Phase 0
proceeds; per the brief's explicit scope for this item ("proceed ONLY to a synthetic-tiny bring-up
… no real-checkpoint work in this pass"), what follows is that rehearsal, not a shipped adapter.

### Phase 0

**(a) Weights are posted, not announced-only** — the specific trap `docs/post-v1.0-models.md`
already recorded for Ling 3.0's July announcement. Confirmed via the HF API, not a model card:
32 real safetensors shards (`model-00000-of-00032.safetensors` … `-00031-of-00032`) plus
`model.safetensors.index.json`, `torch_dtype: "bfloat16"` — readable today (no fp8/mxfp4 blocker,
the thing that stopped DeepSeek-V4-Flash and gates Kimi K3). MIT license. `custom_code`
(`trust_remote_code` needed): `configuration_bailing_moe_v3.py` + `modeling_bailing_moe_v3.py`
ship in the repo.

**(b) Exact mixer inventory**, from the real `config.json` and the real `modeling_bailing_moe_v3.py`
(fetched directly, not summarized from a description): `model_type: "bailing_hybrid"`,
`architectures: ["BailingMoeV3ForCausalLM"]`.

- **MLA**: `q_lora_rank` 256, `kv_lora_rank` 512, `qk_nope_head_dim` 128, `qk_rope_head_dim` 64,
  `qk_head_dim` 192 (= nope+rope), `v_head_dim` 128, `rope_interleave: true` — the SAME field
  names and shapes `deepseekArchitecture` (`decoder/registry.go`) already reads for DeepSeek-V2/V3.
  One real delta: `gated_attention_proj_granularity_type: "head_wise"` adds a per-head output gate
  (`g_proj: Linear(hidden, num_heads)`) that DeepSeek's own MLA doesn't have — structurally the
  same primitive Laguna's `FeatAttnOutputGate` already ships (`docs/task-laguna.md`), not a new one.
- **MoE router**: `topk_method: "noaux_tc"`, `scoring_func`/`score_function: "sigmoid"`,
  `routed_scaling_factor: 2.5`, `n_group` 8 / `topk_group` 4, `num_experts` 128,
  `num_experts_per_tok` 8, `num_shared_experts` 1, `moe_shared_expert_intermediate_size` 512,
  `first_k_dense_replace: 1` (one dense-prefix layer) — byte-for-byte DeepSeek-V3's `noaux_tc`
  shape, already implemented by `routeExperts`/`moeMLP`. `modeling_bailing_moe_v3.py`'s
  `BailingMoeV3Gate.forward` confirms the shared expert is added UNGATED
  (`y = y + self.shared_experts(identity)`), the same convention DeepSeek/GLM already use.
- **KDA:MLA ratio**: `layer_group_size: 4`, `num_hidden_layers: 24`. `BailingMoeV3DecoderLayer`'s
  actual layer-type decision (quoted from source): every layer where
  `(layer_idx+1) % layer_group_size == 0` is MLA ("attention"); every other layer is KDA
  ("linear_attention") — a **repeating 3-KDA-then-1-MLA pattern**, confirming the brief's own
  "3:1 alternating stacking" description exactly: 18 KDA / 6 MLA over 24 layers.
- **KDA itself — the one genuinely new primitive, verified against fla-org/flash-linear-attention's
  actual source** (`fla/ops/kda/{naive,gate}.py`, MIT), not the HF modeling file's paraphrase
  (which only calls the opaque Triton kernels `chunk_kda`/`fused_recurrent_kda`). Structurally
  **identical to the Gated DeltaNet this repo already ships** for `qwen3_5_moe`
  (`gatedDeltaNetStep`, `decoder/deltanet.go`): a q/k/v projection through a depthwise short conv
  (`short_conv_kernel_size: 4`), a beta write-gate (`sigmoid(b_proj(h))`), an outer-product
  delta-rule state update, q/k L2-norm-in-kernel, a gated RMSNorm before `o_proj`
  (`FusedRMSNormGated(..., activation='sigmoid')` — Gated DeltaNet's own gated norm uses SiLU, a
  real but small delta). **The one load-bearing difference**: Gated DeltaNet's decay is ONE SCALAR
  per head, multiplying the WHOLE `[head_k_dim, head_v_dim]` state block
  (`gatedDeltaNetStep`'s `gt := exp(negExpA[headV]*softplus(...))`, applied uniformly to every
  element of `S`); KDA's decay is **PER-CHANNEL** — shape `[H, head_k_dim]`, one independent value
  per ROW of the state matrix (`naive_recurrent_kda`: `S = S * g_i[..., None].exp()`, where `g_i`
  is `[H,K]` and only broadcasts over the value dimension). This is the real, sourced distinction
  between "Gated DeltaNet" and "Kimi Delta Attention" the literature describes — not a
  renaming, and not reducible to a config flag on the existing DeltaNet code.
- Ling-3.0-tiny's own config selects the SIMPLEST variant of two optional wrinkles:
  `no_kda_lora: true` (a single `f_proj`/`g_proj` linear per gate, not the LoRA'd `a_proj`/`b_proj`
  split `modeling_bailing_moe_v3.py` also supports) and `kda_safe_gate: true` with
  `kda_lower_bound: -5` (the gate saturates through `naive_kda_lowerbound_gate`:
  `g = lower_bound · sigmoid(exp(A_log) · (raw_gate + dt_bias))`, rather than the plain
  `naive_kda_gate`'s unbounded `-exp(A_log)·softplus(...)`).

**(c) Mapping onto existing primitives** — MLA + MoE ride `deepseekArchitecture` as config
composition (plus one small new output-gate primitive already shipped via Laguna); KDA needs one
genuinely new sequence-mixer, structurally a variant of the already-shipped Gated DeltaNet with a
per-channel instead of per-head-scalar decay.

**(d) dtype**: BF16 primary. Readable — no blocker.

### The rehearsal: KDA's per-channel-gated delta rule, proven against a real reference

Per the brief's scope for this item, built and verified ONLY the new piece — not a full layer, not
a registered family, no MLA, no MoE, no conv/gated-norm wrapper (those are already-shipped
primitives with their own existing proof).

- `scripts/pin_kda_rehearsal.py`: `naive_kda_lowerbound_gate` and `naive_recurrent_kda` copied
  **verbatim** (MIT license) from `fla-org/flash-linear-attention`'s actual source on GitHub —
  fetched directly, not reconstructed from a description, and not run through the full `fla`
  package (which pulls in Triton, unavailable without a CUDA toolchain on this Mac; the two
  reference functions themselves are plain PyTorch/einops). A tiny synthetic input (1 batch, 6
  timesteps, 2 heads, head_k_dim=4, head_v_dim=4) drives both functions and dumps a golden.
- `decoder/kda_rehearsal.go`: `kdaLowerBoundGate` + `kdaRecurrentStep`, **not wired to any
  registered family** — this is deliberately a standalone rehearsal, matching the brief's own
  framing that no real-checkpoint or registry work belongs in this pass.
  `TestKDARehearsal_matchesReference` (`decoder/kda_rehearsal_test.go`) checks the gate
  computation AND the recurrence output independently (so the two could not cancel out and hide
  behind one passing final comparison): **maxAbsDiff 2.98e-08, cosine 1.00000000** — float32
  rounding, not an approximation.

### Estimate for the full family (the deliverable this rehearsal earns)

- **KDA layer, full**: the rehearsal proves the core recurrence + gate exactly; remaining work is
  the wrapper this pass deliberately didn't re-prove — the depthwise short conv (LFM2/Mamba-2
  precedent exists; `short_conv_kernel_size` 4 matches LFM2's own convention), the gated RMSNorm
  with **sigmoid** activation (Gated DeltaNet's own gated norm uses SiLU — a real, small,
  config-driven delta to thread through, not a new primitive), and the **chunked/parallel** scan
  (`naive_chunk_kda`/`chunk_kda`) for prefill throughput — this rehearsal only covers the
  sequential form, matching `gatedDeltaNetStep`'s own current scope (no chunked Gated DeltaNet
  path exists in Go today either, per `decoder/deltanet_chunked.go`'s own scope — worth checking
  before assuming that work is novel to KDA). Estimate: **small-to-moderate**, closely bounded by
  Gated DeltaNet's own bring-up cost, now that the one genuinely new piece is de-risked.
- **MLA + MoE**: composition on `deepseekArchitecture`, config-mapping delta only, plus the
  Laguna-precedented output gate. Estimate: **cheap**, same class as F1/F3 this doc.
- **Registry + loader wiring, tiny fixture, T1**: same shape as every family in this doc — a new
  `model_type` key, a tensor schema (three tensor sets: MLA's, KDA's, MoE's — all individually
  precedented), `representativeConfig`/`familyDoc`/`archFeatureProfile` entries, a
  `BailingMoeV3ForCausalLM` tiny fixture via `trust_remote_code=True` (the repo ships the modeling
  code needed). Estimate: **one focused session**, similar order to F1/F3.
- **T3 real-checkpoint**: 7.9B total / 1.3B active — small enough to co-reside with an HF f32
  reference even on this Mac's 16GB, unlike every 30B-A3B-class family in this doc. The cheapest
  real-checkpoint validation of the four families touched this pass, if and when the family lands.
- **Total, rough order of magnitude**: comparable to one of F1/F3 above PLUS a Gated-DeltaNet-sized
  new-primitive cost — not a `qwen3_next`-scale undertaking, because the hardest primitive (KDA's
  core math) is now de-risked by this rehearsal rather than an open question.
- **Why this matters beyond Ling itself**: this rehearsal is explicitly filed as the dry run for
  Kimi K3 (`docs/post-v1.0-models.md`'s "Watch — Kimi K3": MLA on only 24 of 93 layers, the rest
  KDA, plus a latent-MoE wrapper and a new activation — "closer to qwen3_5_moe's DeltaNet family
  than to anything MLA-shaped"). K3's own weights are still `mxfp4-pack-quantized` in an unconfirmed
  layout and its arch should still be re-checked when they actually drop (per that doc's own
  caution) — but the one piece independent of K3's own quirks, the KDA recurrence itself, is now
  proven end-to-end against a real upstream reference rather than an open question.

### What was deliberately not done

No registry key, no tensor schema, no adapter function, no CHANGELOG entry (nothing loadable
changed), no real-checkpoint work, no chunked-scan implementation, no MoE/MLA code (both already
exist and weren't re-exercised here). This is Phase 0 plus one scoped, verified rehearsal — exactly
the brief's stated ceiling for this item.

# Batch 2 — one loader gap, three cheap keys, one real family

> **Status: IN PROGRESS.** Order G1 → G3 → G4 → G2 → G5, same independently-shippable rule as
> batch 1. Summary filled in as each item lands; read its own status line before citing it as done.

## G1 · GGUF loader for the `qwen3_5` dense hybrid (Qwen3.8-27B) — ALREADY BUILT; the real gap was tests + docs

**The brief's own framing ("this is a loader, not a family... build: loader, header-level test,
T3 parity") assumed the loader didn't exist. Checked before estimating, per this doc's own
discipline, and it does — has since 2026-08-19 (`e4bbb28`).** `ggufQwen35DenseConfig`
(`decoder/gguf_qwen35.go`) and `buildWeightsFromGGUF`'s dense branch (`decoder/gguf.go`, gated on
`arch.MoE == nil`) both exist; `decoder/qwen3_5_gguf_test.go`'s `TestQwen38GGUF_gate` already loads
a real 16.5 GB Q4_K_M checkpoint, asserts full geometry (64 layers, 5120/256/24/4/17408, the 48
linear / 16 full 3:1 split computed from `full_attention_interval`), and checks actual generation
coherence (three named Paris landmarks). It's wired into `parityRealckptGates`,
`awaitingFirstConfirmation`, and `testdata/assets.json` (`GOINFER_QWEN38_GGUF`) already.

**What was actually stale: the documentation, not the code.** `familyDocs["qwen3_5"]` /
`["qwen3_5_text"]` (`decoder/capability_matrix_test.go`) hand-annotated Loaders as `"safetensors"`
only — a real drift, now fixed and regenerated into `docs/capability-matrix.md`.

### Phase 0 — re-verified against a real file independently, not trusted from an 8-month-old commit message

Fetched the first 40 MB of `bartowski/Qwen3.8-27B-GGUF`'s `Q2_K.gguf` via HTTP Range and parsed it
with `aikit/embed.GGUFFile` directly. Every metadata key and value the loader reads matches exactly
what `TestQwen38GGUF_gate` already asserts — `qwen35.attention.head_count` 24,
`head_count_kv` 4, `key_length` 256, `block_count` 65, `embedding_length` 5120,
`feed_forward_length` 17408, `full_attention_interval` 4, `nextn_predict_layers` 1,
`rope.dimension_count` 64, plus the `ssm.*` DeltaNet geometry keys — confirming the loader has not
drifted since it was built, on a freshly re-quantized real file rather than the one the original
gate ran against. Tensor set: `blk.N.attn_qkv` (fused), `attn_gate`, `ssm_{beta,conv1d,dt.bias,a,
alpha,norm,out}` (the Gated DeltaNet set), `ffn_{gate,up,down}` (plain SwiGLU — **no router, no
expert tensors**, confirming the dense delta is exactly "the FFN" as the brief predicted).

### The one real gap: T3 GGUF-vs-safetensors parity had never been built for the DENSE sibling

`qwen35_gguf_weightdiff_test.go` (`TestQwen35GGUF_weightDiff`) already does this for the MoE
sibling (Qwen3.6-35B-A3B) — needs no HF oracle, since the safetensors loader is already Gate-1
bit-exact vs HF and stands as the reference; every transform-bearing tensor (V-head un-tile, fused
q‖gate q_proj, −exp(A_log) bake, (1+w) norm un-bake) is diffed directly, cosine floor 0.999. No
equivalent existed for the dense checkpoint. Added `decoder/qwen3_5_gguf_weightdiff_test.go`
(`TestQwen38GGUF_weightDiff`), reusing `loadQwen35Slice`/`loadQwen35GGUFSlice`/`tensorAgreement`
verbatim (both are already architecture-generic, not MoE-specific) against `GOINFER_QWEN38`
(safetensors, already registered) and `GOINFER_QWEN38_GGUF` (GGUF, already registered) — no new
asset needed. The router-comparison block in the MoE version is left as dead code for this arch on
purpose (`lr.Router.Rows() > 0` is false on both sides), so the two tests stay structurally
comparable rather than diverging into two designs for one method.

### Also checked while here (batch-1 hygiene, found incidentally)

`TestSerializeCensus_everyFixtureIsListedOrExcluded` (run as part of verifying nothing regressed)
flagged `qwen3moe-tiny` and `granite-dense-tiny` (batch 1's own fixtures) as present-but-unlisted in
`censusList`. Added both — round-trip is clean, no field drop, and the census's own bookkeeping is
now accurate for batch 1's additions too.

### What was deliberately not done

- **`TestQwen38GGUF_weightDiff` has not been run.** Needs both `GOINFER_QWEN38` (55.6 GB bf16
  safetensors) and `GOINFER_QWEN38_GGUF` (16.5 GB Q4_K_M) on the Linux box — registered in
  `awaitingFirstConfirmation`, execution owed alongside F1/F2's own owed T3 runs.
- **Peer row vs Ollama and llama.cpp.** Needs the Linux box and both peers installed; not attempted.
- **No CHANGELOG entry.** The loader itself shipped in August under its own commit
  (`e4bbb28`) and already has a CHANGELOG-worthy history; this pass's changes are a familyDoc
  correction plus new test coverage, not a new capability.

## G3 · `mistral3` / `ministral3` — Ministral 3 (3B/8B/14B) — DONE at T1 (mac, 2026-09-06); real new primitive, not a pure reuse

**The brief's own plan ("register the key; reuse the mistral adapter") turned out optimistic —
checked before estimating, and two of its own explicit Phase 0 questions found real answers that
change the scope.**

### Phase 0 (both released sizes fetched — 3B and 8B — plus the real HF modeling source)

`mistralai/Ministral-3-{3b,8b,14b}-Instruct-2512`: `architectures:
["Mistral3ForConditionalGeneration"]`, `model_type: "mistral3"` (outer VL wrapper) with a nested
`text_config.model_type: "ministral3"`. **The checkpoints ship natively FP8-quantized**
(`quantization_config.quant_method: "fp8"`) — the exact blocker that stopped DeepSeek-V4-Flash
scoping (`docs/post-v1.0-models.md`) — but a `-BF16` sibling repo exists for every size
(`mistralai/Ministral-3-8B-Instruct-2512-BF16`), so this is not a stop condition here; **target
the `-BF16` repos, not the default FP8 ones.**

Two of the brief's own Phase 0 questions resolved to "no, not what was assumed":

- **Sliding window: `sliding_window: null` on both fetched sizes** — not "every layer", not any
  layer. `mistralArchitecture` already treats `SlidingWindow<=0` as full attention (its own
  "0 ⇒ full attention" comment), so this needed no new code — but it's the opposite of the
  brief's framed caution ("verify... whether it is every layer").
- **RoPE scaling is `rope_type: "yarn"`**, not "default" — with a real, load-bearing THIRD field
  inside `rope_parameters` neither the brief nor a `mistralArchitecture` reuse anticipated:
  `llama_4_scaling_beta` (0.1 on both sizes fetched). Traced into the REAL
  `modular_ministral3.py`/`modeling_ministral3.py` (transformers 5.15.0, mainline — no
  `trust_remote_code` needed) rather than guessed: `get_llama_4_attn_scale(position_ids, beta,
  original_max_position_embeddings)` multiplies the QUERY by
  `1 + beta·ln(1 + floor(pos/origMaxPos))`, **applied AFTER RoPE, on every layer** — literally
  Llama 4's own attention-temperature-tuning formula (the function is named after it), generalized
  from Llama 4's single-branch NoPE-only use to run alongside RoPE.

Also confirmed: `mscale`/`mscale_all_dim` (both 1.0 on every released size) are DeepSeek's own
spelling of the YaRN `attention_factor`, not the generic `attention_factor` key `parseRopeScaling`
reads directly — left unhandled, its own default (`0.1·ln(16)+1 ≈ 1.277` at the real `factor: 16`)
would silently override the correct value (1.0, since `mscale == mscale_all_dim` here — same
reasoning `deepseekArchitecture`'s own comment gives for V2-Lite).

`tie_word_embeddings` moves BETWEEN the top level (8B: `false`) and nested inside `text_config`
(3B: `true`) across the two fetched sizes — a real placement inconsistency between releases. Does
not matter in practice: every family in this tree resolves tied-vs-untied from `lm_head.weight`
tensor presence at load, never from the config flag (mistralArchitecture's own comment says so
already), so `loadConfig`'s generic text_config-then-top-level flattening produces the right
`ModelType` regardless of where any other field lives, and the tie flag is unused either way.

### The tensor names, and the registry key — verified, not assumed

Instantiated a real `Ministral3ForCausalLM` (transformers 5.15.0) and read its `state_dict()`:
byte-identical to llama/mistral (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`, no bias, no
QK-norm) — `llamaTensorSchema` reused verbatim, no new tensor schema.

**The registry key is `"mistral3"`, not `"ministral3"`** — verified by reading `loadConfig`'s
generic text_config-flattening code directly (`decoder/config.go`): text_config is unmarshaled
first (setting `ModelType` to the nested `"ministral3"`), then the FULL top-level JSON is
re-unmarshaled OVER it ("so anything authoritative there... wins," per that code's own comment),
which restores `ModelType` to the outer wrapper's `"mistral3"`. A registry key of `"ministral3"`
alone would never be reached by a real released checkpoint. Registered both anyway
(`"ministral3"` as an alias to the same adapter, mirroring `gemma3`/`gemma3_text`): a plain,
unwrapped `Ministral3Config` save (the tiny fixture's own shape, and possibly some future
re-conversion) carries `"ministral3"` directly with no wrapper to flatten away.

The vision tower (`model.vision_tower.*`, `model.multi_modal_projector.*`) is ignored the same way
every other VL-wrapped family here already is: the safetensors loader is pull-based (requests
tensors by the schema's names), so it never queries those prefixes at all — no new code, same
mechanism `gemma4_unified_text`/`qwen2_5_vl` already rely on.

### The real new primitive: attn-temp, generalized from Llama 4's own-forward path to the generic one

`llama4Architecture` already has this exact formula (`attnTemp`/`floorScale`/`attnScale` on
`layerParams`, `decoder/forward_llama4.go`) — but as an EITHER/OR with RoPE, gated per-layer on
`useRope[layer]`, inside Llama 4's own dedicated forward function. Ministral 3 needs it COMBINED
with RoPE, on every layer, which that own-forward path has no branch for.

Added two generic `Architecture` fields, `AttnTempBeta`/`AttnTempOrigMaxPos` (0 ⇒ off, so every
existing family — including Llama 4 itself, which keeps its own separate mechanism — is
unaffected), and wired the same formula into BOTH generic forward paths right after their existing
RoPE step: `decoder/attention.go`'s `causalAttention` (sequential decode) and
`decoder/forwardn.go`'s batched-prefill/verify loop (which computes `pos` per-row and could not
share the sequential helper). `TestMinistral3_batchedMatchesSequential` proves the two independent
call sites agree bit-for-bit — a family whose only test drove the sequential path would never have
caught the batched copy drifting, the same class of gap `TestForwardN_matchesSequential` guards
against generically for every family's ordinary RoPE.

**Registered as `hasRecurrentState`-adjacent risk from day one, the Gated-DeltaNet lesson applied
in the other direction**: this is a NEW capability with no resident (GPU) backend implementation
at all, so a new `ResidentFeature`, `FeatAttnTemp`, was added and gated on `AttnTempBeta != 0` —
with NO backend declaring it, so `TestResidentAdmission_matrix`/`RequiredResidentFeatures`
correctly decline `mistral3` on cuda/metal/webgpu (CPU-only) rather than silently admitting a
family whose GPU path would drop the scale entirely and produce plausible-but-wrong logits at
exactly the context lengths the mechanism exists for.

### T1 — tiny-golden, DONE, deliberately exercising both new mechanisms for real

`scripts/pin_ministral3_tiny.py`: `original_max_position_embeddings=8`, prompt 12 tokens (>8, so
`floor(pos/8)>0` for the last few positions — a shorter prompt would pass with the mechanism
entirely absent, the "minimal repro hides the bug" trap this repo's own culture names explicitly);
`mscale=0.5`, `mscale_all_dim=0.8` (distinct, not both 1.0 like the real releases, so the YaRN
override is a real ratio computation, not the trivially-1.0 case a broken formula would also
pass). `TestMinistral3_forwardParity`: **argmax exact, cosine 0.9999999999999605**, plus direct
assertions that `AttnTempBeta`/`AttnTempOrigMaxPos` resolved correctly and that
`ropeScaling.mscale` equals the computed ratio (not the generic YaRN default) — not just that the
final logits happen to match, which two independently wrong values could still cancel into.
`TestMinistral3_batchedMatchesSequential` (bit-identical, 512/512 logits) as described above.

`parity_manifest.json` row: `status: experimental`, `method: tiny-golden`. Both new gates
registered in `cmd/gate/parity.go`'s `parityGates` and `awaitingFirstConfirmation` in the same
commit that adds them (F1's CI-break lesson applied without re-learning it).

### What was deliberately not done

- **GGUF loader.** Verified the real arch string and metadata against a real file (HTTP Range,
  `mistralai/Ministral-3-8B-Instruct-2512-GGUF`'s Q4_K_M): `general.architecture` is literally
  `"mistral3"`, and llama.cpp exposes the attn-temp beta directly as
  `mistral3.attention.temperature_scale` (= 0.1, confirming the finding independently) and the
  YaRN attention_factor pre-computed as `rope.scaling.yarn_log_multiplier` (no separate
  mscale/mscale_all_dim tracked in GGUF — the converter already folds the ratio). Tensor names are
  plain llama-shaped (`attn_output`, `ffn_down`, `attn_k`, no bias/QK-norm), so the generic GGUF
  `loadLayer` path likely needs zero new code, same as F1/F3's own findings — but no config
  builder or dispatch case was written this pass. Follow-up, same shape as F1's own GGUF gap.
- **T3 real-checkpoint parity.** Same order-of-operations as every family in this doc: ship at T1,
  move on. The 3B or 8B (bf16 ~16GB) is the Linux-box target per the brief's own sizing note.
- **Peer row vs Ollama.** Needs the GGUF (or the Linux box's T3) first; not attempted.
- **CHANGELOG entry**: added — this IS a new registry key with new adapter code and a new
  cross-cutting primitive, unlike F2/G1's documentation-only outcomes.

## G2 · `olmo3` + `olmo_hybrid` (Ai2, Olmo 3 7B/32B + Olmo Hybrid 7B) — BOTH DONE at T1 (mac, 2026-09-06)

**Two keys, one task, per the brief.** Olmo 3 is a pure composition of existing generic mechanisms.
Olmo Hybrid's norm placement genuinely does NOT fit the pre-existing one-scalar-per-model
`Architecture.NormPlacement` — the brief's own named stop condition ("a differing primitive beyond
composition") — so this paused for a design decision (asked directly, mid-session) rather than
building a workaround unreviewed. Decided: extend `Architecture` with a second, optional
`NormPlacement`-typed field keyed on the SAME `layerIsLinear` hook that already selects the mixer
(`NormPlacementLinear *NormPlacement`, nil for every existing family), mirroring the precedent
`RoPELocalBase`/`RoPEGlobalBase` + `layerIsGlobal` already set for Mellum's local/global RoPE
split — not a fully general per-layer `[]NormPlacement` table, which would solve a generality
nothing in this tree has needed yet. Both keys shipped once that was settled.

### `olmo3`: two real departures, both provably reusable, no new math

Fetched `allenai/Olmo-3-{7B,32B}-Think`'s real `config.json` and instantiated `Olmo3ForCausalLM`
(transformers 5.15.0, mainline, no `trust_remote_code`) directly rather than assuming from the
model card. Two findings, neither anticipated by name in the brief:

- **`NormPlacement`: no pre-norm at all.** Reading `modeling_olmo3.py`'s decoder layer directly:
  attention and the MLP each consume the RAW residual-stream input (no `input_layernorm`), and only
  the sublayer OUTPUT is normalized before the residual add
  (`post_attention_layernorm`/`post_feedforward_layernorm`). This is a fourth `NormPlacement`
  distinct from the three that already existed (`NormPre2`, `NormSandwich4` — pre AND post,
  `NormParallel`) — added `NormPostOnly` (`decoder/arch.go`) rather than reusing `NormSandwich4`
  with the pre-norm weights left nil, so a future reader can't mistake "no pre-norm tensor in the
  schema" for "pre-norm tensor happens to be identity."
- **QK-norm over the WHOLE projected vector, not per head.** `q_norm`/`k_norm` in the real
  `OlmoAttention` are sized `num_attention_heads*head_dim` / `num_key_value_heads*head_dim` — one
  RMSNorm over the concatenated multi-head projection, not `num_heads` independent per-head norms
  like every other QK-norm family here (qwen3, glm4_moe, deepseek_v3, …). The existing `rmsNorm(x,
  weight, rows, dim, eps, addOne)` kernel already computes exactly this when called with `rows=1,
  dim=nHeads*headDim` instead of `rows=nHeads, dim=headDim` — zero new math, just a different
  call-site parameterization, gated by a new `QKNormWhole bool` (`decoder/arch.go`) threaded through
  both the sequential (`decoder/attention.go`) and batched (`decoder/forwardn.go`) QK-norm call
  sites and the tensor-loading length in `decoder/weights.go`. Fixed a real latent bug while adding
  it: the existing feature-derivation line, `add(a.QKNorm, FeatQKNorm)`, would have ALSO required
  the per-head kernel for a whole-vector family; corrected to `add(a.QKNorm && !a.QKNormWhole,
  FeatQKNorm)` before it could ship as a false admission-taxonomy positive.
- **YaRN applies to `full_attention` layers only.** Olmo 3 is sliding/full 3:1
  (`sliding_window_pattern`), and the real config's `rope_scaling` (YaRN, `attention_factor: 1.1`)
  is nested per-layer-type in `rope_parameters` on a freshly-`transformers`-saved config
  (`{"full_attention": {...yarn...}, "sliding_attention": {"rope_type": "default"}}`) — confirmed
  against `configuration_olmo3.py`'s `convert_rope_params_to_dict`
  (`self.rope_parameters["full_attention"].update(rope_scaling)`, leaving `sliding_attention` at
  its plain default) rather than assumed. The REAL RELEASE ships the older flat top-level
  `rope_theta`/`rope_scaling` form instead (`olmo3Architecture` branches on
  `len(cfg.RopeParameters) > 0` to handle both). Either way this is the exact local/global RoPE
  split Mellum's own T3-validated gate already implements (`RoPELocalBase`/`RoPEGlobalBase`,
  `ropeScaling`/`ropeScalingLocal`, dispatched on `arch.layerIsGlobal`) — reused with
  `ropeScalingLocal` simply left `nil` (no scaling on sliding layers), no new mechanism.

Tensor names otherwise plain llama-shaped GQA (new `olmo3TensorSchema`,
`decoder/weights.go`, differing from `llamaTensorSchema` only in the norm-tensor name mapping
implied by `NormPostOnly`). No new forward path — rides the generic uniform-layer dispatch, so
`canBatchN`/`specRollbackSafe`/`hasRecurrentState` all answer correctly for free, same as every
other pure-composition family in this doc.

`scripts/pin_olmo3_tiny.py`: 4 layers, MHA (8 heads = 8 kv heads), `sliding_window=8`,
`layer_types=[sliding,sliding,sliding,full]`, real YaRN (`factor=4.0,
original_max_position_embeddings=8, attention_factor=1.1`), prompt of 12 tokens (>
`sliding_window=8`, so the sliding/full split and the local/global RoPE table actually differ
between layers — a shorter prompt would pass with either mechanism silently absent).
`TestOlmo3_forwardParity`: **argmax exact, cosine 0.9999999999997883**, plus direct assertions that
`NormPlacement == NormPostOnly`, `QKNorm && QKNormWhole`, and the loaded QNorm tensor length equals
`NumHeads*HeadDim` (not `HeadDim` — proves the whole-vector length wasn't silently truncated).
`archFeatureProfile["olmo3"]` needed `FeatPerLayerRoPE` in addition to the two new flags — caught by
the chokepoint test itself failing first, not assumed: the local/global RoPE tables genuinely
differ per layer here, same as every other family that has one.

`parity_manifest.json` row: `status: experimental`, `method: tiny-golden`. Gate registered in
`cmd/gate/parity.go`'s `parityGates` AND `awaitingFirstConfirmation` in the same commit (F1's
CI-break lesson applied without re-learning it).

### `olmo_hybrid`: the norm-placement mechanism, plus every other departure — all parameterizations, no new math

Fetched the real `allenai/Olmo-Hybrid-7B` config, instantiated `OlmoHybridForCausalLM`
(transformers 5.15.0), and — critically — went a step further than source-reading: fetched the
REAL published checkpoint's safetensors HEADER via HTTP Range (61KB, not the full 7.43GB file) to
check actual tensor names and shapes. That last step mattered: source-reading alone, and even a
local `save_pretrained` round-trip through this transformers version's own registered
`conversion_mapping.py`, BOTH produced tensor names/splits that turned out not to match the real
release. Neither substitute for reading the real artifact.

- **`Architecture.NormPlacementLinear` (new mechanism, decided above):** full-attention layers use
  `NormPostOnly` (`Architecture.NormPlacement`, olmo3's own scheme, confirmed identical — see
  below); DeltaNet layers use plain `NormPre2` (`NormPlacementLinear`, a pointer, keyed on
  `isLinearLayer(i)`). Threaded through both generic forward paths
  (`decoder/model.go`'s `runLayersFromEmbed`, `decoder/forwardn.go`'s batched twin) — resolved PER
  LAYER inside the loop now rather than hoisted once, so a family whose placement varies by layer
  gets the right answer without a new forward path — AND through `olmo_hybrid`'s own forward
  (`runLayersQwen35`, extended, since this family still needs its own loop for the DeltaNet
  recurrence). Also required the LOADER to become layer-kind-aware for norm tensor NAMES, not just
  placement: `tensorSchema` gained `PreAttnNormLinear`/`PostAttnNormLinear`/`PreMLPNormLinear`/
  `PostMLPNormLinear`, since the real checkpoint's `post_attention_layernorm.weight` is one tensor
  playing TWO different roles depending on layer kind (full-attention: post-attn norm;
  DeltaNet: pre-MLP norm) — confirmed directly from the real file's per-layer tensor list, not
  assumed from the reused name.
- **The Gated-DeltaNet math is qwen3_5's, field for field** (`Qwen3_5GatedDeltaNet`/
  `OlmoHybridGatedDeltaNet` share the same chunked recurrence, the same `beta = b_proj(x).sigmoid()`
  gate, the same `A_log`/`dt_bias` init, the same gated-RMSNorm-then-`out_proj` tail) — reused
  `gatedDeltaNetStep` unmodified in its MATH, parameterized for every real packing/naming
  difference the real file actually has:
  - `linear_allow_neg_eigval` (default `true`) doubles `beta` after the sigmoid, `[0,1)→[0,2)` —
    `qwen35Params.NegEigval`, a scalar multiplier, same shape as `AttnTempBeta`/`RopeMscale`.
  - `q_proj`/`k_proj`/`v_proj` are three fully separate tensors — more unfused than qwen3_5's own
    pre-concatenated `in_proj_qkv` — `SeparateQKVProj`, concatenated at load time (a plain row
    stack; each `nn.Linear`'s output-feature order is already head-major, so no per-group
    interleaving like the qwen3-next fused-tensor split needs).
  - The depthwise causal conv is ALSO three separate tensors, `q_conv1d`/`k_conv1d`/`v_conv1d` —
    `SeparateConv`. **This one needed the real file to get right**: the source shows one combined
    `self.conv1d`, and a local `save_pretrained` round-trip produced an ARBITRARY roughly-equal
    three-way split (43/43/42 rows on a 128-row test conv) that has nothing to do with q/k/v
    boundaries; the REAL release splits at the EXACT q/k/v channel boundaries instead (verified via
    the HTTP-Range-fetched header: `key_dim`/`key_dim`/`value_dim` rows precisely). Two different
    wrong answers, from two different indirect sources, before the real file settled it.
  - The output gate is named `o_norm`/`o_proj` (qwen3_5: `norm`/`out_proj`) —
    `DeltaNetNormSuffix`/`DeltaNetOutProjSuffix`. Its epsilon is HARDCODED `1e-5` in HF's source
    regardless of config ("FLA's FusedRMSNormGated uses eps=1e-5 by default"), diverging from the
    release's own `rms_norm_eps=1e-6` used everywhere else in the model — `ONormEps`. A silent-wrong
    trap if skipped: reusing the model's `NormEps` here is off by 10x on exactly the one norm most
    sensitive to it.
- **Attention layers (`full_attention`) reuse `olmo3`'s `NormPostOnly` + whole-vector QK-norm
  exactly** (`OlmoHybridAttention` literally inherits `Olmo3Attention`'s behavior). Routed through
  the SAME generic `tensorSchema`-driven loader and `causalAttention` every non-hybrid family uses
  — `qwen35Params.PlainFullAttn` skips qwen3_5's own bespoke double-width gated attention for this
  family's full-attention layers, confirming `olmo3`'s adapter generalizes rather than needing its
  own copy.
- **The real released checkpoint sets `rope_theta: null`, disabling RoPE entirely, on EVERY layer**
  — `self.rotary_emb` is only constructed `if rope_parameters.get("rope_theta") is not None`; the
  source comment says so explicitly ("Released ckpt don't use any ROPE"). A new generic
  `Architecture.NoPositionEncoding` flag names this as a FOURTH legitimate "no RoPE table" reason in
  `validateResolved`'s M-06 guard, alongside `LearnedPosEmbed`/`nemotron`/`mla` — named explicitly,
  same as those three, so a family that simply forgot to read `rope_theta` still can't look like one
  that deliberately has none.

Tensor schema: `olmoHybridTensorSchema` (`decoder/weights.go`), identical to `olmo3TensorSchema` for
full-attention layers plus the four `*Linear` norm overrides above. No MoE variant; dense SwiGLU FFN
on every layer, shared with `olmo3`/llama tensor names.

`scripts/pin_olmo_hybrid_tiny.py`: 4 layers (3 linear + 1 full, matching the release's own 3:1
ratio), MHA (`linear_num_key_heads == linear_num_value_heads`, no GVA — matching the one released
size), `rope_parameters: {"rope_theta": null}`. Bypasses `m.save_pretrained`'s own safetensors write
entirely (per the conv1d finding above) — builds `state_dict()` directly and manually re-splits
each linear layer's conv1d weight at the true q/k/v boundaries before writing with
`safetensors.torch.save_file`, reproducing the real release's actual tensor layout rather than
either wrong intermediate guess. `TestOlmoHybrid_forwardParity`: **argmax exact, cosine
0.9999999999998704**, plus direct assertions on `NormPlacement`/`NormPlacementLinear`,
`qwen35.NegEigval`, and that layer 0 (linear) loaded DeltaNet state with no plain `QProj` while
layer 3 (full-attention) loaded the reverse. `archFeatureProfile["olmo_hybrid"]` needed
`{FeatDeltaNet, FeatNoPE, FeatPostOnlyNorm, FeatQKNormWhole}` — CPU-only overall despite
`FeatDeltaNet` being GPU-declared (qwen3_5's own backends), since the other three are not.

`parity_manifest.json` row: `status: experimental`, `method: tiny-golden`. Gate registered in
`cmd/gate/parity.go`'s `parityGates` AND `awaitingFirstConfirmation` in the same commit as `olmo3`'s
(F1's CI-break lesson applied without re-learning it, for both keys at once).

### A found-along-the-way bug: `capability-matrix.md`'s "GPU-resident" column believed the wrong gate

Checking `olmo_hybrid`'s own generated row surfaced a pre-existing defect, not something this pass
introduced: `docs/capability-matrix.md`'s "GPU-resident" column was computed from
`arch.decodeRunnerEligible()` ALONE — an arch-SHAPE predicate ("can the generic resident forward
represent this at all") — rather than the full admission truth ("does at least one real backend's
declared feature set actually cover what this arch needs"), which `decoder/features.go`'s own
`ResidentEligible(arch, backend)` already computes and which `hardware-matrix.md`'s generator
already uses. This is the EXACT failure mode the 2026-08-31 LFM2 fix patched for one family
(`decodeRunnerEligible` gained a special-case decline once the matrix was caught reading
"GPU-resident: yes" for a family no backend could run) — except this time it was hitting FIVE
families through undeclared FEATURES rather than an arch-shape incompatibility: `cohere`, `cohere2`,
`mistral3`, `smollm3`, and `olmo3` all read "yes" while their own `admissionGolden` rows are empty
(CPU-only). Fixed by pointing `capability_matrix_test.go`'s `GPUResident` column at the same
`ResidentEligible` gate `hardware-matrix.md` uses, rather than adding a sixth one-off special case —
the two generated docs can no longer disagree about the same family. All six corrected rows (the
five pre-existing plus `olmo_hybrid`) flip from "yes" to "no"; no other row changed.

- **GGUF loader / T3 real-checkpoint parity / peer row.** Same order-of-operations as every family
  in this doc: ship at T1, move on. `olmo3` 7B fits the Mac's T3 budget; 32B and `olmo_hybrid` 7B
  are Linux-box targets.
- **CHANGELOG entry**: added for both `olmo3` and `olmo_hybrid`.

## G4 · `smollm3` — SmolLM3-3B — DONE at T1 (mac, 2026-09-06); real-checkpoint embed deliverable pending on disk space

### Phase 0 (real config.json + real modeling_smollm3.py)

`HuggingFaceTB/SmolLM3-3B`: `model_type: "smollm3"`, `architectures: ["SmolLM3ForCausalLM"]`. Llama
rails — GQA (16 heads / 4 kv, 36 layers, hidden 2048, inter 11008), no QK-norm, no bias, single-base
RoPE (theta 5e6, `rope_scaling: null`), `tie_word_embeddings: true`. `layer_types` is present but is
a RED HERRING: every one of its 36 entries reads `"full_attention"` regardless of which layers are
actually NoPE — checked directly against the real config, not assumed from how Gemma/cohere2 use
the same-named field. `no_rope_layer_interval: 4`, `no_rope_layers` an explicit 36-entry list.

**The field name is the opposite of its own values — verified against the real
`modeling_smollm3.py`, not guessed from the name**: `self.use_rope = config.no_rope_layers[layer_idx]`
— an entry of `1` means the layer HAS RoPE, `0` means NoPE. The real list is `[1,1,1,0]` repeating
(0 at 0-indexed positions 3, 7, 11, ..., 35 — 9 of 36 layers), matching `configuration_smollm3.py`'s
own generation formula for when the list is absent (`(layer_idx+1) % interval != 0`), which
independently confirms both the polarity and the "every 4th layer" pattern the brief itself named.
Getting the polarity backwards would silently flip 27 RoPE layers to NoPE and 9 NoPE layers to
RoPE — correct shapes, plausible logits, wrong model, no crash. **Proved the gate actually catches
this**: flipped the polarity in the adapter and re-ran the T1 test — it failed, but the final-logit
cosine only dropped to 0.999797 (NOT catastrophic), confirming a cosine-only threshold would have
missed it; only the direct per-layer `isNoPELayer` assertions caught it cleanly.

Tensor names confirmed byte-identical to llama (instantiated `SmolLM3ForCausalLM` directly, read
its `state_dict()`) — `llamaTensorSchema` reused verbatim.

### Adapter decision: pure composition, reusing an EXISTING Config field across two unrelated families

`decoder/registry.go`'s `smollm3Architecture` is `llamaArchitecture` plus a `layerNoPE` closure —
the SAME generic `Architecture.layerNoPE` hook `cohere2Architecture` already populates for its own
NoPE layers, reused directly rather than a new mechanism. **`Config.NoRopeLayers` (json
`no_rope_layers`) already existed** — `llama4_text` set it first, with the IDENTICAL "1 = has RoPE,
0 = NoPE" convention (its own comment: `NoRopeLayers[i]==1 ⇒ layer i uses RoPE, ==0 ⇒ NoPE`),
confirmed by reading the field before adding a duplicate. Only one genuinely new field was needed:
`NoRopeLayerInterval`, the fallback-generation divisor for a checkpoint that omits the explicit
list (SmolLM3's own default; llama4_text always ships the full list explicitly so never needed it).

### T1 — tiny-golden, DONE

`scripts/pin_smollm3_tiny.py` (transformers, `SmolLM3ForCausalLM`, 4 layers,
`no_rope_layer_interval=4` giving `no_rope_layers=[1,1,1,0]` — exactly the real release's own
pattern, so the fixture exercises the identical polarity/interval logic, not a config chosen to
avoid triggering it): `TestSmolLM3_forwardParity` — **argmax exact, cosine 0.9999999999999544**,
plus direct per-layer `isNoPELayer` assertions matching `[false,false,false,true]`.

`archFeatureProfile["smollm3"]` = `{FeatNoPE}` — CPU-only, same as `cohere2` (no resident backend
declares `FeatNoPE` at all yet, so this is not a new gap). `parity_manifest.json` row:
`status: experimental`, `method: tiny-golden`. Gate registered in `parityGates` and
`awaitingFirstConfirmation` in the same commit.

### The extra deliverable this key exists for — status

The brief's own framing: "a `pull -embed` build of SmolLM3-3B int4 — record the binary size and
cold-start-to-first-token on the Mac... That number is the README's 'model compiled into the
binary' claim made concrete on a model people know." This needs the real ~6 GB bf16 checkpoint
downloaded to this Mac. **Not done this pass**: this Mac's disk is at 23 GB free (95% full) at
time of writing — tight enough that a 6 GB download plus the quantize/embed/build round-trip is a
real risk on a box this close to full, even though 3B itself is comfortably the right SIZE class
for this machine's 16 GB RAM. Recorded as owed rather than attempted under pressure; the
T1/adapter work above does not depend on it.

### What was deliberately not done

- **T3 real-checkpoint parity** on the real 3B (bf16 ~6 GB) — sized for this Mac per the brief's
  own note, but blocked on the same disk-space constraint as the embed deliverable above. Both are
  the same download; doing one first would set up the other.
- **Peer row vs Ollama** — gated on T3, per this doc's own rule.
- **GGUF loader** — not attempted; not requested by the brief for this family, unlike G1/G3.

## G5 · `bailing_hybrid` (inclusionAI, Ling 3.0) — DONE at T1 (mac, 2026-09-06); no stop condition fired

**Verdict up front, per the brief's own framing: no primitive beyond KDA + MLA + MoE composition
appeared, and the weights are (would be) loadable, so this became a real family** — building
directly on batch 1 F4's Phase 0 and KDA rehearsal (the per-channel-decay delta rule, proven
against `fla-org/flash-linear-attention`'s actual reference at cosine 1.0, maxAbsDiff 2.98e-08).

### Phase 0 refresh — the real checkpoint and its real modeling source, re-verified

F4 already established the shape (MLA + MoE ride `deepseekArchitecture` composition; KDA is the
one new primitive). Building the real adapter surfaced departures F4's Phase-0 pass, scoped to a
rehearsal, hadn't needed to check:

- **`layer_types` is not a `config.json` field for this family at all** — no released checkpoint
  carries it (confirmed on the real `inclusionAI/Ling-3.0-tiny` fetch). The MLA/KDA pattern is
  COMPUTED from `layer_group_size`, replicated exactly from `BailingMoeV3DecoderLayer.__init__`
  (`normalizeBailingLayerTypes`), including a tail-cleanup clause for a layer count that isn't a
  clean multiple of the group size — a detail the brief's own paraphrase omitted.
- **Both mixers are named `self.attention`, not `self.self_attn`** — confirmed from the real
  decoder layer's `__init__` (`self.attention = BailingMoeV3MultiLatentAttention(...)` /
  `BailingMoeV3KimiDeltaAttention(...)`), and MLA's own output projection is `self.dense`, not
  `o_proj`. `mlaParams` gained `AttnPrefix`/`DenseSuffix` overrides (both empty/default for every
  existing DeepSeek family) so `loadDeepseekAttn`/`mlaAttention` stay shared code, not a fork.
- **An optional per-head sigmoid output gate on MLA** (`gated_attention_proj_granularity_type:
  "head_wise"`, `self.g_proj`, applied to the attention context BEFORE `dense`) — structurally the
  same mechanism Laguna's own attention-output gate already ships, but sigmoid-activated where
  Laguna's is softplus (`applySigmoidGateRow`, a sibling to the existing `applyGateRow`, not a
  parameter — the two activations are genuinely different functions). `mlaWeights` gained an
  optional `gProj` field, nil for every non-Bailing MLA family.
- **The MoE spells its own field names**: `num_experts`/`num_experts_per_tok`/
  `num_shared_experts`/`moe_shared_expert_intermediate_size` (the qwen3_moe/nemotron_h
  convention), NOT DeepSeek's own `n_routed_experts`/`n_shared_experts` — a new
  `Config.NumSharedExperts` field, and `bailingHybridArchitecture` reads `cfg.NumExperts` directly
  rather than `cfg.NRoutedExperts`. The router's expert-bias buffer is `expert_bias`, not
  DeepSeek's `e_score_correction_bias` — otherwise BYTE-FOR-BYTE DeepSeek-V3's `noaux_tc` shape
  (sigmoid + bias + group-limited top-k + `routed_scaling_factor`, an ungated shared expert),
  confirmed from `BailingMoeV3Gate`/`SparseMoeBlock`'s own forward, not assumed from the field-name
  similarity.
- **KDA's wrapper has real departures beyond the per-channel decay F4 already proved**: q/k/v are
  three FULLY SEPARATE projections AND three separate depthwise causal convs
  (`self.q_conv1d`/`k_conv1d`/`v_conv1d`, not one combined conv like Gated DeltaNet's), `dt_bias` is
  shaped `[H·head_dim]` (per-channel, matching the decay) rather than Gated DeltaNet's per-head
  `[H]`, and the output gated-RMSNorm is sigmoid-activated (`FusedRMSNormGated(activation=
  'sigmoid')`) where Gated DeltaNet's is SiLU. None of these are new math — every one is a
  parameterization or an extra tensor, reusing `kdaLowerBoundGate`/`kdaRecurrentStep`
  (F4's rehearsal) UNCHANGED for the actual recurrence (`decoder/kda.go`).
- **Norm placement is uniform Pre2 for BOTH mixer kinds** — confirmed directly from
  `BailingMoeV3DecoderLayer.forward`: `input_layernorm` before either mixer, `residual+hidden`,
  `post_attention_layernorm` before the MLP, for both `attention_layer_type` branches alike. Unlike
  Olmo Hybrid (G2), this family needed NO `NormPlacementLinear` — one real hybrid finding that
  didn't repeat the last one.

### Building and instantiating a real checkpoint — blocked, worked around, not skipped

The real `BailingMoeV3ForCausalLM` (`trust_remote_code=True`) cannot be instantiated on this Mac:
`modeling_bailing_moe_v3.py` imports `fla.ops.kda` at module top level, which transitively imports
Triton (`fla/ops/__init__.py` → `fla/ops/abc/chunk.py` → `import triton`) — no Triton wheel exists
for this platform, the same constraint F4's rehearsal already hit and worked around for the KDA
core alone. `scripts/pin_bailing_hybrid_tiny.py` extends that workaround to a full tiny model:
RMSNorm/MLP/Gate/SparseMoeBlock/MLA are reproduced near-verbatim from the real source (none of
them import `fla` at all — confirmed by reading the file), and KDA's wrapper is hand-assembled
around `naive_kda_lowerbound_gate`/`naive_recurrent_kda` (`fla-org`'s own reference, copied
verbatim, MIT — the SAME functions F4's rehearsal already proved match `decoder/kda.go`) instead of
the Triton-only `chunk_kda`/`fused_recurrent_kda` entry points, which are a different
parallelization of the identical recurrence per `fla`'s own naming convention, not a different
model. Tensor attribute names match the real checkpoint's actual tensor names exactly (verified
against real names read earlier in Phase 0), so the resulting fixture loads through the real
`bailingHybridArchitecture` adapter unmodified.

This is weaker evidence than a real-HF-class T1 (every other family in this doc): the WRAPPER
around KDA's core (conv/gates/gated-norm) and the MLA+MoE composition are checked against a
hand-assembled model, not an independently-executed real one — only the recurrence itself has a
genuinely external oracle. Recorded honestly, not glossed over.

### T1 result and its verification

`TestBailingHybrid_forwardParity`: **argmax exact, cosine 0.9999999999999437**, plus direct
assertions that layer 0 (KDA, per `layer_group_size=4`) loaded `kda` weights and no `mla`, and
layer 3 (MLA) the reverse. Given the reduced-strength reference above, ran a NEGATIVE CONTROL
before trusting the pass: deliberately flipped KDA's decay from per-channel back to a per-head
scalar (Gated DeltaNet's own convention) — cosine dropped to **0.93984**, confirming the test
actually discriminates the one genuinely new piece of math rather than passing regardless.

Building the loader also caught a REAL bug, independent of this family's own correctness: adding
`l.kda` to `LayerWeights` without teaching `decoder/serialize.go` about it reproduced the exact
failure class `audit-2026-09-02`'s R3/C-03 already named (LFM2's short-conv weights silently
missing from the `.giw` format, nil-dereferencing in the decode goroutine on the first token) —
caught here by `TestSerializeCensus_noSilentFieldDrop`, which panicked with a nil `*kdaWeights` on
round-trip. Fixed by bumping the `.giw` format to v9 (`kda.go`'s nine tensors, plus `mlaWeights`'
own `gProj` addition, which had the identical gap for any MLA family using the output gate) rather
than retrofitting v6Layer's fixed byte layout, which would have corrupted every already-shipped
v6/v7/v8 file's read.

`archFeatureProfile["bailing_hybrid"]` = `{FeatMLA, FeatMoE, FeatKDA}` — CPU-only overall: FeatMLA/
FeatMoE are ordinary and declared (webgpu), but the new `FeatKDA` (Gated DeltaNet's own
per-head-scalar-decay taxonomy would have been the wrong bucket for a genuinely different
recurrence) is undeclared everywhere. `parity_manifest.json` row: `status: experimental`,
`method: tiny-golden`. Gate registered in `parityGates` and `awaitingFirstConfirmation` in the same
commit.

### What was deliberately not done

- **T3 real-checkpoint parity** on the real Ling-3.0-tiny (7.9B total / 1.3B active — the cheapest
  real-checkpoint validation of any family this batch touched, per F4's own estimate). Not
  attempted: the real HF class still can't run END-TO-END on this Mac (the Triton blocker above
  applies to VALIDATION the same way it did to fixture-building), so a real-checkpoint T3 needs
  either the Linux box (if Triton installs there) or a from-scratch reference forward at real
  scale — a bigger undertaking than this pass's own ceiling.
- **GGUF loader / peer row** — gated on T3, per this doc's own rule.
- **The LoRA'd KDA gate variant (`no_kda_lora: false`) and the plain unbounded decay gate
  (`kda_safe_gate: false`)** — `validateBailingHybrid` refuses both rather than silently
  mis-running an unimplemented variant. Only Ling-3.0-tiny's own released config (`no_kda_lora:
  true`, `kda_safe_gate: true`) is supported; Ling-3.0-flash or a future release using either
  unimplemented variant would need it added, not assumed to already work.
- **KDA's chunked/parallel scan** (`chunk_kda`) for prefill throughput — this family, like
  `qwen3_5_moe`'s own Gated DeltaNet, only implements the sequential form
  (`decoder/deltanet_chunked.go` has no chunked Gated DeltaNet path either, so this isn't a gap
  novel to KDA).
